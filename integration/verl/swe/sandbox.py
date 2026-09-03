# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal client for GKE agent-sandbox Sandboxes used by SWE RL rollouts.

Creates Sandboxes in a namespace excluded from the managed
``secure-sandbox-policy`` (default ``agents-system``) so task images can run
as root, while keeping gVisor, no SA token, and dropped capabilities - see
configs/swe_sandbox_example.yaml for the rationale.

Implementation notes (learned the hard way, see docs/swe_bench_guide.md):
- Write files via exec + base64, never `kubectl cp`: the sandbox drops
  CAP_CHOWN so tar exits nonzero even though content lands.
- The Sandbox controller names the backing pod identically to the Sandbox.
"""

from __future__ import annotations

import base64
import hashlib
import re
import shlex
import threading
import time
from uuid import uuid4

from kubernetes import client as k8s_client  # type: ignore[import-not-found]
from kubernetes import config as k8s_config  # type: ignore[import-not-found]
from kubernetes.client.rest import ApiException  # type: ignore[import-not-found]
from kubernetes.stream import stream as k8s_stream  # type: ignore[import-not-found]

GROUP = "agents.x-k8s.io"
VERSION = "v1alpha1"
PLURAL = "sandboxes"
CONTAINER = "sandbox"

EXEC_TIMEOUT_DEFAULT = 330  # generous vs the 5-minute grading cap


class SandboxError(RuntimeError):
    pass


_thread_local = threading.local()


def get_thread_client(namespace: str = "agents-system") -> SandboxClient:
    """One SandboxClient per (thread, namespace).

    The kubernetes client's GKE exec-plugin auth races when an ApiClient is
    shared across threads (observed as intermittent NoneType decode errors),
    so every worker thread gets its own client.
    """
    cache = getattr(_thread_local, "clients", None)
    if cache is None:
        cache = _thread_local.clients = {}
    if namespace not in cache:
        cache[namespace] = SandboxClient(namespace=namespace)
    return cache[namespace]


def make_name(prefix: str, instance_id: str) -> str:
    """Unique RFC-1123 sandbox name for one use (trajectory, grading, ...)."""
    slug = re.sub(r"[^a-z0-9-]", "-", instance_id.lower()).strip("-")[:30]
    digest = hashlib.sha1(instance_id.encode()).hexdigest()[:4]  # noqa: S324
    return f"{prefix}-{slug}-{digest}-{uuid4().hex[:6]}"


class SandboxClient:
    def __init__(self, namespace: str = "agents-system") -> None:
        try:
            k8s_config.load_incluster_config()
        except Exception:  # noqa: BLE001
            k8s_config.load_kube_config()
        self.namespace = namespace
        self.custom = k8s_client.CustomObjectsApi()
        self.core = k8s_client.CoreV1Api()

    def create(  # noqa: PLR0913
        self,
        name: str,
        image: str,
        *,
        cpu_request: str = "500m",
        memory_request: str = "1Gi",
        cpu_limit: str = "2",
        memory_limit: str = "4Gi",
    ) -> None:
        manifest = {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "Sandbox",
            "metadata": {"name": name, "namespace": self.namespace},
            "spec": {
                "podTemplate": {
                    "spec": {
                        "runtimeClassName": "gvisor",
                        "automountServiceAccountToken": False,
                        "nodeSelector": {"sandbox.gke.io/runtime": "gvisor"},
                        "tolerations": [
                            {
                                "key": "sandbox.gke.io/runtime",
                                "operator": "Equal",
                                "value": "gvisor",
                                "effect": "NoSchedule",
                            }
                        ],
                        "containers": [
                            {
                                "name": CONTAINER,
                                "image": image,
                                "command": ["sleep", "infinity"],
                                "securityContext": {"capabilities": {"drop": ["ALL"]}},
                                "resources": {
                                    "requests": {"cpu": cpu_request, "memory": memory_request},
                                    "limits": {"cpu": cpu_limit, "memory": memory_limit},
                                },
                            }
                        ],
                    }
                }
            },
        }
        self.custom.create_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL,
            manifest)

    def wait_ready(self, name: str, timeout: float = 300) -> float:
        """Block until the backing pod is Running; returns elapsed seconds."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                pod = self.core.read_namespaced_pod(name, self.namespace)
                if pod.status.phase == "Running":
                    return time.monotonic() - start
                if pod.status.phase in ("Failed", "Succeeded"):  # noqa: PLR6201
                    raise SandboxError(f"{name}: pod reached terminal phase {pod.status.phase}")
            except ApiException as e:
                if e.status != 404:  # pod not created yet  # noqa: PLR2004
                    raise
            time.sleep(3)
        raise SandboxError(f"{name}: not Running after {timeout}s")

    def exec(
        self, name: str, cmd: str, timeout: float = EXEC_TIMEOUT_DEFAULT, retries: int = 2
    ) -> tuple[int, str]:
        """Run a shell command; returns (returncode, combined output). rc 124 on timeout.

        Retries the whole exec on transient websocket errors (the kubernetes
        stream client intermittently dies with e.g. NoneType frame data under
        concurrency) - callers' commands must therefore be idempotent.
        """
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self._exec_once(name, cmd, timeout)
            except Exception as e:  # noqa: BLE001,PERF203 - websocket layer raises bare Exceptions
                last = e
                time.sleep(2 * (attempt + 1))
        raise SandboxError(f"{name}: exec failed after {retries + 1} attempts: {last}")

    def _exec_once(self, name: str, cmd: str, timeout: float) -> tuple[int, str]:
        resp = k8s_stream(
            self.core.connect_get_namespaced_pod_exec,
            name,
            self.namespace,
            container=CONTAINER,
            command=["/bin/sh", "-c", cmd],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        chunks: list[str] = []
        deadline = time.monotonic() + timeout
        while resp.is_open() and time.monotonic() < deadline:
            resp.update(timeout=1)
            if resp.peek_stdout():
                chunks.append(resp.read_stdout())
            if resp.peek_stderr():
                chunks.append(resp.read_stderr())
        if resp.is_open():
            resp.close()
            return 124, "".join(chunks)
        try:
            rc = resp.returncode
        except Exception:  # noqa: BLE001
            rc = -1
        return rc if rc is not None else -1, "".join(chunks)

    # Exec commands travel in the API request; chunk large payloads to stay
    # comfortably under apiserver URL limits.
    _WRITE_CHUNK = 24 * 1024

    def write_file(self, name: str, path: str, content: str | bytes) -> None:
        """Write file content via exec + base64 (CAP_CHOWN-safe, quote-safe)."""
        data = content.encode() if isinstance(content, str) else content
        quoted = shlex.quote(path)
        chunks = [data[i : i + self._WRITE_CHUNK] for i in range(0, len(data),
            self._WRITE_CHUNK)] or [b""]
        for i, chunk in enumerate(chunks):
            b64 = base64.b64encode(chunk).decode()
            prefix = f"mkdir -p $(dirname {quoted}) && " if i == 0 else ""
            op = ">" if i == 0 else ">>"
            rc, out = self.exec(name, f"{prefix}echo '{b64}' | base64 -d {op} {quoted}")
            if rc != 0:
                raise SandboxError(f"{name}: write_file({path}) chunk {i} rc={rc}: {out[-500:]}")

    def delete(self, name: str) -> None:
        try:
            self.custom.delete_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL,
                name)
        except ApiException as e:
            if e.status != 404:  # noqa: PLR2004
                raise
