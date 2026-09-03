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
"""Preflight checks for the SWE-bench RL pipeline (verl + GKE agent-sandbox).

Validates, against the current kubectl context, everything a training run
needs — each check encodes a failure we actually hit (see
docs/swe_bench_guide.md):

  sandbox env   CRDs, gVisor nodes, policy-exempt namespace, RBAC, and a live
                create->exec->delete smoke test with a real task image
                (validates mirror pull-through + admission-policy shape)
  ray cluster   head/worker pods healthy, verl importable at ONE commit on
                every pod (replacement pods come up blank), W&B key present
  data          parquets on EVERY replica with matching row counts and the
                full extra_info schema (emptyDirs are per-pod)
  hygiene       stray sandboxes, worker disk, repo-side files for the run

Runs from a workstation; needs only kubectl (+ pandas locally). Exit 0 iff
no FAIL. Usage:

    uv run --with pandas --with pyarrow python -m integration.verl.swe.preflight
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess  # noqa: S404
import sys
import time

RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)

REQUIRED_EXTRA_INFO_KEYS = {
    "instance_id", "dataset_kind", "docker_image", "repo", "base_commit",
    "expected_output_json", "fail_to_pass", "pass_to_pass", "test_patch",
    "num_non_test_lines",
}

SMOKE_SANDBOX = "preflight-smoke"


def record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    """Return (rc, stdout) - stderr only when stdout is empty.

    kubectl exec prints 'Defaulted container ...' on stderr for multi-container
    pods, which must not pollute parsed output.
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603
        return p.returncode, (p.stdout.strip() or p.stderr.strip())
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def kubectl(args: list[str], timeout: int = 60) -> tuple[int, str]:
    return run(["kubectl", *args], timeout=timeout)


def pod_exec(
    pod: str, script: str, timeout: int = 120, namespace: str = "default"
) -> tuple[int, str]:
    return kubectl(["exec", "-n", namespace, pod, "--", "sh", "-c", script], timeout=timeout)


def check_sandbox_environment(args) -> None:
    rc, out = kubectl(["get", "crd", "sandboxes.agents.x-k8s.io", "-o", "name"])
    if rc != 0:
        record("FAIL", "sandbox CRD installed", out[-150:] or "agent-sandbox not installed")
        return  # everything downstream needs the CRD
    record("PASS", "sandbox CRD installed", "sandboxes.agents.x-k8s.io")

    rc, out = kubectl(["get", "nodes", "-l", "sandbox.gke.io/runtime=gvisor", "--no-headers"])
    n_nodes = len([line for line in out.splitlines() if line.strip()]) if rc == 0 else 0
    record("PASS" if n_nodes else "FAIL", "gVisor node pool", f"{n_nodes} nodes ready")

    rc, out = kubectl(["get", "namespace", args.namespace, "-o", "name"])
    record("PASS" if rc == 0 else "FAIL", "sandbox namespace exists", args.namespace)

    # NB: pods/exec verb is "get" - the kubernetes python client's exec uses a
    # GET websocket upgrade (kubectl's own exec uses create/POST instead).
    checks = [("create", "sandboxes.agents.x-k8s.io"), ("get", "pods/exec"), ("get", "pods")]
    for verb, resource in checks:
        rc, out = kubectl([
            "auth", "can-i", verb, resource, "-n", args.namespace,
            "--as", f"system:serviceaccount:{args.ray_namespace}:{args.ray_service_account}",
        ])
        ok = rc == 0 and "yes" in out
        record("PASS" if ok else "FAIL", f"RBAC: ray SA can {verb} {resource}",
               "" if ok else "apply configs/swe_sandbox_rbac.yaml")


def check_ray_cluster(args) -> list[str]:
    """Returns [head, worker, worker...] pod names; empty on failure."""
    rc, out = kubectl(["get", "pods", "-n", args.ray_namespace, "-l", "ray.io/node-type=head",
                       "-o", "jsonpath={.items[*].metadata.name}"])
    heads = out.split() if rc == 0 else []
    rc, out = kubectl(["get", "pods", "-n", args.ray_namespace, "-l", "ray.io/node-type=worker",
                       "-o", "jsonpath={.items[*].metadata.name}"])
    workers = out.split() if rc == 0 else []
    if not heads:
        record("FAIL", "ray head pod", "no pod with label ray.io/node-type=head")
        return []
    record("PASS", "ray pods", f"head={heads[0]}, {len(workers)} workers")

    pods = [heads[0], *workers]
    commits: dict[str, str] = {}
    for pod in pods:
        rc, out = pod_exec(
            pod,
            "python3 -c 'import verl' 2>/dev/null && "
            "(cd /tmp/verl/verl 2>/dev/null && git rev-parse --short HEAD) || echo NO_VERL",
            namespace=args.ray_namespace,
        )
        commits[pod] = out.splitlines()[-1] if out else "NO_VERL"
    missing = [p for p, c in commits.items() if "NO_VERL" in c]
    distinct = {c for c in commits.values() if "NO_VERL" not in c}
    if missing:
        record("FAIL", "verl on all ray pods",
               f"missing on {missing} - provision per guide 'pets to cattle' snippet")
    elif len(distinct) > 1:
        record("FAIL", "verl commit consistent", f"mismatched commits: {commits}")
    else:
        record("PASS", "verl on all ray pods", f"commit {distinct.pop()} on {len(pods)} pods")

    rc, out = pod_exec(
        heads[0], '[ -n "$WANDB_API_KEY" ] && echo yes || echo no', namespace=args.ray_namespace
    )
    record("PASS" if "yes" in out else "WARN", "WANDB_API_KEY on head pod",
           "" if "yes" in out else "wandb logging will fail; metrics still in console logs")

    rc, out = kubectl(
        ["get", "configmap", "scheduler-config", "-n", args.ray_namespace, "-o", "name"]
    )
    record("PASS" if rc == 0 else "WARN", "scheduler-config ConfigMap",
           "" if rc == 0 else "needed for the scheduler (treatment) arm only")
    return pods


def check_data(args, pods: list[str]) -> str | None:
    """Returns a task image ref from the head parquet for the smoke test."""
    probe = (
        "python3 - <<'EOF'\n"
        "import pandas as pd, json\n"
        f"files = {json.dumps([args.train_file, args.calibrated_file, args.eval_file])}\n"
        "out = {}\n"
        "for f in files:\n"
        "    try:\n"
        "        df = pd.read_parquet(f, columns=['extra_info'])\n"
        "        keys = set(df.iloc[0]['extra_info'].keys())\n"
        "        out[f] = {'rows': len(df), 'keys': sorted(keys),\n"
        "                  'image': df.iloc[0]['extra_info']['docker_image']}\n"
        "    except Exception as e:\n"
        "        out[f] = {'error': f'{type(e).__name__}: {e}'[:120]}\n"
        "print(json.dumps(out))\n"
        "EOF"
    )
    reports = {}
    for pod in pods:
        _, out = pod_exec(pod, probe, timeout=180, namespace=args.ray_namespace)
        try:
            reports[pod] = json.loads(out.splitlines()[-1])
        except Exception:  # noqa: BLE001
            reports[pod] = {"error": out[-150:]}
            record("FAIL", f"data probe on {pod}", out[-150:])
            continue

    smoke_image = None
    head = pods[0]
    for f in [args.train_file, args.calibrated_file, args.eval_file]:
        per_pod = {p: reports.get(p, {}).get(f, {}) for p in pods}
        errors = {p: r.get("error", "no probe result") for p, r in per_pod.items()
                  if not isinstance(r, dict) or "rows" not in r}
        if errors:
            record("FAIL", f"parquet {f}",
                   f"unreadable on {list(errors)}: {next(iter(errors.values()))}")
            continue
        rows = {r["rows"] for r in per_pod.values()}
        if len(rows) != 1:
            record("FAIL", f"parquet {f} row counts match",
                   str({p: r["rows"] for p, r in per_pod.items()}))
            continue
        missing_keys = REQUIRED_EXTRA_INFO_KEYS - set(per_pod[head]["keys"])
        if missing_keys:
            record("FAIL", f"parquet {f} schema",
                   f"missing extra_info keys: {sorted(missing_keys)} - rerun preprocessing")
            continue
        record("PASS", f"parquet {f}", f"{rows.pop()} rows on all {len(pods)} pods, schema OK")
        # Any readable parquet can source the smoke-test image; prefer the
        # calibrated file but don't let its absence skip the smoke test.
        if smoke_image is None or f == args.calibrated_file:
            smoke_image = per_pod[head]["image"]
    return smoke_image


def check_sandbox_smoke(args, image: str) -> None:
    manifest = {
        "apiVersion": "agents.x-k8s.io/v1alpha1",
        "kind": "Sandbox",
        "metadata": {"name": SMOKE_SANDBOX, "namespace": args.namespace},
        "spec": {"podTemplate": {"spec": {
            "runtimeClassName": "gvisor",
            "automountServiceAccountToken": False,
            "nodeSelector": {"sandbox.gke.io/runtime": "gvisor"},
            "tolerations": [{"key": "sandbox.gke.io/runtime", "operator": "Equal",
                             "value": "gvisor", "effect": "NoSchedule"}],
            "containers": [{"name": "sandbox", "image": image,
                            "command": ["sleep", "infinity"],
                            "securityContext": {"capabilities": {"drop": ["ALL"]}},
                            "resources": {"requests": {"cpu": "250m", "memory": "1Gi"},
                                          "limits": {"cpu": "2", "memory": "4Gi"}}}],
        }}},
    }
    kubectl(["delete", "sandbox", SMOKE_SANDBOX, "-n", args.namespace, "--ignore-not-found"])
    p = subprocess.run(["kubectl", "apply", "-f", "-"], input=json.dumps(manifest),  # noqa: PLW1510,S607
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        record("FAIL", "sandbox smoke: create", p.stderr.strip()[-200:])
        return
    try:
        start = time.monotonic()
        phase = ""
        while time.monotonic() - start < args.smoke_timeout:
            _, phase = kubectl(["get", "pod", SMOKE_SANDBOX, "-n", args.namespace,
                                 "-o", "jsonpath={.status.phase}"])
            if phase == "Running":
                break
            time.sleep(5)
        if phase != "Running":
            record("FAIL", "sandbox smoke: pod Running",
                   f"phase={phase or 'none'} after {args.smoke_timeout}s "
                   "(capacity? image pull? policy?)")
            return
        elapsed = time.monotonic() - start
        _, out = kubectl(
            ["exec", "-n", args.namespace, SMOKE_SANDBOX, "--",
             "sh", "-c", "[ -d /testbed ] && echo TESTBED_OK || echo NO_TESTBED"],
            timeout=60,
        )
        if "TESTBED_OK" in out:
            record("PASS", "sandbox smoke",
                   f"create->Running {elapsed:.0f}s, exec OK, "
                   f"/testbed present ({image.rsplit('/', 1)[-1][:40]})")
        else:
            record("FAIL", "sandbox smoke: exec/layout", out[-150:])
    finally:
        kubectl(["delete", "sandbox", SMOKE_SANDBOX, "-n", args.namespace, "--ignore-not-found"])


def check_hygiene(args, pods: list[str]) -> None:
    rc, out = kubectl(["get", "sandboxes", "-n", args.namespace, "--no-headers"])
    strays = [line.split()[0] for line in out.splitlines()
              if line.strip() and "No resources found" not in line
              and line.split()[0] != SMOKE_SANDBOX] if rc == 0 else []
    record("PASS" if not strays else "WARN", "no stray sandboxes",
           "" if not strays else f"{len(strays)} lingering (capacity leak): {strays[:3]}...")

    for pod in pods[1:]:
        rc, out = pod_exec(
            pod, "df -BG / | tail -1 | awk '{print $4}'", namespace=args.ray_namespace
        )
        try:
            free_gb = int(out.strip().rstrip("G"))
            record("PASS" if free_gb > 100 else "WARN", f"disk on {pod}", f"{free_gb}G free")  # noqa: PLR2004
        except ValueError:
            record("WARN", f"disk on {pod}", out[-80:])

    for path in ["integration/verl/examples/swe_agent_loop.yaml",
                 "integration/verl/examples/run_swe.sh",
                 "integration/verl/examples/runtime-env-swe.yaml",
                 "configs/swe_sandbox_rbac.yaml"]:
        record("PASS" if pathlib.Path(path).exists() else "FAIL", f"repo file {path}",
               "" if pathlib.Path(path).exists() else "run from the repo root?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="agents-system",
                        help="sandbox namespace (policy-exempt)")
    parser.add_argument("--ray-namespace", default="default")
    parser.add_argument("--ray-service-account", default="default")
    parser.add_argument("--data-dir", default="/home/ray/data/swe")
    parser.add_argument("--smoke-timeout", type=int, default=300)
    parser.add_argument("--skip-smoke", action="store_true",
                        help="skip the live sandbox create/exec test")
    args = parser.parse_args()
    args.train_file = f"{args.data_dir}/train.parquet"
    args.calibrated_file = f"{args.data_dir}/train_calibrated.parquet"
    args.eval_file = f"{args.data_dir}/test.parquet"

    check_sandbox_environment(args)
    pods = check_ray_cluster(args)
    smoke_image = check_data(args, pods) if pods else None
    if smoke_image and not args.skip_smoke:
        check_sandbox_smoke(args, smoke_image)
    elif not args.skip_smoke:
        record("FAIL", "sandbox smoke", "skipped: no task image available from calibrated parquet")
    check_hygiene(args, pods)

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"\npreflight: {len(RESULTS) - len(fails) - len(warns)} pass, "
          f"{len(warns)} warn, {len(fails)} fail")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
