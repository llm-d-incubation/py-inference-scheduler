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

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("httpx")  # FastAPI's TestClient is built on httpx
from fastapi.testclient import TestClient

from integration.slime.server import create_app
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig

_METRICS = (
    'sglang:num_running_reqs{model_name="m"} 0.0\n'
    'sglang:num_queue_reqs{model_name="m"} 0.0\n'
    'sglang:token_usage{model_name="m"} 0.1\n'
)


def _least_queue_scheduler() -> Scheduler:
    config = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "lq": {
                "scorers": [{"type": "least_queue", "weight": 1.0}],
                "picker": {"type": "max_score"},
            }
        },
    }
    return Scheduler.new_with_config(SchedulerConfig.from_dict(config))


class GatedWorker:
    """A stub worker that parks every /generate until released, so a request stays in flight."""

    def __init__(self, worker_id: str) -> None:
        self.arrived = threading.Event()
        self.gate = threading.Event()
        arrived, gate = self.arrived, self.gate

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                self._send(200 if self.path == "/metrics" else 404, _METRICS.encode())

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path != "/generate":
                    self._send(404, b"")
                    return
                arrived.set()
                gate.wait(timeout=10)
                self._send(200, json.dumps({"worker": worker_id, "meta_info": {}}).encode())

            def _send(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.gate.set()
        self.server.shutdown()
        self.server.server_close()


def test_dispatch_is_visible_to_the_next_decision_before_the_poller_ticks():
    """A second request arriving inside one poll interval must see the first dispatch.

    queue_len used to be republished only by the metrics poller, so every
    decision within one interval scored the same snapshot and a burst herded
    onto one worker (max_score breaks ties on the first endpoint). The poller
    is set far apart so only the dispatch-time publish can tell the second
    decision that the first worker is busy.
    """
    app = create_app(_least_queue_scheduler(), metrics_refresh_ms=60_000)
    with GatedWorker("a") as a, GatedWorker("b") as b, TestClient(app) as client:
        client.post("/workers", json={"url": a.url})
        client.post("/workers", json={"url": b.url})
        payload = {"input_ids": [1, 2, 3]}
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post, "/generate", json=payload)
            assert a.arrived.wait(timeout=10), "first request never reached worker a"
            second = pool.submit(client.post, "/generate", json=payload)
            assert b.arrived.wait(timeout=10), "second request herded onto the busy worker"
            a.gate.set()
            b.gate.set()
            workers = {first.result().json()["worker"], second.result().json()["worker"]}
        assert workers == {"a", "b"}
