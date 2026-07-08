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

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

import pytest

pytest.importorskip("httpx")  # FastAPI's TestClient is built on httpx
from fastapi.testclient import TestClient

from integration.vime.server import create_app
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig

_METRICS = (
    "# TYPE vllm:num_requests_running gauge\n"
    'vllm:num_requests_running{model_name="m"} 0.0\n'
    "# TYPE vllm:num_requests_waiting gauge\n"
    'vllm:num_requests_waiting{model_name="m"} 0.0\n'
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="m"} 0.1\n'
)


def _scheduler():
    config = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "backpressure": {
                "scorers": [
                    {"type": "waiting_queue", "weight": 5.0},
                    {"type": "least_queue", "weight": 2.0},
                    {"type": "kv_cache", "weight": 1.0},
                    {"type": "prefix_cache", "weight": 1.0},
                ],
                "picker": {"type": "max_score"},
            }
        },
    }
    return Scheduler.new_with_config(SchedulerConfig.from_dict(config))


def _make_handler(worker_id):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/metrics":
                self._send(200, _METRICS.encode(), "text/plain")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            if self.path == "/inference/v1/generate":
                body = json.dumps({"worker": worker_id, "choices": [{"token_ids": [1]}]}).encode()
                self._send(200, body, "application/json")
            else:
                self._send(404, b"", "text/plain")

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class StubWorker:
    def __init__(self, worker_id):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(worker_id))
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def test_registry_endpoints_roundtrip_delete_by_url():
    with TestClient(create_app(_scheduler())) as client:
        assert client.get("/workers").json() == {"workers": []}

        url = "http://127.0.0.1:9"
        resp = client.post("/workers", json={"url": url, "worker_type": "regular"})
        assert resp.status_code == 200
        wid = resp.json()["id"]

        listed = client.get("/workers").json()["workers"]
        assert listed == [{"url": url, "id": wid, "worker_type": "regular"}]
        assert client.get("/list_workers").json() == {"urls": [url]}

        # vime deregisters by the url-encoded worker url, not by our id.
        assert client.delete(f"/workers/{quote(url, safe='')}").status_code == 200
        assert client.get("/workers").json() == {"workers": []}
        assert client.delete(f"/workers/{quote(url, safe='')}").status_code == 404


def test_add_worker_requires_url():
    with TestClient(create_app(_scheduler())) as client:
        assert client.post("/workers", json={}).status_code == 400


def test_generate_with_no_workers_returns_503():
    with TestClient(create_app(_scheduler())) as client:
        resp = client.post("/inference/v1/generate", json={"token_ids": [1, 2, 3]})
        assert resp.status_code == 503


def test_generate_proxies_and_keeps_prefix_affinity():
    with StubWorker("a") as a, StubWorker("b") as b, TestClient(create_app(_scheduler())) as client:
        client.post("/workers", json={"url": a.url})
        client.post("/workers", json={"url": b.url})

        # A long-enough prompt so the prefix scorer engages; identical prompts
        # should stick to the same worker once its prefix is recorded.
        payload = {"token_ids": list(range(64)), "sampling_params": {"max_tokens": 1}}
        first = client.post("/inference/v1/generate", json=payload)
        second = client.post("/inference/v1/generate", json=payload)

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["worker"] in {"a", "b"}
        assert first.json()["worker"] == second.json()["worker"]
