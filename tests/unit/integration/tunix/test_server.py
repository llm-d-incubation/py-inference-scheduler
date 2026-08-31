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

import pytest

pytest.importorskip("httpx")  # FastAPI's TestClient is built on httpx
from fastapi.testclient import TestClient

from integration.tunix.server import create_app
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig

# Long enough for the prefix_cache scorer to produce at least one 64-byte block.
_PROMPT = "Solve the following math problem step by step and be careful: " * 5


def _scheduler(config=None):
    config = config or {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "rl_rollout": {
                "filters": [{"type": "worker_state"}],
                "scorers": [
                    {"type": "waiting_queue", "weight": 5.0},
                    {"type": "least_queue", "weight": 2.0},
                ],
                "picker": {"type": "max_score"},
            }
        },
    }
    return Scheduler.new_with_config(SchedulerConfig.from_dict(config))


def _candidate(name, state="READY", waiting=0, running=0, queue_len=0):
    return {
        "name": name,
        "attributes": {
            "state": state,
            "queue_len": queue_len,
            "routing_stats": {
                "num_waiting_reqs": waiting,
                "num_running_reqs": running,
                "kv": 0.0,
            },
        },
    }


def _schedule(client, candidates, prompt=_PROMPT, request_id="r1"):
    return client.post(
        "/schedule",
        json={"request_id": request_id, "prompt": prompt, "candidates": candidates},
    )


def test_healthz():
    with TestClient(create_app(_scheduler())) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_picks_least_loaded_worker():
    with TestClient(create_app(_scheduler())) as client:
        resp = _schedule(
            client,
            [
                _candidate("busy", waiting=8, running=4, queue_len=12),
                _candidate("idle", waiting=0, running=0, queue_len=0),
            ],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["picked"] == "idle"
        assert body["fallback"] is False
        assert "idle" in body["scores"]


def test_syncing_worker_excluded():
    with TestClient(create_app(_scheduler())) as client:
        resp = _schedule(
            client,
            [
                # Idle but mid weight-sync: must not be picked.
                _candidate("syncing", state="SYNCING", waiting=0, queue_len=0),
                _candidate("ready", state="READY", waiting=5, queue_len=5),
            ],
        )
        assert resp.json()["picked"] == "ready"


def test_all_syncing_still_picks_someone():
    with TestClient(create_app(_scheduler())) as client:
        resp = _schedule(
            client,
            [
                _candidate("a", state="SYNCING", waiting=3),
                _candidate("b", state="SYNCING", waiting=0),
            ],
        )
        body = resp.json()
        assert body["picked"] in {"a", "b"}
        assert body["fallback"] is False


def test_missing_candidates_is_400():
    with TestClient(create_app(_scheduler())) as client:
        assert client.post("/schedule", json={"prompt": "hi"}).status_code == 400
        assert client.post("/schedule", json={"candidates": []}).status_code == 400
        assert (
            _schedule(client, [{"attributes": {}}]).status_code == 400
        )  # candidate without a name


def test_prefix_affinity_and_reset():
    # Prefix dominates load, so repeated prompts stick to the first winner
    # until /reset drops the affinity and load takes over.
    config = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "rl_rollout": {
                "scorers": [
                    {"type": "prefix_cache", "weight": 10.0},
                    {"type": "waiting_queue", "weight": 1.0},
                ],
                "picker": {"type": "max_score"},
            }
        },
    }
    with TestClient(create_app(_scheduler(config))) as client:
        idle = [_candidate("a", waiting=0), _candidate("b", waiting=0)]
        first = _schedule(client, idle, request_id="r1").json()["picked"]
        other = "b" if first == "a" else "a"

        # Same prompt, winner now heavily loaded: prefix affinity still wins.
        loaded = [
            _candidate(first, waiting=100),
            _candidate(other, waiting=0),
        ]
        assert _schedule(client, loaded, request_id="r2").json()["picked"] == first

        resp = client.post("/reset")
        assert resp.status_code == 200
        assert resp.json()["reset_scorers"] >= 1

        # Affinity gone: load decides.
        assert _schedule(client, loaded, request_id="r3").json()["picked"] == other
