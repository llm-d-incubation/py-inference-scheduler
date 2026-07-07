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

import time
from typing import Any, Sequence

from py_inference_scheduler.framework import (
    Endpoint,
    FlowControlPlugin,
    LLMRequest,
    register_flow_control,
)


@register_flow_control("kv_saturation")
class KVSaturationPlugin(FlowControlPlugin):
    def __init__(self, **config: Any) -> None:  # noqa: ANN401
        self.config = config
        # rollout_request_id -> {isl, max_osl}
        self._rollout_request_stats: dict[str, dict[str, int]] = {}
        # replica_id -> token_usage_at_replica
        self._replica_token_usage: dict[str, int] = {}
        # request_id -> (replica_id, tokens)
        self._request_at_replica: dict[str, tuple[str, int]] = {}
        # request_id -> set of budgeted requests
        self._budgeted_requests: set[str] = set()
        self._last_drip_at = 0.0

    def get_allowed_candidates(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> Sequence[Endpoint]:
        fc = self.config

        if request.request_id in self._budgeted_requests:
            return candidates

        rollout_id, char_len = self._get_rollout_request_id(request.body)
        tokens_required = self._estimate_tokens_required(rollout_id, char_len, fc)

        allowed = []
        for c in candidates:
            kv_cache_size = c.attributes.get("kv_cache_size", -1)
            if kv_cache_size <= 0:  # type: ignore[operator]
                continue

            current_usage = self._replica_token_usage.get(c.name, 0)
            if current_usage + tokens_required <= kv_cache_size:  # type: ignore[operator]
                allowed.append(c)

        if allowed:
            return allowed

        return self._get_drip_candidates(candidates, fc)

    def _get_drip_candidates(self, candidates: Sequence[Endpoint], fc: dict) -> Sequence[Endpoint]:
        enable_drip = fc.get("enable_drip", False)
        if not enable_drip:
            return []

        drip_threshold_kv = fc.get("drip_threshold_kv", 0.1)
        now = time.time()
        drip_interval_s = fc.get("drip_interval_s", 2.0)
        if (now - self._last_drip_at) >= drip_interval_s:
            for c in candidates:
                routing_stats = c.attributes.get("routing_stats", {})
                physical_kv = routing_stats.get("kv", 1.0)  # type: ignore[attr-defined]
                if physical_kv < drip_threshold_kv:
                    self._last_drip_at = now
                    print(f"[BUDGET] Drip Admission to {c.name} (Physical KV: {physical_kv:.2f})")
                    return [c]
        return []

    def reserve(self, request: LLMRequest, selected: Endpoint) -> None:
        fc = self.config

        if request.request_id in self._budgeted_requests:
            return

        rollout_id, char_len = self._get_rollout_request_id(request.body)
        tokens_required = self._estimate_tokens_required(rollout_id, char_len, fc)

        self._replica_token_usage[selected.name] = (
            self._replica_token_usage.get(selected.name, 0) + tokens_required
        )
        self._request_at_replica[request.request_id] = (
            selected.name,
            tokens_required,
        )
        self._budgeted_requests.add(request.request_id)
        print(
            f"[BUDGET] Admission committed for req={request.request_id} to replica={selected.name} "
            f"(New Usage: {self._replica_token_usage[selected.name]})"
        )

    def release(self, request: LLMRequest, endpoint_name: str) -> None:
        if request.request_id in self._request_at_replica:
            replica_id_to_reclaim, tokens = self._request_at_replica.pop(request.request_id)
            self._replica_token_usage[replica_id_to_reclaim] = max(
                0,
                self._replica_token_usage.get(replica_id_to_reclaim, 0) - tokens,
            )
            print(
                f"[BUDGET] Released req={request.request_id} from replica={replica_id_to_reclaim} "
                f"(New Usage: {self._replica_token_usage[replica_id_to_reclaim]})"
            )
            self._budgeted_requests.discard(request.request_id)

    def update_learned_stats(self, rollout_request_id: str, isl: int, osl: int) -> None:
        self._rollout_request_stats[rollout_request_id] = {
            "isl": isl,
            "osl": osl,
        }
        print(f"[BUDGET] Learned stats for rollout={rollout_request_id}: ISL={isl}, OSL={osl}")

    def _get_rollout_request_id(self, body: Any) -> tuple[str, int]:  # noqa: ANN401
        import struct
        import uuid

        NAMESPACE_FOR_UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "llmd.inference.scheduler")  # noqa: N806

        if not body:
            return "", 0

        try:
            if isinstance(body, list) and len(body) > 0 and isinstance(body[0], int):
                prompt_bytes = struct.pack(f"{len(body)}i", *body)
                return (
                    uuid.uuid5(NAMESPACE_FOR_UUID, prompt_bytes.hex()).hex,
                    len(body) * 4,
                )

            if isinstance(body, (str, bytes)):
                prompt_str = body if isinstance(body, str) else body.hex()
                return (
                    uuid.uuid5(NAMESPACE_FOR_UUID, prompt_str).hex,
                    len(body),
                )

            if isinstance(body, list):
                extracted_text = ""
                for m in body:
                    if isinstance(m, dict):
                        extracted_text += str(m.get("content", ""))
                    else:
                        extracted_text += str(getattr(m, "content", ""))
                char_len = len(extracted_text)
                return (
                    uuid.uuid5(NAMESPACE_FOR_UUID, extracted_text).hex,
                    char_len,
                )

        except Exception as e:  # noqa: BLE001
            print(f"[PLUGIN ERROR] Failed to parse request ID securely: {e}")

        return "", 0

    def _estimate_tokens_required(self, rollout_id: str, char_len: int, fc: dict) -> int:
        stats = self._rollout_request_stats.get(rollout_id)
        if stats:
            return stats["isl"] + stats["osl"]

        default_osl: int = fc.get("default_osl", 1024)
        return (char_len // 4) + default_osl
