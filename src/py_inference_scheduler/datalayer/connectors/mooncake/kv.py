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

import os

ENABLE_ENV = "ENABLE_MOONCAKE_KV"

# vLLM workers import the connector from this path.
_CONNECTOR_MODULE = "py_inference_scheduler.datalayer.connectors.mooncake.decode_save"

# Default when MOONCAKE_CONFIG_PATH is not set
_DEFAULT_CONFIG_PATH = "/etc/mooncake/mooncake_config.json"


def mooncake_enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1"


def mooncake_engine_kwargs() -> dict:
    # sha256_cbor is stable across Python versions and sha isn't.
    return {
        "kv_transfer_config": {
            "kv_connector": "DecodeKVSavingConnector",
            "kv_connector_module_path": _CONNECTOR_MODULE,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"save_decode_kv": True},
        },
        "prefix_caching_hash_algo": "sha256_cbor",
    }


def mooncake_env_vars() -> dict:
    return {
        # Seeds vLLM's block-hash chain; must match on every replica.
        "PYTHONHASHSEED": "0",
        "MOONCAKE_CONFIG_PATH": os.environ.get(
            "MOONCAKE_CONFIG_PATH", _DEFAULT_CONFIG_PATH
        ),
        # Error out if no RDMA verbs device exists instead of silent TCP.
        "MC_FORCE_HCA": "1",
        # GKE COS nodes have no nvidia-peermem module; GPU KV registration must
        # go through the dmabuf path (ibv_reg_dmabuf_mr). Override on hosts that
        # do have peermem.
        "WITH_NVIDIA_PEERMEM": os.environ.get("WITH_NVIDIA_PEERMEM", "0"),
    }
