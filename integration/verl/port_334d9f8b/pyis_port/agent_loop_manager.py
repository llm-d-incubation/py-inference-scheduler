"""AgentLoopManager wiring PyInferenceLLMClient into verl commit 334d9f8b.

Reuses LlmdBaseAgentLoopManager (from the llm-d-rl integration package, which
handles the retrieve-servers / swap-client lifecycle on this verl commit) and
its proven "vllm_server_{rank}_0" actor-name lookup.

Wire in via hydra:
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=pyis_port.agent_loop_manager.PyInferenceAgentLoopManager

Required runtime env vars (pass via +ray_kwargs.ray_init.runtime_env.env_vars.*):
  PYTHONPATH        - must include the py-inference-scheduler repo root and this package's parent
  ROUTER_CONFIG_PATH - scheduler config, e.g. /etc/scheduler/scheduler.yaml
"""

from __future__ import annotations

import logging

import ray

from llm_d_rl_verl_integration.base_agent_loop_manager import LlmdBaseAgentLoopManager
from verl.workers.rollout.llm_server import LLMServerClient

from pyis_port.llm_client import PyInferenceLLMClient

logger = logging.getLogger(__name__)


class PyInferenceAgentLoopManager(LlmdBaseAgentLoopManager):
    """Swaps verl's llm_client for the py-inference-scheduler-routed client."""

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        self._address_to_handle = {}
        for i, addr in enumerate(server_addresses):
            actor_name = f"vllm_server_{i}_0"
            try:
                self._address_to_handle[addr] = ray.get_actor(actor_name)
            except ValueError:
                raise RuntimeError(
                    f"Could not find Ray actor {actor_name!r} for server {addr}. "
                    "Make sure the rollout backend is vllm and servers are started."
                )
        logger.info("[PyInferenceAgentLoopManager] address→handle map: %s",
                    list(self._address_to_handle.keys()))

    def _create_llm_client(self) -> LLMServerClient:
        return PyInferenceLLMClient(
            config=self.config,
            load_balancer_handle=self.llm_client._load_balancer,
            address_to_handle=self._address_to_handle,
        )
