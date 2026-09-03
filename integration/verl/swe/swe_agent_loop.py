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
"""verl AgentLoop that drives a SWE agent against a GKE agent-sandbox.

Per trajectory: claim a sandbox for the instance image, loop
generate -> parse bash block -> exec in sandbox -> append observation tokens
(mask 0), until the model submits or a budget trips. On submit, extract the
non-test diff and grade it in a FRESH sandbox (pristine grading, see
pristine_grader.py); the reward rides back on AgentLoopOutput.reward_score.

Token accounting follows verl's ToolAgentLoop: LLM tokens are appended
verbatim (never re-tokenized), observations are tokenized as delta user
messages with remove_system_prompt=True.

Register via agent_loop_config_path (see examples/swe_agent_loop.yaml);
dataset rows select it with agent_name == "swe_agent". Requires
data.return_raw_chat=True and the RBAC in configs/swe_sandbox_rbac.yaml.

Compatible with the verl build on the cluster (0.9.0.dev interface:
server_manager.generate returns TokenOutput); tolerates older builds where
generate returned a bare token id list.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer  # type: ignore[import-not-found]

from integration.verl.swe.pristine_grader import grade_patch_pristine
from integration.verl.swe.sandbox import get_thread_client, make_name
from integration.verl.swe.scaffold import (
    NO_COMMAND_OBSERVATION,
    build_system_prompt,
    format_observation,
    is_submit,
    parse_bash_command,
)

logger = logging.getLogger(__name__)

DIFF_CMD = "cd /testbed && git add -N . >/dev/null 2>&1; git diff HEAD"
MAX_DIFF_BYTES = 512 * 1024

# R2E images ship with a dirty git state (the harness's baked-in edits).
# Committing it as a baseline right after sandbox start makes `git diff HEAD`
# agent-only — the fresh grading sandbox has the identical dirty working tree,
# so an agent-only patch applies cleanly there.
BASELINE_CMD = (
    "cd /testbed && git add -A >/dev/null 2>&1; "
    "git -c user.email=swe@llm-d.dev -c user.name=swe commit -qm baseline >/dev/null 2>&1; true"
)


@register("swe_agent")
class SWEAgentLoop(AgentLoopBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        rollout = self.rollout_config
        self.response_length: int = rollout.response_length
        multi_turn = rollout.get("multi_turn", None)
        self.max_assistant_turns: int = (
            multi_turn.get("max_assistant_turns") if multi_turn else None
        ) or 32
        self._init_tunables()

    def _init_tunables(self) -> None:
        """Initialize env-tunable knobs in one place.

        Grouped so test drivers that bypass __init__ can initialize them in
        one call (see integration_check.py).
        """
        self.namespace = os.getenv("SWE_SANDBOX_NAMESPACE", "agents-system")
        self.cmd_timeout_s = int(os.getenv("SWE_CMD_TIMEOUT_S", "60"))
        self.obs_max_chars = int(os.getenv("SWE_OBS_MAX_CHARS", "6000"))
        # Cover autoscaler scale-up + node-cold pulls of multi-GiB task images.
        self.sandbox_wait_s = int(os.getenv("SWE_SANDBOX_WAIT_S", "600"))
        # Rollout sandboxes idle between commands; small requests pack more per
        # node (CPU quota is the concurrency ceiling). Bursts use the limit,
        # not the request. Grading sandboxes keep beefier defaults - graded
        # test runs are timing-sensitive.
        self.sandbox_cpu_request = os.getenv("SWE_SANDBOX_CPU_REQUEST", "500m")
        self.sandbox_mem_request = os.getenv("SWE_SANDBOX_MEM_REQUEST", "1Gi")

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:  # noqa: C901,PLR0912,PLR0914,PLR0915,ANN401
        extra_info = dict(kwargs["extra_info"])
        messages = [dict(m) for m in kwargs["raw_prompt"]]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = build_system_prompt(messages[0]["content"])
        prompt_ids: list[int] = await self.apply_chat_template(messages)

        full_ids = list(prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        has_logprobs = True
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        sandbox_name = make_name("swe", extra_info["instance_id"])
        assistant_turns = user_turns = 0
        submitted = False
        reward, reason = 0.0, "no-submit"

        client = None
        # Boot the sandbox concurrently with the first generate: turn 1 never
        # needs the sandbox until its command executes, and sandbox startup
        # (scheduling + image pull) is tens of seconds.
        sandbox_future = self.loop.run_in_executor(
            None, lambda: self._create_sandbox(sandbox_name, extra_info["docker_image"])
        )
        try:
            while (
                assistant_turns < self.max_assistant_turns
                and len(response_mask) < self.response_length
            ):
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=full_ids,
                        sampling_params=sampling_params,
                    )
                token_ids = getattr(output, "token_ids", output)
                log_probs = getattr(output, "log_probs", None)
                budget = self.response_length - len(response_mask)
                token_ids = list(token_ids)[:budget]
                full_ids += token_ids
                response_mask += [1] * len(token_ids)
                if log_probs is None:
                    has_logprobs = False
                elif has_logprobs:
                    response_logprobs += list(log_probs)[:budget]
                assistant_turns += 1
                if len(response_mask) >= self.response_length:
                    reason = "token-budget"
                    break

                text = await self.loop.run_in_executor(
                    None, lambda t=tuple(token_ids): self.tokenizer.decode(t)
                )
                command = parse_bash_command(text)
                if command is not None and is_submit(command):
                    submitted = True
                    break

                if command is None:
                    observation = NO_COMMAND_OBSERVATION
                else:
                    with simple_timer("tool_calls", metrics):
                        if client is None:
                            client = await sandbox_future
                        rc, out = await self.loop.run_in_executor(
                            None, lambda c=command: self._run_command(client, sandbox_name, c)  # noqa: B023
                        )
                    observation = format_observation(rc, out, self.obs_max_chars)

                obs_ids = await self.apply_chat_template(
                    [{"role": "user", "content": observation}], remove_system_prompt=True
                )
                if len(response_mask) + len(obs_ids) >= self.response_length:
                    reason = "token-budget"
                    break
                full_ids += obs_ids
                response_mask += [0] * len(obs_ids)
                if has_logprobs:
                    response_logprobs += [0.0] * len(obs_ids)
                user_turns += 1

            if assistant_turns >= self.max_assistant_turns and not submitted:
                reason = "max-turns"

            if submitted:
                with simple_timer("tool_calls", metrics):
                    if client is None:
                        client = await sandbox_future
                    reward, reason = await self.loop.run_in_executor(
                        None, lambda: self._grade(client, sandbox_name, extra_info)
                    )
        except Exception as e:  # sandbox/infra failure: salvage the trajectory  # noqa: BLE001
            logger.warning("swe_agent rollout error on %s: %s", extra_info.get("instance_id"), e)
            reward, reason = 0.0, f"rollout-error: {type(e).__name__}"
        finally:
            if client is None:
                # Consume the boot future's result/exception so it never goes unobserved.
                try:  # noqa: SIM105
                    client = await sandbox_future
                except Exception:  # noqa: BLE001,S110
                    pass
            # Delete the Sandbox CR unconditionally: create may have succeeded
            # even when wait_ready timed out (pod stuck Pending).
            try:  # noqa: SIM105
                await self.loop.run_in_executor(
                    None, lambda: get_thread_client(self.namespace).delete(sandbox_name)
                )
            except Exception:  # noqa: BLE001,S110
                pass

        response_ids = full_ids[len(prompt_ids):]
        if not response_ids:
            # verl's padding path can't handle empty responses ('list' has no
            # attribute 'dim'); emit one loss-masked pad token instead.
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            response_ids = [pad_id]
            response_mask = [0]
            response_logprobs = [0.0]
        logger.info(
            "swe_agent %s: reward=%s reason=%s turns=%d resp_tokens=%d",
            extra_info.get("instance_id"), reward, reason, assistant_turns, len(response_ids),
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if has_logprobs else None,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            reward_score=reward,
            extra_fields={"swe_reason": reason, "swe_submitted": submitted},
        )

    # --- sync helpers, always called via run_in_executor -------------------

    def _create_sandbox(self, name: str, image: str):
        client = get_thread_client(self.namespace)
        client.create(
            name, image,
            cpu_request=self.sandbox_cpu_request,
            memory_request=self.sandbox_mem_request,
        )
        client.wait_ready(name, timeout=self.sandbox_wait_s)
        client.exec(name, BASELINE_CMD, timeout=120)
        return client

    def _run_command(self, client, name: str, command: str) -> tuple[int, str]:
        import shlex

        wrapped = f"cd /testbed && timeout {self.cmd_timeout_s} sh -c {shlex.quote(command)}"
        return client.exec(name, wrapped, timeout=self.cmd_timeout_s + 30)

    def _grade(self, client, name: str, extra_info: dict) -> tuple[float, str]:
        rc, diff = client.exec(name, DIFF_CMD, timeout=120)
        if rc != 0:
            return 0.0, f"diff-failed: {diff[-200:]}"
        if len(diff) > MAX_DIFF_BYTES:
            return 0.0, "diff-too-large"
        return grade_patch_pristine(extra_info, diff, namespace=self.namespace)
