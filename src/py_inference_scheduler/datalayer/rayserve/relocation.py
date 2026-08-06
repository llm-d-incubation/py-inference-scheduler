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
import uuid

from ray.llm._internal.serve.core.configs.openai_api_models import (  # noqa: PLC2701
    CompletionRequest,
    ErrorResponse,
)
from ray.llm._internal.serve.core.ingress.ingress import (  # noqa: PLC2701
    DEFAULT_LLM_ROUTER_HTTP_TIMEOUT,
    OpenAIHTTPException,
    OpenAiIngress,
    router_request_timeout,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

ENABLE_ENV = "ENABLE_MOONCAKE_RELOCATION"


# Header and id names that travel inside relocation requests and responses.
# Defined once here so the ingress, router, engine-side triggers, and clients
# all use them identically.
PULL_MARKER = "rls-pull-"  # id prefix labeling a request as a pull
PULL_OF_HEADER = "x-rls-pull-of"  # on a pull: id of the request it continues
TARGET_HEADER = "x-rls-target-replica"  # route this request to exactly this replica
PUSH_PULL_HEADER = "x-rls-push-pull"  # on a response: the merged push and pull ids


def relocation_enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1"


def merge_completion_responses(push_response: dict, pull_response: dict) -> dict:
    """One client response from the two halves.

    The pushed request's partial output followed by the pull's continuation.
    No recompute, just formatting the final object to look like what was
    requested.
    """
    push_choice = push_response["choices"][0]
    pull_choice = pull_response["choices"][0]

    choice = dict(pull_choice)
    choice["index"] = 0
    choice["text"] = (push_choice.get("text") or "") + (pull_choice.get("text") or "")
    if push_choice.get("token_ids") is not None:
        choice["token_ids"] = list(push_choice["token_ids"]) + list(
            pull_choice.get("token_ids") or []
        )

    merged = dict(pull_response)
    merged["id"] = push_response["id"]
    merged["created"] = push_response["created"]
    merged["choices"] = [choice]
    # The end client sent prompt p; the model produced d1 (before the abort)
    # + d2 (after the pull). The pull's own usage claims prompt=p+d1 only
    # because the ingress resubmitted d1 inside the pull's prompt, so prompt
    # accounting must come from the push half.
    merged["prompt_token_ids"] = push_response.get("prompt_token_ids")
    push_usage = push_response.get("usage") or {}
    pull_usage = pull_response.get("usage") or {}
    if push_usage and pull_usage:
        prompt_tokens = push_usage["prompt_tokens"]
        completion_tokens = (
            push_usage["completion_tokens"] + pull_usage["completion_tokens"]
        )
        merged["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return merged


def _strip_token_ids(response: dict) -> None:
    response.pop("prompt_token_ids", None)
    for choice in response.get("choices", []):
        choice.pop("token_ids", None)
        choice.pop("prompt_token_ids", None)


def _with_headers(request: Request, extra: dict[str, str]) -> Request:
    # Same synthetic-request shape as RawRequestInfo.to_starlette_request.
    headers = {**dict(request.headers), **extra}
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


class RelocatingIngress(OpenAiIngress):
    """
    Resubmits "aborted" requests as continuations on another replica.

    Merges the two responses into one.
    """

    async def completions(self, body: CompletionRequest, request: Request) -> Response:
        multi_prompt = (
            isinstance(body.prompt, list)
            and bool(body.prompt)
            and isinstance(body.prompt[0], (str, list))
        )
        if (
            not relocation_enabled()
            or bool(body.stream)
            or (body.n or 1) != 1
            or multi_prompt
        ):
            return await super().completions(body, request)  # type: ignore[no-any-return]

        client_wants_ids = bool(body.return_token_ids)
        push_id = uuid.uuid4().hex
        push_body = body.model_copy(update={"return_token_ids": True})

        async with router_request_timeout(DEFAULT_LLM_ROUTER_HTTP_TIMEOUT):
            push_response = await self._single_response(
                push_body, request, {"x-request-id": push_id}
            )
            push_choice = push_response["choices"][0]
            if push_choice.get("finish_reason") != "abort":
                if not client_wants_ids:
                    _strip_token_ids(push_response)
                return JSONResponse(content=push_response)

            prompt_ids = (
                push_response.get("prompt_token_ids")
                or push_choice.get("prompt_token_ids")
                or []
            )
            generated_ids = push_choice.get("token_ids") or []
            update: dict = {
                "prompt": list(prompt_ids) + list(generated_ids),
                "return_token_ids": True,
            }
            if body.max_tokens is not None:
                update["max_tokens"] = max(1, body.max_tokens - len(generated_ids))
            pull_body = body.model_copy(update=update)
            pull_id = f"{PULL_MARKER}{uuid.uuid4().hex}"
            pull_response = await self._single_response(
                pull_body,
                request,
                {"x-request-id": pull_id, PULL_OF_HEADER: push_id},
            )

            merged = merge_completion_responses(push_response, pull_response)
            if not client_wants_ids:
                _strip_token_ids(merged)
            return JSONResponse(
                content=merged,
                headers={PUSH_PULL_HEADER: f"{push_id},{pull_id}"},
            )

    async def _single_response(
        self, body: CompletionRequest, request: Request, extra_headers: dict[str, str]
    ) -> dict:
        """Non-streaming completions yield exactly one response object."""
        synthetic = _with_headers(request, extra_headers)
        async for response in self._get_response(
            body=body, call_method="completions", raw_request=synthetic
        ):
            if isinstance(response, ErrorResponse):
                raise OpenAIHTTPException(
                    message=response.error.message,
                    status_code=response.error.code,
                    type=response.error.type,
                )
            return response.model_dump()  # type: ignore[no-any-return]
        raise OpenAIHTTPException(
            message="empty response stream", status_code=500, type="InternalError"
        )
