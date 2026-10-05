# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online tests for the scale-out (Disaggregated Everything) surface on an omni server.

``vllm-omni serve --tokens-only`` turns the server into a token-in / token-out
worker: upstream ``build_app`` registers ``/v1/chat/completions/render``,
``/inference/v1/generate`` and ``/v1/chat/completions/derender``. The render
and derender handlers read ``serving_render`` / ``serving_derender`` from
``app.state``; ``omni_init_app_state`` has to wire them or those routes answer
"The model does not support ... API". These tests drive the full round trip
on a single-stage thinker-only Qwen2.5-Omni deployment, the smallest omni
pipeline whose final stage emits text tokens.
"""

import os

import pytest
import requests

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen2.5-Omni-7B"
# Same memory caps as the thinker-only sleep test: the 16.78 GiB thinker leaves
# a few GiB of KV cache on a 24 GiB L4 once len/batch are capped at 2048.
STAGE_CONFIG = modify_stage_config(
    get_deploy_config_path("ci/qwen2_5_omni_thinker_only.yaml"),
    updates={"stages": {0: {"max_model_len": 2048, "max_num_batched_tokens": 2048}}},
)
test_params = [
    pytest.param(
        OmniServerParams(model=MODEL, stage_config_path=STAGE_CONFIG, server_args=["--tokens-only"]),
        id="tokens_only",
    )
]
_HTTP_TIMEOUT_S = 300.0


def _post(base_url: str, path: str, payload: dict) -> requests.Response:
    return requests.post(f"{base_url}{path}", json=payload, timeout=_HTTP_TIMEOUT_S)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_render_generate_derender_round_trip(omni_server, online_client) -> None:
    """render -> generate -> derender must produce a chat completion with text.

    Fails if the render or derender state is not wired (404 "not supported"),
    if the token-in / token-out path rejects the rendered request, or if the
    derendered completion disagrees with the generated token count.
    """
    base_url = online_client.base_url
    chat_request = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": "What color is the sky? Answer in one word."}],
        "max_tokens": 8,
        "temperature": 0,
    }

    rendered = _post(base_url, "/v1/chat/completions/render", chat_request)
    assert rendered.status_code == 200, rendered.text
    generate_request = rendered.json()
    prompt_token_ids = generate_request["token_ids"]
    assert prompt_token_ids, generate_request

    generated = _post(base_url, "/inference/v1/generate", generate_request)
    assert generated.status_code == 200, generated.text
    generate_response = generated.json()
    (choice,) = generate_response["choices"]
    assert choice["token_ids"], generate_response
    assert choice["finish_reason"] in {"stop", "length"}

    derendered = _post(
        base_url,
        "/v1/chat/completions/derender",
        {
            "model": omni_server.model,
            "generate_response": generate_response,
            "prompt_tokens": len(prompt_token_ids),
            "chat_request": chat_request,
        },
    )
    assert derendered.status_code == 200, derendered.text
    completion = derendered.json()
    content = completion["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip(), completion
    assert completion["usage"]["prompt_tokens"] == len(prompt_token_ids)
    assert completion["usage"]["completion_tokens"] == len(choice["token_ids"])


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_stop_strings_rejected_on_tokens_only_server(omni_server, online_client) -> None:
    """``--tokens-only`` forces ``detokenize=False`` after request validation.

    Stop strings can never match without detokenization, so the server must
    reject them (vLLM #57058) instead of silently generating past them.
    """
    base_url = online_client.base_url
    payload = {
        "model": omni_server.model,
        "token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 5, "stop": ["never"]},
        "stream": False,
    }

    rejected = _post(base_url, "/inference/v1/generate", payload)
    assert rejected.status_code == 400, rejected.text
    assert "stop strings" in rejected.json()["error"]["message"]

    payload["sampling_params"].pop("stop")
    accepted = _post(base_url, "/inference/v1/generate", payload)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["choices"][0]["token_ids"]
