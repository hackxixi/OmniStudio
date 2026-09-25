"""The llama.cpp twin of mlx_jev_server.py: one GGUF model, two roles (JEV + chat decode).

llama-server (started separately, one slot) does the forward passes. JEV needs the log-probabilities
of specific label tokens at the next position; llama.cpp only returns the top-N tokens (`n_probs`),
so this asks for a wide top-N and treats a label outside it as far less likely than anything listed.
Chat prompts are rendered here with the model's *own* chat template (fine-tunes change it) and sent
as token ids, so JEV and chat see exactly the same tokenization; llama.cpp's `cache_prompt` reuses
the shared prefix (the tool definitions) across requests.

Usage:
    llama-server -m model.gguf --port 18131 -c 8192 -np 1 -ngl 99 &
    PYTHONPATH=<openjev>/src python llama_jev_server.py --tokenizer <dir with tokenizer.json> \
        --llama-url http://127.0.0.1:18131 --port 18130
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from openjev.api import create_app
from openjev.backend import Generation
from openjev.config import Settings
from openjev.prompts import PromptCompiler
from openjev.service import EvaluationService
from transformers import AutoTokenizer

TOP_N = 400
# A label that is not even in the top TOP_N gets this much less than the smallest listed logprob.
OUTSIDE_PENALTY = 5.0


class LlamaBackend:
    def __init__(self, client: httpx.AsyncClient, lock: asyncio.Lock):
        self.client = client
        self.lock = lock
        self.jev_ms: list[float] = []
        self.missing_labels = 0

    async def health(self) -> bool:
        try:
            return (await self.client.get("/health", timeout=5)).is_success
        except httpx.HTTPError:
            return False

    async def generate(self, input_ids, label_ids=None, image_data=None):
        if image_data:
            raise ValueError("text only")
        if not label_ids:
            return Generation(logprobs=[], input_tokens=0, output_tokens=0, cached_tokens=None)
        async with self.lock:
            started = time.perf_counter()
            r = await self.client.post(
                "/completion",
                json={
                    "prompt": input_ids,
                    "n_predict": 1,
                    "n_probs": TOP_N,
                    "temperature": 0,
                    "cache_prompt": True,
                    "post_sampling_probs": False,
                },
                timeout=600,
            )
            self.jev_ms.append((time.perf_counter() - started) * 1000)
        r.raise_for_status()
        top = r.json()["completion_probabilities"][0]["top_logprobs"]
        by_id = {t["id"]: t["logprob"] for t in top}
        floor = min(by_id.values()) - OUTSIDE_PENALTY
        logprobs = []
        for label in label_ids:
            if label not in by_id:
                self.missing_labels += 1
            logprobs.append(by_id.get(label, floor))
        return Generation(logprobs=logprobs, input_tokens=len(input_ids), output_tokens=0, cached_tokens=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="dir with tokenizer.json (same vocabulary as the GGUF)")
    parser.add_argument("--chat-template", help="jinja file; defaults to the template llama-server reports")
    parser.add_argument("--llama-url", default="http://127.0.0.1:18131")
    parser.add_argument("--port", type=int, default=18130)
    parser.add_argument("--name", default="gguf")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.chat_template:
        tokenizer.chat_template = open(args.chat_template).read()
    else:
        props = httpx.get(f"{args.llama_url}/props", timeout=30).json()
        tokenizer.chat_template = props["chat_template"]

    client = httpx.AsyncClient(base_url=args.llama_url)
    lock = asyncio.Lock()
    backend = LlamaBackend(client, lock)
    settings = Settings(model=args.tokenizer, served_model_name=args.name, model_alias="jev-latest", images="off", api_key=None)
    service = EvaluationService(settings, PromptCompiler(tokenizer, False), backend)
    app = create_app(settings, service=service)
    stats: list[dict] = []

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        prompt = tokenizer.apply_chat_template(
            body["messages"],
            tools=body.get("tools"),
            add_generation_prompt=True,
            tokenize=False,
            **(body.get("chat_template_kwargs") or {}),
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        async with lock:
            started = time.perf_counter()
            r = await client.post(
                "/completion",
                json={
                    "prompt": tokens,
                    "n_predict": int(body.get("max_tokens", 1024)),
                    "temperature": float(body.get("temperature", 0) or 0),
                    "cache_prompt": True,
                    "stop": ["<|im_end|>", "<|endoftext|>"],
                },
                timeout=600,
            )
            total_s = time.perf_counter() - started
        r.raise_for_status()
        j = r.json()
        t = j.get("timings", {})
        stats.append(
            {
                "prompt_n": t.get("prompt_n"),
                "cache_n": t.get("cache_n"),
                "prompt_tps": t.get("prompt_per_second"),
                "generation_tps": t.get("predicted_per_second"),
                "total_s": round(total_s, 2),
            }
        )
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "model": args.name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": j["content"]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(tokens), "completion_tokens": t.get("predicted_n", 0)},
            }
        )

    @app.get("/stats")
    async def get_stats():
        def median(xs):
            xs = [x for x in xs if x is not None]
            return sorted(xs)[len(xs) // 2] if xs else None

        return {
            "model": args.name,
            "jev_calls": len(backend.jev_ms),
            "jev_ms_median": median(backend.jev_ms),
            "jev_missing_labels": backend.missing_labels,
            "chat_calls": len(stats),
            "prompt_tps_median": median([s["prompt_tps"] for s in stats]),
            "generation_tps_median": median([s["generation_tps"] for s in stats]),
            "cached_prompt_tokens_total": sum(s["cache_n"] or 0 for s in stats),
            "chat_total_s_median": median([s["total_s"] for s in stats]),
        }

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
