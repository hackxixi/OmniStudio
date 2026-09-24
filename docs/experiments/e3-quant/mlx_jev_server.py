"""One MLX model, two roles: JEV (OpenJev's /v1/systemone) and decoding (/v1/chat/completions).

OpenJev scores a question with a single forward pass: it renders the prompt, then reads the
log-probabilities of the option-label tokens at the next position. Nothing in that needs
SGLang, so this server reuses OpenJev's own prompt compiler, scoring and HTTP API unchanged
and swaps only the backend call for an MLX forward pass. The same weights then also serve an
OpenAI-style chat endpoint, which is the deployment shape under test: a single local model that
both judges (JEV) and fills in tool calls (decode).

Usage (Apple Silicon):
    PYTHONPATH=<openjev>/src python mlx_jev_server.py --model <mlx model dir> --port 18130 [--prefix-cache]

--prefix-cache reuses the computed state of a shared prompt prefix across requests. Qwen3.5 renders
the tool definitions first, so every chat turn starts with the same ~2.6K tokens. Its linear-attention
layers keep a running state that cannot be trimmed back, so instead of truncating one cache this keeps
snapshots taken exactly at a common-prefix boundary and hands out copies.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import resource
import time
import uuid

import mlx.core as mx
from fastapi import Request
from fastapi.responses import JSONResponse
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from openjev.api import create_app
from openjev.backend import Generation
from openjev.config import Settings
from openjev.prompts import PromptCompiler
from openjev.service import EvaluationService

PREFILL_CHUNK = 512
MIN_SHARED_PREFIX = 256


def clone_cache(cache: list) -> list:
    """Independent copy of a prompt cache (KV caches and linear-attention state alike)."""
    out = []
    for c in cache:
        n = copy.copy(c)
        state = c.state
        if isinstance(state, tuple):
            n.state = tuple(mx.array(x) for x in state)
        else:
            n.state = [None if x is None else mx.array(x) for x in state]
        out.append(n)
    return out


def prefill(model, cache: list, tokens: list[int]) -> None:
    arr = mx.array(tokens)
    for start in range(0, len(tokens), PREFILL_CHUNK):
        model(arr[None, start : start + PREFILL_CHUNK], cache=cache)
        mx.eval([c.state for c in cache])


class PrefixCache:
    """Snapshots at the longest prefix shared with the previous request; copies handed out per request."""

    def __init__(self, model, enabled: bool, keep: int = 4):
        self.model = model
        self.enabled = enabled
        self.keep = keep
        self.snapshots: list[tuple[tuple[int, ...], list]] = []
        self.last: list[int] = []
        self.reused_tokens = 0

    def take(self, tokens: list[int]) -> tuple[list, int]:
        """A cache already holding tokens[:start], and start. Never covers the last token."""
        if not self.enabled:
            return make_prompt_cache(self.model), 0
        best = None
        for key, snap in self.snapshots:
            if len(key) < len(tokens) and tuple(tokens[: len(key)]) == key and (best is None or len(key) > len(best[0])):
                best = (key, snap)
        if best is None and self.last:
            shared = 0
            for a, b in zip(tokens, self.last):
                if a != b:
                    break
                shared += 1
            shared = min(shared, len(tokens) - 1)
            if shared >= MIN_SHARED_PREFIX:
                snap = make_prompt_cache(self.model)
                prefill(self.model, snap, tokens[:shared])
                best = (tuple(tokens[:shared]), snap)
                self.snapshots = [best, *self.snapshots][: self.keep]
        self.last = tokens
        if best is None:
            return make_prompt_cache(self.model), 0
        self.reused_tokens += len(best[0])
        return clone_cache(best[1]), len(best[0])


class MLXBackend:
    """Drop-in for openjev.backend.SGLangClient."""

    def __init__(self, model, lock: asyncio.Lock, prefix: PrefixCache):
        self.model = model
        self.lock = lock
        self.prefix = prefix
        self.jev_ms: list[float] = []

    async def health(self) -> bool:
        return True

    def _label_logprobs(self, input_ids: list[int], label_ids: list[int]) -> list[float]:
        # Prefill all but the last token (reusing a shared prefix when allowed), then read
        # the next-token distribution at the last position.
        cache, start = self.prefix.take(input_ids)
        prefill(self.model, cache, input_ids[start:-1])
        logits = self.model(mx.array(input_ids[-1:])[None], cache=cache)[0, -1].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits)
        picked = logprobs[mx.array(label_ids)]
        mx.eval(picked)
        return picked.tolist()

    async def generate(self, input_ids, label_ids=None, image_data=None):
        if image_data:
            raise ValueError("this MLX backend is text only")
        # OpenJev warms the shared prefix first so SGLang's radix cache can reuse it. With one
        # branch per question there is nothing to reuse here, so the warm-up is a no-op.
        if not label_ids:
            return Generation(logprobs=[], input_tokens=0, output_tokens=0, cached_tokens=None)
        async with self.lock:
            started = time.perf_counter()
            logprobs = await asyncio.to_thread(self._label_logprobs, input_ids, label_ids)
            self.jev_ms.append((time.perf_counter() - started) * 1000)
        return Generation(logprobs=logprobs, input_tokens=len(input_ids), output_tokens=0, cached_tokens=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18130)
    parser.add_argument("--prefix-cache", action="store_true")
    args = parser.parse_args()

    loaded_at = time.perf_counter()
    model, wrapper = load(args.model)
    hf_tokenizer = wrapper._tokenizer
    load_s = time.perf_counter() - loaded_at
    weights_bytes = mx.get_active_memory()

    lock = asyncio.Lock()
    jev_prefix = PrefixCache(model, args.prefix_cache)
    chat_prefix = PrefixCache(model, args.prefix_cache)
    backend = MLXBackend(model, lock, jev_prefix)
    settings = Settings(
        model=args.model,
        served_model_name=args.model.rstrip("/").split("/")[-1],
        model_alias="jev-latest",
        images="off",
        api_key=None,
    )
    service = EvaluationService(settings, PromptCompiler(hf_tokenizer, False), backend)
    app = create_app(settings, service=service)
    decode_stats: list[dict] = []

    def run_chat(body: dict) -> dict:
        kwargs = body.get("chat_template_kwargs") or {}
        prompt = hf_tokenizer.apply_chat_template(
            body["messages"],
            tools=body.get("tools"),
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )
        sampler = make_sampler(temp=float(body.get("temperature", 0) or 0))
        tokens = hf_tokenizer.encode(prompt, add_special_tokens=False)
        # Timed from here so building a snapshot counts against the request that pays for it.
        started = time.perf_counter()
        cache, start = chat_prefix.take(tokens)
        text = ""
        last = None
        first_token_s = None
        for last in stream_generate(
            model,
            wrapper,
            tokens[start:],
            max_tokens=int(body.get("max_tokens", 1024)),
            sampler=sampler,
            prompt_cache=cache,
        ):
            if first_token_s is None:
                first_token_s = time.perf_counter() - started
            text += last.text
        stats = {
            "prompt_tokens": last.prompt_tokens,
            "completion_tokens": last.generation_tokens,
            "prompt_tps": round(last.prompt_tps, 1),
            "generation_tps": round(last.generation_tps, 1),
            "peak_memory_gb": round(last.peak_memory, 2),
            "first_token_s": round(first_token_s or 0, 2),
            "total_s": round(time.perf_counter() - started, 2),
        }
        decode_stats.append(stats)
        return {"text": text, **stats}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        async with lock:
            out = await asyncio.to_thread(run_chat, body)
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "model": settings.served_model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": out["text"]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"]},
                "timing": {k: out[k] for k in ("prompt_tps", "generation_tps", "peak_memory_gb")},
            }
        )

    @app.get("/stats")
    async def stats():
        def median(xs):
            return sorted(xs)[len(xs) // 2] if xs else None

        return {
            "model": args.model,
            "load_seconds": round(load_s, 1),
            "weights_gb": round(weights_bytes / 1e9, 2),
            "active_gb": round(mx.get_active_memory() / 1e9, 2),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "process_max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
            "jev_calls": len(backend.jev_ms),
            "jev_ms_median": median(backend.jev_ms),
            "chat_calls": len(decode_stats),
            "prompt_tps_median": median([s["prompt_tps"] for s in decode_stats]),
            "first_token_s_median": median([s["first_token_s"] for s in decode_stats]),
            "chat_total_s_median": median([s["total_s"] for s in decode_stats]),
            "prefix_cache": args.prefix_cache,
            "reused_prefix_tokens": {"jev": jev_prefix.reused_tokens, "chat": chat_prefix.reused_tokens},
            "generation_tps_median": median([s["generation_tps"] for s in decode_stats]),
            "chat_peak_gb_max": max((s["peak_memory_gb"] for s in decode_stats), default=None),
        }

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
