import { afterEach, beforeEach, expect, mock, test } from "bun:test";

import {
  getServedContext,
  invalidateServedContext,
  localMaxTokens,
  peekServedContext,
  probeServedContext,
  resetServedContextCache,
  SERVED_CONTEXT_TTL_MS,
} from "./served-context";

/**
 * 假服务：按路径分流。`props` / `models` 省略 = 该端点 404（mlx-lm 没有 /props、
 * vLLM 也没有）；给 "throw" = 连接失败。
 */
function fakeServer(opts: { props?: unknown | "throw"; models?: unknown | "throw" }) {
  const calls: string[] = [];
  const fn = mock(async (url: unknown) => {
    const u = String(url);
    calls.push(u);
    const body = u.endsWith("/props") ? opts.props : u.endsWith("/v1/models") ? opts.models : undefined;
    if (body === "throw") throw new Error("ECONNREFUSED");
    if (body === undefined) return new Response("not found", { status: 404 });
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  globalThis.fetch = fn as never;
  return calls;
}

const realFetch = globalThis.fetch;
const realNow = Date.now;
beforeEach(() => resetServedContextCache());
afterEach(() => {
  globalThis.fetch = realFetch;
  Date.now = realNow;
});

test("llama.cpp：读 /props 的每 slot n_ctx，不再去问 /v1/models", async () => {
  const calls = fakeServer({ props: { default_generation_settings: { n_ctx: 16384 }, total_slots: 4 } });
  expect(await probeServedContext("http://127.0.0.1:8080/", "m.gguf")).toEqual({
    window: 16384,
    source: "llama-props",
  });
  expect(calls).toEqual(["http://127.0.0.1:8080/props"]);
});

test("vLLM：/props 404 → 用 /v1/models 里对应 id 的 max_model_len", async () => {
  fakeServer({
    models: {
      data: [
        { id: "other", max_model_len: 4096 },
        { id: "qwen", max_model_len: 32768 },
      ],
    },
  });
  expect(await probeServedContext("http://h:8000", "qwen")).toEqual({ window: 32768, source: "max-model-len" });
});

test("vLLM：只有一个条目时 id 对不上也认它（服务名与请求 id 不一致的情况）", async () => {
  fakeServer({ models: { data: [{ id: "/abs/path/Qwen", max_model_len: 8192 }] } });
  expect(await probeServedContext("http://h:8000", "Qwen")).toEqual({ window: 8192, source: "max-model-len" });
});

test("mlx-lm 等不暴露窗口 / 连不上 → null", async () => {
  fakeServer({ models: { data: [{ id: "mlx", object: "model" }] } });
  expect(await probeServedContext("http://h:8080", "mlx")).toBeNull();
  fakeServer({ props: "throw", models: "throw" });
  expect(await probeServedContext("http://h:8080", "mlx")).toBeNull();
  // n_ctx 不是正数也当没拿到
  fakeServer({ props: { default_generation_settings: { n_ctx: 0 } } });
  expect(await probeServedContext("http://h:8080", "x")).toBeNull();
});

test("缓存：TTL 内不再发请求（包括「探不到」），过期后重新探测", async () => {
  const calls = fakeServer({ props: { default_generation_settings: { n_ctx: 8192 } } });
  let now = 1_000_000;
  Date.now = () => now;

  expect(peekServedContext("http://h:1", "m")).toBeUndefined();
  await getServedContext("http://h:1", "m");
  await getServedContext("http://h:1", "m");
  expect(calls).toHaveLength(1);
  expect(peekServedContext("http://h:1", "m")).toEqual({ window: 8192, source: "llama-props" });

  now += SERVED_CONTEXT_TTL_MS + 1;
  expect(peekServedContext("http://h:1", "m")).toBeUndefined();
  await getServedContext("http://h:1", "m");
  expect(calls).toHaveLength(2);

  // 不同模型 / 不同端口各自缓存
  await getServedContext("http://h:2", "m");
  expect(calls).toHaveLength(3);
});

test("并发探测合并成一次", async () => {
  const calls = fakeServer({ props: { default_generation_settings: { n_ctx: 4096 } } });
  const [a, b] = await Promise.all([getServedContext("http://h:1", "m"), getServedContext("http://h:1", "m")]);
  expect(a).toEqual(b);
  expect(calls).toHaveLength(1);
});

test("invalidate：作废后下一次重新探测（实例重启换了 ctx）", async () => {
  fakeServer({ props: { default_generation_settings: { n_ctx: 4096 } } });
  expect((await getServedContext("http://h:1/", "m"))?.window).toBe(4096);
  fakeServer({ props: { default_generation_settings: { n_ctx: 65536 } } });
  expect((await getServedContext("http://h:1", "m"))?.window).toBe(4096); // 仍在缓存（尾部斜杠不影响 key）
  invalidateServedContext("http://h:1");
  expect((await getServedContext("http://h:1", "m"))?.window).toBe(65536);
});

test("localMaxTokens：llama.cpp 直接给每 slot 窗口，且不会超过它（哪怕小于 1024）", () => {
  expect(localMaxTokens({ served: { window: 16384, source: "llama-props" }, fallbackWindow: 131072 })).toBe(16384);
  expect(localMaxTokens({ served: { window: 512, source: "llama-props" }, fallbackWindow: 8192 })).toBe(512);
  // prompt 不扣：llama.cpp 超窗只会以 length 收尾，不报错
  expect(
    localMaxTokens({
      served: { window: 8192, source: "llama-props" },
      fallbackWindow: 8192,
      promptMessages: [{ role: "user", content: "x".repeat(8000) }],
    }),
  ).toBe(8192);
  // 256K 封顶
  expect(localMaxTokens({ served: { window: 1_048_576, source: "llama-props" }, fallbackWindow: 8192 })).toBe(262_144);
});

test("localMaxTokens：max_model_len 要给 prompt 让位，prompt + max_tokens 不超窗", () => {
  const served = { window: 8192, source: "max-model-len" as const };
  const promptMessages = [{ role: "user", content: "x".repeat(4000) }]; // ≈1000 token
  const out = localMaxTokens({ served, fallbackWindow: 8192, promptMessages });
  expect(out).toBeLessThan(8192 - 1000);
  expect(out).toBeGreaterThan(4096);
  // 没给 prompt：按一半窗口保守
  expect(localMaxTokens({ served, fallbackWindow: 8192 })).toBe(4096);
  // prompt 把窗口吃满：至少留 256，不出现 0 / 负数
  expect(localMaxTokens({ served, fallbackWindow: 8192, promptMessages: [{ role: "user", content: "字".repeat(9000) }] })).toBe(256);
});

test("localMaxTokens：探不到时走兜底窗口，夹在 [1024, 256K]", () => {
  expect(localMaxTokens({ served: null, fallbackWindow: 8192 })).toBe(8192);
  expect(localMaxTokens({ served: undefined, fallbackWindow: 100 })).toBe(1024);
  expect(localMaxTokens({ served: null, fallbackWindow: 10_000_000 })).toBe(262_144);
});
