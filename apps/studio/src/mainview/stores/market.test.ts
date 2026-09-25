import { afterAll, expect, test } from "bun:test";

/**
 * 模型市场的平台偏好：没手动选过就跟随下载源路由的默认平台；手动选过就记住（跨重启），
 * 之后路由结论变了也不覆盖。localStorage 不可用（隐私模式）时当次会话照样生效。
 */

const store = new Map<string, string>();
const savedWindow = (globalThis as { window?: unknown }).window;
(globalThis as { window?: unknown }).window = {
  localStorage: {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
  },
};

afterAll(() => {
  (globalThis as { window?: unknown }).window = savedWindow;
});

/** 每次拿一份全新的 store 模块（模拟重启应用）。 */
async function freshStore() {
  const mod = await import(`./market?fresh=${Math.random()}`);
  return mod as typeof import("./market");
}

test("没选过：先用 ModelScope 占位，路由结论到了就跟随", async () => {
  store.clear();
  const { useMarketStore } = await freshStore();
  expect(useMarketStore.getState().source).toBe("modelscope");
  expect(useMarketStore.getState().sourceChosen).toBe(false);
  useMarketStore.getState().applyDefaultSource("huggingface");
  expect(useMarketStore.getState().source).toBe("huggingface");
  // 路由默认值不落盘：下次启动仍按当时的路由结论走。
  expect(store.size).toBe(0);
});

test("手动选过：记住选择，路由默认值不再覆盖；重启后恢复", async () => {
  store.clear();
  const { useMarketStore, MARKET_SOURCE_KEY } = await freshStore();
  useMarketStore.getState().setSource("huggingface");
  expect(store.get(MARKET_SOURCE_KEY)).toBe("huggingface");
  useMarketStore.getState().applyDefaultSource("modelscope");
  expect(useMarketStore.getState().source).toBe("huggingface");

  const restarted = (await freshStore()).useMarketStore;
  expect(restarted.getState().source).toBe("huggingface");
  expect(restarted.getState().sourceChosen).toBe(true);
  restarted.getState().applyDefaultSource("modelscope");
  expect(restarted.getState().source).toBe("huggingface");
});

test("存了坏值 → 当没选过", async () => {
  store.clear();
  store.set("omni.market.source", "github");
  const { useMarketStore } = await freshStore();
  expect(useMarketStore.getState().sourceChosen).toBe(false);
  expect(useMarketStore.getState().source).toBe("modelscope");
});

test("localStorage 抛错（隐私模式）→ 当次会话仍能切换", async () => {
  const saved = (globalThis as { window?: unknown }).window;
  (globalThis as { window?: unknown }).window = {
    localStorage: {
      getItem: () => {
        throw new Error("SecurityError");
      },
      setItem: () => {
        throw new Error("SecurityError");
      },
    },
  };
  try {
    const { useMarketStore } = await freshStore();
    expect(useMarketStore.getState().source).toBe("modelscope");
    useMarketStore.getState().setSource("huggingface");
    expect(useMarketStore.getState().source).toBe("huggingface");
  } finally {
    (globalThis as { window?: unknown }).window = saved;
  }
});
