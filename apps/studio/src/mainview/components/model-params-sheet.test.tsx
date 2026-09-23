import { afterAll, afterEach, beforeEach, expect, mock, test } from "bun:test";
import { Window } from "happy-dom";

/**
 * 「模型参数」抽屉：有效值 + 来源徽标、保存时只发填了的字段、需重启提示、
 * 以及「加载中 → 加载完」两次渲染 hook 数量一致（这个仓库吃过 React #300/#310）。
 */
const dom = new Window({ url: "http://localhost/" });
const DOM_GLOBALS = [
  "window",
  "document",
  "navigator",
  "location",
  "history",
  "localStorage",
  "HTMLElement",
  "HTMLDivElement",
  "HTMLButtonElement",
  "HTMLInputElement",
  "Element",
  "Node",
  "Text",
  "DocumentFragment",
  "SVGElement",
  "DOMRect",
  "CustomElementRegistry",
  "Event",
  "CustomEvent",
  "MouseEvent",
  "PointerEvent",
  "KeyboardEvent",
  "FocusEvent",
  "InputEvent",
  "MutationObserver",
  "ResizeObserver",
  "NodeFilter",
  "requestAnimationFrame",
  "cancelAnimationFrame",
  "getComputedStyle",
  "matchMedia",
] as const;
const savedGlobals = new Map<string, unknown>();
for (const key of DOM_GLOBALS) {
  const value = (dom as unknown as Record<string, unknown>)[key];
  if (value === undefined) continue;
  savedGlobals.set(key, (globalThis as unknown as Record<string, unknown>)[key]);
  (globalThis as unknown as Record<string, unknown>)[key] = value;
}
(globalThis as unknown as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;

const { act, createElement } = await import("react");
const { createRoot } = await import("react-dom/client");
const { QueryClient, QueryClientProvider } = await import("@tanstack/react-query");
const { TooltipProvider } = await import("@ui/tooltip");

const MODEL = "/models/Qwen3-8B-Q4_K_M.gguf";

type GetResult = {
  model: string;
  params: Record<string, unknown> | null;
  sampling: {
    values: Record<string, number>;
    sources: Record<string, string>;
    preset: string | null;
  };
  launchPreview?: string;
  needsRestart: boolean;
};

function baseResult(): GetResult {
  return {
    model: MODEL,
    params: { ctxSize: 8192, sampling: { temperature: 0.5 } },
    sampling: {
      values: { temperature: 0.5, topP: 0.95, topK: 20, minP: 0, presencePenalty: 0, repeatPenalty: 1 },
      sources: {
        temperature: "model-override",
        topP: "family-preset",
        topK: "family-preset",
        minP: "model-metadata",
        presencePenalty: "global",
        repeatPenalty: "family-preset",
      },
      preset: "qwen3",
    },
    launchPreview: "llama-server -m /models/Qwen3-8B-Q4_K_M.gguf -c 8192",
    needsRestart: false,
  };
}

let nextResult: GetResult = baseResult();
/** 非空时 getModelParams 挂住，直到测试手动放行（造「加载中」）。 */
let gate: Promise<void> | null = null;
const saved: { model: string; params: unknown }[] = [];
let saveResponse: { ok: boolean; params: unknown; needsRestart: boolean; error?: string } = {
  ok: true,
  params: null,
  needsRestart: false,
};

mock.module("@lib/rpc", () => ({
  rpcClient: {
    getModelParams: async () => {
      if (gate) await gate;
      return nextResult;
    },
    setModelParams: async (args: { model: string; params: unknown }) => {
      saved.push(args);
      return saveResponse;
    },
    clearModelParams: async () => ({ ok: true, needsRestart: false }),
    listModelParams: async () => ({ entries: [] }),
    restartServedModel: async () => ({ ok: true }),
  },
}));

const { ModelParamsSheet, buildModelParams, draftFromParams } = await import("./model-params-sheet");

let container: HTMLDivElement | null = null;
let root: ReturnType<typeof createRoot> | null = null;

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

async function renderSheet() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root!.render(
      createElement(
        QueryClientProvider,
        { client: queryClient },
        createElement(
          TooltipProvider,
          null,
          createElement(ModelParamsSheet, { model: MODEL, label: "Qwen3-8B", open: true, onOpenChange: () => {} }),
        ),
      ),
    );
  });
  await settle();
}

const input = (name: string) => document.body.querySelector(`input[name="${name}"]`) as HTMLInputElement;
const buttonByText = (text: string) =>
  Array.from(document.body.querySelectorAll("button")).find((b) => (b.textContent ?? "").trim() === text) as
    | HTMLButtonElement
    | undefined;

/** 受控 input：走原生 setter + input 事件，React 才认。 */
async function typeInto(el: HTMLInputElement, value: string) {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(
      (window as unknown as { HTMLInputElement: { prototype: object } }).HTMLInputElement.prototype,
      "value",
    )!.set!;
    setter.call(el, value);
    el.dispatchEvent(new (window as unknown as { Event: typeof Event }).Event("input", { bubbles: true }));
  });
}

beforeEach(() => {
  nextResult = baseResult();
  gate = null;
  saved.length = 0;
  saveResponse = { ok: true, params: null, needsRestart: false };
});

afterEach(async () => {
  if (root) await act(async () => root!.unmount());
  root = null;
  container?.remove();
  container = null;
  document.body.innerHTML = "";
});

afterAll(() => {
  for (const [key, value] of savedGlobals) {
    (globalThis as unknown as Record<string, unknown>)[key] = value;
  }
});

test("采样项显示有效值（占位）和来源徽标，家族推荐带预设名", async () => {
  await renderSheet();
  expect(document.body.textContent).toContain("模型参数");
  // 覆盖过的温度：输入框里是覆盖值
  expect(input("temperature").value).toBe("0.5");
  // 没覆盖的：输入框空，占位是有效值
  expect(input("topP").value).toBe("");
  expect(input("topP").placeholder).toBe("0.95");
  expect(input("topK").placeholder).toBe("20");

  const badge = (field: string) =>
    document.body.querySelector(`[data-field="${field}"] [data-source]`)?.textContent ?? "";
  expect(badge("temperature")).toBe("按模型设置");
  expect(badge("topP")).toBe("家族推荐（Qwen3）");
  expect(badge("minP")).toBe("模型自带");
  expect(badge("presencePenalty")).toBe("全局默认");
});

test("保存只发填了的字段；清空的采样项不发", async () => {
  await renderSheet();
  await typeInto(input("temperature"), "");
  await typeInto(input("topK"), "40");
  const save = buttonByText("保存")!;
  expect(save.disabled).toBe(false);
  await act(async () => save.click());
  await settle();
  expect(saved).toHaveLength(1);
  expect(saved[0]).toEqual({ model: MODEL, params: { ctxSize: 8192, sampling: { topK: 40 } } });
});

test("越界值就地报错，不能保存", async () => {
  await renderSheet();
  await typeInto(input("topP"), "1.5");
  expect(document.body.textContent).toContain("范围 0 ~ 1");
  expect(buttonByText("保存")!.disabled).toBe(true);
});

test("保存后需重启：显示「重启模型后生效」；后端报错原样显示", async () => {
  await renderSheet();
  expect(document.body.querySelector('[data-testid="model-params-restart"]')).toBeNull();
  saveResponse = { ok: true, params: { ctxSize: 8192 }, needsRestart: true };
  await act(async () => buttonByText("保存")!.click());
  await settle();
  expect(document.body.textContent).toContain("重启模型后生效");

  saveResponse = { ok: false, params: null, needsRestart: false, error: "db locked" };
  await act(async () => buttonByText("保存")!.click());
  await settle();
  expect(document.body.textContent).toContain("db locked");
});

test("getModelParams 带 needsRestart 时一打开就提示", async () => {
  nextResult = { ...baseResult(), needsRestart: true };
  await renderSheet();
  expect(document.body.textContent).toContain("重启模型后生效");
});

test("加载中 → 加载完成：不因 hook 数量变化崩掉", async () => {
  let release!: () => void;
  gate = new Promise<void>((r) => (release = r));
  const errors: unknown[] = [];
  const origError = console.error;
  console.error = (...args: unknown[]) => errors.push(args);
  try {
    await renderSheet();
    expect(input("temperature")).toBeNull(); // 还在加载
    await act(async () => release());
    await settle();
    expect(input("temperature").value).toBe("0.5");
  } finally {
    console.error = origError;
  }
  expect(errors.filter((e) => String(e).includes("Rendered"))).toEqual([]);
});

test("buildModelParams：空 / 跟随 不出现，思考跟随模板不写", () => {
  const draft = draftFromParams(null);
  expect(buildModelParams(draft)).toEqual({ params: {}, errors: {} });
  const d2 = { ...draft, gpuLayers: "-1", thinking: "off" as const, extraArgs: "  --jinja ", cacheTypeK: "q8_0" };
  expect(buildModelParams(d2).params).toEqual({ gpuLayers: -1, thinking: "off", extraArgs: "--jinja", cacheTypeK: "q8_0" });
  expect(buildModelParams({ ...draft, parallel: "1.5" }).errors.parallel).toEqual({ kind: "int" });
  expect(buildModelParams({ ...draft, ctxSize: "100" }).errors.ctxSize).toEqual({ kind: "range", min: 256, max: 1048576 });
  // 往返：参数 → 草稿 → 参数
  const p = { ctxSize: 4096, flashAttn: "on" as const, sampling: { minP: 0.05, presencePenalty: 1.5 } };
  expect(buildModelParams(draftFromParams(p)).params).toEqual(p);
});
