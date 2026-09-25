import { afterEach, beforeAll, afterAll, beforeEach, describe, expect, mock, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { tmpdir } from "os";

import { buildLaunchPlanKeyFromSettings, __setLaunchPlanForTest } from "../launch-plan";
import { mockModulePartial } from "../test-mocks";
import { llamaCppBinaryPath } from "../engine-paths";
import { effectiveFlashAttnForPlan } from "./llama";
import { DEFAULT_CUSTOM_SERVER_ARGS } from "./llama";
import type { LaunchPlan } from "../../shared/launch-planner";

/**
 * llama.cpp 运行时的命令行构造：purpose=embedding 两态快照（③-A1a）。
 *
 * 桩掉设置表 / 模型扫描 / 模型库，只测 buildArgs / buildCommandLine 自己：
 *   1. chat 实例命令行与旧版逐字节一致（聊天路径零变化）；
 *   2. embedding 实例追加 `--embeddings --pooling`、裁剪聊天采样参数、端口回落嵌入段。
 */

// 桩一律「读写同源」并尽量展开真实模块：bun 的 mock.module 是进程级共享、
// 且 ESM 绑定在首次导入时固化——getSetting 覆盖而 updateSettings 留真实（写临时
// DB）会让之后评估的 central-repo.test.ts 出现「写入丢失」假失败（读写分家）。
// 形态对齐 model-servers.test.ts 那份在全量跑里验证过的桩：同一本地 store 读写。
// 展开真实模块再覆盖（mock-hygiene.test.ts 规矩）：db/settings 新增导出后字面量替身会炸。
const SETTINGS: Record<string, string> = {};
let PORT_OVERRIDE: string | null = null;
const realSettings = await import("../db/settings");
mock.module("../db/settings", () => ({
  ...realSettings,
  // 与真 getSetting 同一读路径：未设过（undefined）回落 DEFAULTS，显式空串保持空
  // （真 DB 里 "" 就是空，不会回落默认）。GPU 层的 -1 哨兵依赖这个默认值。
  getSetting: (key: string) => {
    if (key in SETTINGS) return SETTINGS[key];
    return (realSettings as { DEFAULTS?: Record<string, string> }).DEFAULTS?.[key] ?? "";
  },
  getNumericSetting: (key: string) => Number(SETTINGS[key] ?? 0) || 0,
  updateSettings: (values: Record<string, string>) => Object.assign(SETTINGS, values),
  getAllSettings: () => ({ ...SETTINGS }),
  getServerPort: (engine: string) =>
    engine === "llama.cpp" ? (SETTINGS.SERVER_PORT || "18080") : (SETTINGS.VLLM_PORT || "8081"),
  setActiveServerPortOverride: (port: string | null) => {
    PORT_OVERRIDE = port;
  },
  getActiveServerPort: () => PORT_OVERRIDE ?? SETTINGS.SERVER_PORT ?? "18080",
}));

const realModelScan = await import("../model-scan");
mock.module("../model-scan", () => ({
  ...realModelScan,
  modelNameForPath: (p: string) => p.split("/").pop() ?? p,
}));

const realModelStore = await import("../model-store");
mock.module("../model-store", () => ({
  ...realModelStore,
  slugModelFileName: (name: string) => name.replace(/\.(gguf|safetensors)$/i, "").toLowerCase(),
}));

// launch-plan 的 getSetting 读的是真 DB（测试临时数据目录），这里换成与 llama 测试同一份
// 内存 settings，保证「缓存 key 由设置拼出」这一环与 buildArgs 读到的是同一份值。
// 注意：mock.module 必须**晚于** llama 的导入（llama 在测试文件末尾才 await import，先跑到这里），
// 否则 mock 对 llama 内已冻结的绑定无效。这里 import 只是拿真实函数引用供 seedPlan 用。

const realStats = await import("../stats");
mock.module("../stats", () => ({
  ...realStats,
  markServerStarted: () => {},
}));

const { LlamaRuntime } = await import("./llama");

// T4e 回读用例的假模块（proc / app-log / llama-flash-attn）在模块顶层导入真实引用，
// beforeAll 里再 mock.module 叠覆盖、afterAll 换回。
const realProc = await import("./proc");
const realAppLog = await import("../app-log");
const realFlashAttn = await import("./llama-flash-attn");

const tmpDir = mkdtempSync(join(tmpdir(), "llama-runtime-test-"));
const chatModel = join(tmpDir, "e2e-chat.gguf");
const embedModel = join(tmpDir, "wemm-emb.gguf");
writeFileSync(chatModel, "gguf");
writeFileSync(embedModel, "gguf");

/**
 * mmproj 注入场景目录：每个目录自包含（模型 + 不同投影文件组合），
 * 验证嵌入实例的自动配对与选择规则（多文件优先 f16）。
 */
function scenarioDir(name: string, files: string[]): string {
  const dir = join(tmpDir, name);
  mkdirSync(dir, { recursive: true });
  for (const f of files) writeFileSync(join(dir, f), "gguf");
  return dir;
}

const mmprojBothDir = scenarioDir("mmproj-both", ["model.gguf", "mmproj-f16.gguf", "mmproj-bf16.gguf"]);
const mmprojBf16OnlyDir = scenarioDir("mmproj-bf16-only", ["model.gguf", "mmproj-bf16.gguf"]);
const mmprojNoneDir = scenarioDir("mmproj-none", ["model.gguf"]);

/** 模块级共用：一份中性计划（T4e 回读用例与 auto-tune 用例都注入缓存）。 */
function makePlan(overrides: Partial<LaunchPlan> = {}): LaunchPlan {
  return {
    ctxTokens: 131072,
    ctxPerSlot: 43690,
    parallel: 3,
    batch: 1024,
    ubatch: 256,
    cacheTypeK: "q8_0",
    cacheTypeV: "q8_0",
    flashAttn: true,
    kvUnified: true,
    gpuLayers: null,
    fits: true,
    estimates: {
      budgetBytes: Math.round(7.8 * 1024 ** 3),
      weightsBytes: 4 * 1024 ** 3,
      kvBytes: Math.round(4.8 * 1024 ** 3),
      computeBufferBytes: 64 * 1024 * 1024,
      ctxComputeBytes: 128 * 1024 * 1024,
      totalBytes: Math.round(9.0 * 1024 ** 3),
      overflowBytes: 0,
    },
    reasons: [{ code: "budget.vram" }, { code: "ctx.reduced" }],
    ...overrides,
  };
}

/** 按 llama 自己的 key 规则把计划注进缓存（模型必须是临时目录里真存在的 .gguf）。 */
function seedPlan(modelPath: string, plan: LaunchPlan): void {
  __setLaunchPlanForTest(
    buildLaunchPlanKeyFromSettings(
      modelPath,
      (k) => SETTINGS[k] ?? "",
      effectiveFlashAttnForPlan(SETTINGS.SERVER_FLASH_ATTN, undefined),
    ),
    plan,
  );
}

/** 与实现同一规则的二进制解析（快照断言需要完整命令行）。 */
const bin =
  ["/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"].find((p) => existsSync(p)) ??
  "llama-server";

function setChatSettings() {
  SETTINGS.SERVER_PORT = "18400";
  SETTINGS.SERVER_CTX_SIZE = "8192";
  SETTINGS.SERVER_IMAGE_MAX_TOKENS = "2048";
  SETTINGS.SERVER_BATCH_SIZE = "256";
  SETTINGS.SERVER_UBATCH_SIZE = "64";
  SETTINGS.SERVER_PARALLEL = "1";
  SETTINGS.SERVER_TEMP = "0.1";
  SETTINGS.SERVER_TOP_P = "0.8";
  SETTINGS.SERVER_CACHE_TYPE_K = "q8_0";
  SETTINGS.SERVER_CACHE_TYPE_V = "q8_0";
  // 显式重置 GPU 层数为默认哨兵（-1 = 交给引擎），防止测试间串状态影响自动推算分支。
  SETTINGS.SERVER_GPU_LAYERS = "-1";
}

beforeEach(() => {
  for (const key of Object.keys(SETTINGS)) delete SETTINGS[key];
});

describe("buildCommandLine / chat", () => {
  test("chat 命令行快照（纯文本模型：没有投影文件就不发 --image-max-tokens / --no-mmproj-offload）", () => {
    setChatSettings();
    const rt = new LlamaRuntime();
    const cmd = rt.buildCommandLine(chatModel);
    expect(cmd).toBe(
      `${bin} -m ${chatModel} --alias e2e-chat --host 127.0.0.1 --port 18400 --ctx-size 8192` +
        " --parallel 1 --batch-size 256 --ubatch-size 64" +
        " --cache-type-k q8_0 --cache-type-v q8_0 --repeat-penalty 1 --repeat-last-n 256" +
        " --temp 0.1 --top-p 0.8 --top-k 40 --min-p 0.05 --presence-penalty 0",
    );
  });

  test("加载模式：设置值不会以原始形态进 argv（白名单之外一律不发，防参数注入）", () => {
    setChatSettings();
    // 这个值最终进 argv，所以合法取值是白名单；手改设置行塞进来的坏值必须被丢掉。
    SETTINGS.SERVER_LOAD_MODE = "--evil-flag";
    const cmd = new LlamaRuntime().buildCommandLine(chatModel);
    expect(cmd).not.toContain("--evil-flag");
    // 合法值在「还没探测过这台 llama-server」时也不发 —— 宁可按默认启动，不赌开关存在
    // （探测结果按二进制路径缓存，一旦启动过就会带上；映射规则见 llama-load-mode.test.ts）。
    SETTINGS.SERVER_LOAD_MODE = "mlock";
    expect(new LlamaRuntime().buildCommandLine(chatModel)).not.toContain("--evil-flag");
  });

  test("采样参数：没有按模型 / 模型自带 / 家族推荐时走全局设置（ENG-04，解析见 model-sampling）", () => {
    setChatSettings();
    const base = new LlamaRuntime().buildCommandLine(chatModel);
    expect(base).toContain("--temp 0.1");
    expect(base).toContain("--top-p 0.8");
    // 全局设置改了，发出去的跟着改
    SETTINGS.SERVER_TOP_K = "7";
    SETTINGS.SERVER_REPEAT_PENALTY = "1.3";
    SETTINGS.SERVER_MIN_P = "0.1";
    SETTINGS.SERVER_PRESENCE_PENALTY = "1.5";
    const cmd = new LlamaRuntime().buildCommandLine(chatModel);
    expect(cmd).toContain("--top-k 7");
    expect(cmd).toContain("--repeat-penalty 1.3");
    expect(cmd).toContain("--min-p 0.1");
    expect(cmd).toContain("--presence-penalty 1.5");
    // 空串 = 没设过，落回默认（而不是把参数发成空）
    SETTINGS.SERVER_TOP_K = "";
    expect(new LlamaRuntime().buildCommandLine(chatModel)).toMatch(/--top-k \d+/);
  });
});

describe("buildCommandLine / embedding", () => {
  test("embedding 实例：追加 --embeddings --pooling，裁剪聊天采样参数，端口用 overrides", () => {
    setChatSettings();
    const rt = new LlamaRuntime({
      model: embedModel,
      port: "18500",
      servedName: "wemm",
      purpose: "embedding",
    });
    const cmd = rt.buildCommandLine();
    expect(cmd).toBe(
      `${bin} -m ${embedModel} --alias wemm --host 127.0.0.1 --port 18500 --ctx-size 8192` +
        " --parallel 1 --batch-size 8192 --ubatch-size 8192 --cache-type-k q8_0 --cache-type-v q8_0" +
        " --embeddings --pooling last",
    );
  });

  test("嵌入模式物理 batch 取 ctx-size：超过它的输入会让 llama.cpp 崩进程（KB 导入即崩的根因）", () => {
    setChatSettings();
    const mk = () => new LlamaRuntime({ model: embedModel, purpose: "embedding" }).buildCommandLine();
    // ctx 8192 → batch/ubatch 8192（不能沿用聊天调优的 256/64：--embeddings 下
    // llama.cpp 强制 n_batch = n_ubatch，512 的物理 batch 塞不下真实文档）
    const cmd = mk();
    expect(cmd).toContain("--batch-size 8192 --ubatch-size 8192");
    // ctx 调大时 batch 跟着走，不留静默上限
    SETTINGS.SERVER_CTX_SIZE = "32768";
    expect(mk()).toContain("--batch-size 32768 --ubatch-size 32768");
    // 聊天路径不受影响（仍是聊天调优的 256/64）
    expect(new LlamaRuntime({ model: chatModel, servedName: "chat" }).buildCommandLine()).toContain(
      "--batch-size 256 --ubatch-size 64",
    );
  });

  test("无 overrides.port 时端口回落 EMBEDDING_PORT（18190 段，③-A1a）", () => {
    SETTINGS.EMBEDDING_PORT = "18190";
    const rt = new LlamaRuntime({ model: embedModel, purpose: "embedding" });
    const cmd = rt.buildCommandLine();
    expect(cmd).toContain("--port 18190");
    // alias 未显式给时按文件名 slug 生成。
    expect(cmd).toContain("--alias wemm-emb");
  });

  test("pooling 跟随设置键（EMBEDDING_POOLING）", () => {
    SETTINGS.EMBEDDING_POOLING = "mean";
    const rt = new LlamaRuntime({
      model: embedModel,
      port: "18500",
      purpose: "embedding",
    });
    const cmd = rt.buildCommandLine();
    expect(cmd).toContain("--pooling mean");
    expect(cmd).not.toContain("--temp");
  });

  test("裁剪清单：--temp/--top-p/--repeat-penalty/--repeat-last-n/--image-max-tokens 都不发", () => {
    setChatSettings();
    const rt = new LlamaRuntime({ model: embedModel, port: "18500", purpose: "embedding" });
    const cmd = rt.buildCommandLine();
    for (const flag of ["--temp", "--top-p", "--repeat-penalty", "--repeat-last-n", "--image-max-tokens"]) {
      expect(cmd).not.toContain(flag);
    }
    // 保留清单：上下文 / batch / 缓存类型仍在。
    for (const flag of ["--ctx-size", "--batch-size", "--ubatch-size", "--cache-type-k", "--cache-type-v"]) {
      expect(cmd).toContain(flag);
    }
  });

  test("purpose 不传即 chat：overrides 只差端口时命令仍是聊天形态", () => {
    setChatSettings();
    const rt = new LlamaRuntime({ model: chatModel, port: "18405" });
    const cmd = rt.buildCommandLine();
    expect(cmd).toContain("--port 18405");
    expect(cmd).toContain("--temp 0.1");
    expect(cmd).not.toContain("--embeddings");
  });
});

describe("buildArgs / mmproj 注入", () => {
  /** 嵌入实例命令行：模型与投影文件同目录的场景。 */
  function embedCmd(modelPath: string): string {
    setChatSettings();
    return new LlamaRuntime({
      model: modelPath,
      port: "18500",
      purpose: "embedding",
    }).buildCommandLine();
  }
  test("同目录有 f16 + bf16 → 注入一个 --mmproj 且选 f16", () => {
    const cmd = embedCmd(join(mmprojBothDir, "model.gguf"));
    expect(cmd).toContain(`--mmproj ${join(mmprojBothDir, "mmproj-f16.gguf")}`);
    expect(cmd.split("--mmproj").length).toBe(2);
  });

  test("只有 bf16 → 选它（f16 缺席时字典序首个兜底）", () => {
    const cmd = embedCmd(join(mmprojBf16OnlyDir, "model.gguf"));
    expect(cmd).toContain(`--mmproj ${join(mmprojBf16OnlyDir, "mmproj-bf16.gguf")}`);
  });

  test("同目录无 mmproj → 不注入（行为与旧版一致）", () => {
    const cmd = embedCmd(join(mmprojNoneDir, "model.gguf"));
    expect(cmd).not.toContain("--mmproj");
  });

  test("聊天实例也按同目录配对 --mmproj（本地视觉模型要能看图），并带上视觉参数", () => {
    setChatSettings();
    const rt = new LlamaRuntime({ model: join(mmprojBothDir, "model.gguf"), port: "18406" });
    const cmd = rt.buildCommandLine();
    expect(cmd).toContain(`--mmproj ${join(mmprojBothDir, "mmproj-f16.gguf")}`);
    expect(cmd.split("--mmproj ").length).toBe(2);
    expect(cmd).toContain("--image-max-tokens 2048");
    expect(cmd).toContain("--no-mmproj-offload");
  });

  test("聊天实例同目录无投影 → 不发 --mmproj / --image-max-tokens / --no-mmproj-offload", () => {
    setChatSettings();
    const cmd = new LlamaRuntime({ model: join(mmprojNoneDir, "model.gguf"), port: "18407" }).buildCommandLine();
    expect(cmd).not.toContain("--mmproj");
    expect(cmd).not.toContain("--image-max-tokens");
    expect(cmd).not.toContain("--no-mmproj-offload");
  });

  test("模型路径是目录（扫描器聚合的 GGUF 仓库）→ -m 换成主文件，投影从同目录配对", () => {
    setChatSettings();
    const cmd = new LlamaRuntime({ model: mmprojBothDir, port: "18408" }).buildCommandLine();
    expect(cmd).toContain(`-m ${join(mmprojBothDir, "model.gguf")} `);
    expect(cmd).toContain(`--mmproj ${join(mmprojBothDir, "mmproj-f16.gguf")}`);
    // buildCommandLine(modelOverride) 与 resolveModel 同一规则
    expect(new LlamaRuntime().buildCommandLine(mmprojBothDir)).toContain(`-m ${join(mmprojBothDir, "model.gguf")} `);
  });

  test("hf ref 聊天实例保持旧行为：照发 --image-max-tokens / --no-mmproj-offload", () => {
    setChatSettings();
    const cmd = new LlamaRuntime({ model: "unsloth/Qwen2.5-VL-7B-GGUF", port: "18409" }).buildCommandLine();
    expect(cmd).toContain("--image-max-tokens 2048");
    expect(cmd).toContain("--no-mmproj-offload");
    expect(cmd).not.toContain("--mmproj ");
  });

  test("mmproj 字节数进规划 key（预算要扣掉投影文件），无投影时为 null", () => {
    const withMm = buildLaunchPlanKeyFromSettings(join(mmprojBothDir, "model.gguf"), () => "", false);
    expect(withMm.mmprojBytes).toBe(4); // 场景文件内容是 "gguf"
    const without = buildLaunchPlanKeyFromSettings(join(mmprojNoneDir, "model.gguf"), () => "", false);
    expect(without.mmprojBytes).toBeNull();
  });

  test("hf ref 嵌入模型（-hf 自管缓存）不注入 mmproj（Non-Goal）", () => {
    setChatSettings();
    const rt = new LlamaRuntime({
      model: "unsloth/gme-Qwen2-VL-2B-GGUF",
      port: "18500",
      purpose: "embedding",
    });
    const cmd = rt.buildCommandLine();
    expect(cmd).toContain("-hf unsloth/gme-Qwen2-VL-2B-GGUF");
    expect(cmd).not.toContain("--mmproj");
  });
});

describe("buildArgs / 自动启动参数（SERVER_AUTO_TUNE）", () => {
  /** 一份可用的计划：只改与断言相关的字段，其余给中性值。 */
  function makePlan(overrides: Partial<LaunchPlan> = {}): LaunchPlan {
    return {
      ctxTokens: 131072,
      ctxPerSlot: 43690,
      parallel: 3,
      batch: 1024,
      ubatch: 256,
      cacheTypeK: "q8_0",
      cacheTypeV: "q8_0",
      flashAttn: true,
      kvUnified: true,
      gpuLayers: null,
      fits: true,
      estimates: {
        budgetBytes: Math.round(7.8 * 1024 ** 3),
        weightsBytes: 4 * 1024 ** 3,
        kvBytes: Math.round(4.8 * 1024 ** 3),
        computeBufferBytes: 64 * 1024 * 1024,
        ctxComputeBytes: 128 * 1024 * 1024,
        totalBytes: Math.round(9.0 * 1024 ** 3),
        overflowBytes: 0,
      },
      reasons: [{ code: "budget.vram" }, { code: "ctx.reduced" }],
      ...overrides,
    };
  }

  /**
   * 把一份计划按 llama 自己的 key 规则注进缓存。key 与 llama.ts 共用同一构造函数
   * （buildLaunchPlanKeyFromSettings），从根上消灭两份规则各自演化导致的不一致。
   * 模型必须是临时目录里的真 .gguf（cachedLaunchPlan 会重新 stat 比对指纹，文件不存在会未命中）。
   */
  function seedPlan(modelPath: string, plan: LaunchPlan): void {
    __setLaunchPlanForTest(
      buildLaunchPlanKeyFromSettings(
        modelPath,
        (k) => SETTINGS[k] ?? "",
        effectiveFlashAttnForPlan(SETTINGS.SERVER_FLASH_ATTN, undefined),
      ),
      plan,
    );
  }

  test("开启但缓存没有计划 → argv 与关闭时完全一致（任何失败都不能挡住启动）", () => {
    setChatSettings();
    SETTINGS.SERVER_AUTO_TUNE = "1";
    const withAuto = new LlamaRuntime({ model: chatModel, port: "18410" }).buildArgs(
      { kind: "local", path: chatModel, alias: "e2e-chat" },
      DEFAULT_CUSTOM_SERVER_ARGS,
    );
    SETTINGS.SERVER_AUTO_TUNE = "0";
    const without = new LlamaRuntime({ model: chatModel, port: "18410" }).buildArgs(
      { kind: "local", path: chatModel, alias: "e2e-chat" },
      DEFAULT_CUSTOM_SERVER_ARGS,
    );
    expect(withAuto).toEqual(without);
  });

  test("开启且缓存有计划 → ctx / batch / ubatch 用计划值，parallel 用设置值", () => {
    setChatSettings();
    SETTINGS.SERVER_AUTO_TUNE = "1";
    SETTINGS.SERVER_CTX_SIZE = "8192";
    SETTINGS.SERVER_BATCH_SIZE = "256";
    SETTINGS.SERVER_UBATCH_SIZE = "64";
    SETTINGS.SERVER_PARALLEL = "3";
    seedPlan(chatModel, makePlan());

    const rt = new LlamaRuntime({ model: chatModel, port: "18411" });
    const cmd = rt.buildCommandLine(chatModel);
    expect(cmd).toContain("--ctx-size 131072");
    expect(cmd).toContain("--batch-size 1024");
    expect(cmd).toContain("--ubatch-size 256");
    // 并发是用户的业务选择：计划里的 parallel 不覆盖设置值
    expect(cmd).toContain("--parallel 3");
    // 缓存类型保持设置值（它们参与 key 计算，不参与覆盖）
    expect(cmd).toContain("--cache-type-k q8_0");
    expect(cmd).toContain("--cache-type-v q8_0");
  });

  test("gpuLayers 只在 SERVER_GPU_LAYERS == -1 时采纳；用户填了具体数字就听用户的", () => {
    setChatSettings();
    SETTINGS.SERVER_AUTO_TUNE = "1";
    const plan = makePlan({ gpuLayers: 20 });

    // 用户未显式指定（默认 -1）→ 采纳计划的 20
    seedPlan(chatModel, plan);
    let rt = new LlamaRuntime({ model: chatModel, port: "18412" });
    expect(rt.buildCommandLine(chatModel)).toContain("--n-gpu-layers 20");

    // 用户填了具体数字 → 听用户的
    SETTINGS.SERVER_GPU_LAYERS = "28";
    rt = new LlamaRuntime({ model: chatModel, port: "18412" });
    expect(rt.buildCommandLine(chatModel)).toContain("--n-gpu-layers 28");
    expect(rt.buildCommandLine(chatModel)).not.toContain("--n-gpu-layers 20");
  });

  test("FA 实测值：新实例（内存态未知）回落持久化的 SERVER_FLASH_ATTN_EFFECTIVE，与预览同一 key", () => {
    setChatSettings();
    SETTINGS.SERVER_AUTO_TUNE = "1";
    SETTINGS.SERVER_FLASH_ATTN = "auto";
    SETTINGS.SERVER_FLASH_ATTN_EFFECTIVE = "on";
    // 预览 RPC 的 key 构造：设置 + 持久化实测
    const previewKey = buildLaunchPlanKeyFromSettings(
      chatModel,
      (k) => SETTINGS[k] ?? "",
      effectiveFlashAttnForPlan(SETTINGS.SERVER_FLASH_ATTN, SETTINGS.SERVER_FLASH_ATTN_EFFECTIVE as "on"),
    );
    expect(previewKey.flashAttn).toBe(true);
    __setLaunchPlanForTest(previewKey, makePlan({ ctxTokens: 65536 }));
    // 全新实例（从没启动过）也要命中预览那份计划，而不是按「FA 关」另算一份
    const cmd = new LlamaRuntime({ model: chatModel, port: "18415" }).buildCommandLine(chatModel);
    expect(cmd).toContain("--ctx-size 65536");
    // 持久化值是垃圾 → 当未知（按关），不命中 FA=on 那份
    SETTINGS.SERVER_FLASH_ATTN_EFFECTIVE = "garbage";
    expect(new LlamaRuntime({ model: chatModel, port: "18415" }).buildCommandLine(chatModel)).not.toContain(
      "--ctx-size 65536",
    );
  });

  test("buildCommandLine 与 buildArgs 在同一缓存状态下产生相同的参数序列", () => {
    setChatSettings();
    SETTINGS.SERVER_AUTO_TUNE = "1";
    seedPlan(chatModel, makePlan({ gpuLayers: 12 }));

    const rt = new LlamaRuntime({ model: chatModel, port: "18413" });
    const viaBuildArgs = rt.buildArgs(
      { kind: "local", path: chatModel, alias: "e2e-chat" },
      rt.getProfileServerArgs(),
    );
    // buildArgs 与 buildCommandLine 内部是同一份函数，参数序列必然逐字节一致
    // （「复制的命令」与实际发出去的那条不会分叉）。
    const viaCommandLine = rt.buildCommandLine(chatModel).replace(/^\S+\s+/, ""); // 去掉二进制路径
    expect(viaCommandLine).toBe(viaBuildArgs.join(" "));

    // 真正有价值的一致性：同一 key 写缓存后，两个入口读到的计划相同 ——
    // buildCommandLine 里那份（含 GPU 层建议）与 buildArgs 里那份逐字段一致。
    const plan = makePlan({ gpuLayers: 12 });
    seedPlan(chatModel, plan);
    const rt2 = new LlamaRuntime({ model: chatModel, port: "18414" });
    const fromCommandLine = rt2.buildCommandLine(chatModel);
    expect(fromCommandLine).toContain("--n-gpu-layers 12");
    expect(fromCommandLine).toContain("--ctx-size 131072");
    const fromArgs = rt2.buildArgs(
      { kind: "local", path: chatModel, alias: "e2e-chat" },
      rt2.getProfileServerArgs(),
    );
    expect(fromArgs.join(" ")).toBe(fromCommandLine.replace(/^\S+\s+/, ""));
  });
});

describe("buildArgs / flash attention（T4d）", () => {
  // 探测结果按二进制路径进程级缓存（llama-flash-attn.ts）：这些测试用「本机解析出的那个
  // llama-server」同一个路径注缓存，测完 clearServerHelpSupportCache 恢复未探测状态，
  // 与真实「启动前」一致（未探测 = none，一个参数都不发）。
  const { setCachedServerHelpSupport, clearServerHelpSupportCache } =
    require("./llama-flash-attn") as typeof import("./llama-flash-attn");

  const resolvedBin = () =>
    ["/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"].find((p) => existsSync(p)) ??
    "llama-server";

  const chatArgs = (rt: InstanceType<typeof LlamaRuntime>) =>
    rt.buildArgs(
      { kind: "local", path: chatModel, alias: "e2e-chat" },
      DEFAULT_CUSTOM_SERVER_ARGS,
    );

  /**
   * 断言 argv（string[]）里 `flag` 紧跟 `value`。
bun 的数组 toContain 是精确匹配（单元素），
   * 不能用 toContain("--flash-attn auto") 这种字符串子序列写法，所以拆成两个断言。
   */
  function expectFlagWithValue(args: string[], flag: string, value: string) {
    const i = args.indexOf(flag);
    expect(i).toBeGreaterThan(-1);
    expect(args[i + 1]).toBe(value);
  }

  /** 断言 argv 里**没有**该 flag（精确元素匹配，不是子串）。 */
  function expectNoFlag(args: string[], flag: string) {
    expect(args).not.toContain(flag);
  }

  afterEach(() => clearServerHelpSupportCache());

  test("tristate：--help 含取值列表 → auto 发 --flash-attn auto（#1）", () => {
    setChatSettings();
    const rt = new LlamaRuntime({ model: chatModel, port: "18420" });
    // 未探测状态先验证不发（#4 的另一半）
    expectNoFlag(chatArgs(rt), "--flash-attn");

    // 注一次「探测成功：三态」（不跑真 --help，保证确定性）
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate" });

    // 默认 auto
    expectFlagWithValue(chatArgs(rt), "--flash-attn", "auto");
    // on / off 原样发
    SETTINGS.SERVER_FLASH_ATTN = "on";
    expectFlagWithValue(chatArgs(rt), "--flash-attn", "on");
    SETTINGS.SERVER_FLASH_ATTN = "off";
    expectFlagWithValue(chatArgs(rt), "--flash-attn", "off");
  });

  test("boolean：老版只有 -fa, --flash-attn → on 发不带值的开关，auto 不发（#2）", () => {
    setChatSettings();
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "legacy", flashAttn: "boolean" });
    const rt = new LlamaRuntime({ model: chatModel, port: "18421" });

    SETTINGS.SERVER_FLASH_ATTN = "auto";
    const autoArgs = chatArgs(rt);
    expectNoFlag(autoArgs, "--flash-attn");

    SETTINGS.SERVER_FLASH_ATTN = "on";
    const onArgs = chatArgs(rt);
    const idx = onArgs.indexOf("--flash-attn");
    expect(idx).toBeGreaterThan(-1);
    expect(onArgs[idx + 1]).not.toBe("on"); // 不带值（下一段是别的东西，比如 -m 或 --no-mmproj-offload）

    SETTINGS.SERVER_FLASH_ATTN = "off";
    expectNoFlag(chatArgs(rt), "--flash-attn");
  });

  test("none：--help 里没有 flash-attn → argv 完全不包含（#3）", () => {
    setChatSettings();
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "none" });
    const rt = new LlamaRuntime({ model: chatModel, port: "18422" });
    for (const v of ["auto", "on", "off"]) {
      SETTINGS.SERVER_FLASH_ATTN = v;
      expectNoFlag(chatArgs(rt), "--flash-attn");
    }
  });

  test("未探测状态：buildCommandLine 与 buildArgs 都不发该参数（#4）", () => {
    setChatSettings();
    // clearServerHelpSupportCache 在 afterEach 里已跑；再清一次保证
    clearServerHelpSupportCache();
    const rt = new LlamaRuntime({ model: chatModel, port: "18423" });
    SETTINGS.SERVER_FLASH_ATTN = "on";
    const cmd = rt.buildCommandLine(chatModel);
    expect(cmd).not.toContain("--flash-attn");
    const args = chatArgs(rt);
    expectNoFlag(args, "--flash-attn");
    // 两者一致（同一份缓存、同一份函数）
    expect(cmd.replace(/^\S+\s+/, "")).toBe(args.join(" "));
  });

  test("--kv-unified：parallel > 1 且 --help 认这个开关才发；没探过 / 不认 / parallel=1 都不发", () => {
    setChatSettings();
    SETTINGS.SERVER_PARALLEL = "4";
    const rt = new LlamaRuntime({ model: chatModel, port: "18426" });
    // 没探过：不赌开关存在
    expectNoFlag(chatArgs(rt), "--kv-unified");
    // 探到不支持
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate", kvUnified: false });
    expectNoFlag(chatArgs(rt), "--kv-unified");
    // 探到支持
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate", kvUnified: true });
    expect(chatArgs(rt)).toContain("--kv-unified");
    // 复制的命令与实际参数一致
    expect(rt.buildCommandLine(chatModel)).toContain("--kv-unified");
    // parallel = 1 时没有可共享的 slot，不发
    SETTINGS.SERVER_PARALLEL = "1";
    expectNoFlag(chatArgs(rt), "--kv-unified");
  });

  test("--kv-unified 与自动计划：听计划的 kvUnified；计划 key 带上探测到的支持情况", () => {
    setChatSettings();
    SETTINGS.SERVER_AUTO_TUNE = "1";
    SETTINGS.SERVER_PARALLEL = "3";
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate", kvUnified: true });
    const key = buildLaunchPlanKeyFromSettings(chatModel, (k) => SETTINGS[k] ?? "", false);
    expect(key.supportsKvUnified).toBe(true);
    seedPlan(chatModel, makePlan({ kvUnified: false }));
    const rt = new LlamaRuntime({ model: chatModel, port: "18427" });
    expectNoFlag(chatArgs(rt), "--kv-unified");
    seedPlan(chatModel, makePlan({ kvUnified: true }));
    expect(chatArgs(rt)).toContain("--kv-unified");
    // 不支持时 key 的 supportsKvUnified=false（规划器按每 slot 均分计价）
    clearServerHelpSupportCache();
    expect(buildLaunchPlanKeyFromSettings(chatModel, (k) => SETTINGS[k] ?? "", false).supportsKvUnified).toBe(false);
  });

  test("SERVER_FLASH_ATTN=off（tristate）→ argv 含 --flash-attn off（#5）", () => {
    setChatSettings();
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate" });
    SETTINGS.SERVER_FLASH_ATTN = "off";
    const rt = new LlamaRuntime({ model: chatModel, port: "18424" });
    expectFlagWithValue(chatArgs(rt), "--flash-attn", "off");
  });

  test("设置值白名单：非法值回落 auto，不产生 argv 注入（tristate）", () => {
    setChatSettings();
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate" });
    SETTINGS.SERVER_FLASH_ATTN = "--evil-flag";
    const rt = new LlamaRuntime({ model: chatModel, port: "18425" });
    const args = chatArgs(rt);
    expectNoFlag(args, "--evil-flag");
    // 回落 auto
    expectFlagWithValue(args, "--flash-attn", "auto");
  });
});

// ---------- 按模型参数（model_params 表，覆盖全局设置） ----------
// 放在 T4e 之前：那一组会 mock.module 替换 llama-flash-attn，之后注的探测缓存 llama.ts 读不到。

describe("buildArgs / 按模型参数", () => {
  const { setModelParams, clearModelParams } =
    require("../db/model-params") as typeof import("../db/model-params");
  const { setCachedServerHelpSupport, clearServerHelpSupportCache } =
    require("./llama-flash-attn") as typeof import("./llama-flash-attn");
  const { splitShellArgs } = require("./shell-args") as typeof import("./shell-args");

  const resolvedBin = () =>
    ["/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"].find((p) => existsSync(p)) ??
    "llama-server";
  const argsFor = (rt: InstanceType<typeof LlamaRuntime>) =>
    rt.buildArgs({ kind: "local", path: chatModel, alias: "e2e-chat" }, DEFAULT_CUSTOM_SERVER_ARGS, chatModel);
  const valueOf = (args: string[], flag: string) => {
    const i = args.lastIndexOf(flag);
    return i === -1 ? undefined : args[i + 1];
  };

  afterEach(() => {
    clearModelParams(chatModel);
    clearModelParams(embedModel);
    clearServerHelpSupportCache();
  });

  test("启动参数：按模型 > 全局（ctx / parallel / gpuLayers / KV 类型 / FA）", () => {
    setChatSettings();
    SETTINGS.SERVER_GPU_LAYERS = "28";
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate" });
    SETTINGS.SERVER_FLASH_ATTN = "auto";
    setModelParams(chatModel, {
      ctxSize: 32768,
      parallel: 2,
      gpuLayers: 10,
      cacheTypeK: "bf16",
      cacheTypeV: "q5_1",
      flashAttn: "off",
    });
    const args = argsFor(new LlamaRuntime({ model: chatModel, port: "18430" }));
    expect(valueOf(args, "--ctx-size")).toBe("32768");
    expect(valueOf(args, "--parallel")).toBe("2");
    expect(valueOf(args, "--n-gpu-layers")).toBe("10");
    // bf16 / q5_1 是 llama-server --help 认的值（argv 白名单已与规划器 KV 表对齐）
    expect(valueOf(args, "--cache-type-k")).toBe("bf16");
    expect(valueOf(args, "--cache-type-v")).toBe("q5_1");
    expect(valueOf(args, "--flash-attn")).toBe("off");

    // 清掉按模型的 → 回到全局
    clearModelParams(chatModel);
    const g = argsFor(new LlamaRuntime({ model: chatModel, port: "18430" }));
    expect(valueOf(g, "--ctx-size")).toBe("8192");
    expect(valueOf(g, "--n-gpu-layers")).toBe("28");
    expect(valueOf(g, "--cache-type-k")).toBe("q8_0");
    expect(valueOf(g, "--flash-attn")).toBe("auto");
  });

  test("gpuLayers = -1（按模型显式交给引擎）盖过全局的固定层数，采纳计划建议", () => {
    setChatSettings();
    SETTINGS.SERVER_GPU_LAYERS = "28";
    setModelParams(chatModel, { gpuLayers: -1 });
    expect(argsFor(new LlamaRuntime({ model: chatModel, port: "18431" }))).not.toContain("--n-gpu-layers");
    SETTINGS.SERVER_AUTO_TUNE = "1";
    __setLaunchPlanForTest(
      buildLaunchPlanKeyFromSettings(chatModel, (k) => SETTINGS[k] ?? "", false, {
        modelParams: { gpuLayers: -1 },
      }),
      makePlan({ gpuLayers: 20 }),
    );
    expect(valueOf(argsFor(new LlamaRuntime({ model: chatModel, port: "18431" })), "--n-gpu-layers")).toBe("20");
  });

  test("自动规划开着时按模型的 ctx 作为 ctxOverride 进 key：命中的是尊重该窗口的那份计划", () => {
    setChatSettings();
    SETTINGS.SERVER_AUTO_TUNE = "1";
    SETTINGS.SERVER_PARALLEL = "3";
    // 全局设置的那份计划（ctxOverride = null）
    seedPlan(chatModel, makePlan({ ctxTokens: 131072, batch: 1024 }));
    setModelParams(chatModel, { ctxSize: 16384 });
    const key = buildLaunchPlanKeyFromSettings(chatModel, (k) => SETTINGS[k] ?? "", false, {
      modelParams: { ctxSize: 16384 },
    });
    expect(key.ctxOverride).toBe(16384);
    const rt = new LlamaRuntime({ model: chatModel, port: "18432" });
    // 还没算过带 ctxOverride 的计划：不能误用全局那份 131072，直接用按模型的窗口
    expect(valueOf(argsFor(rt), "--ctx-size")).toBe("16384");
    expect(valueOf(argsFor(rt), "--batch-size")).toBe("256");
    // 算过之后：ctx 是计划的（规划器对用户显式窗口不缩），其余参数照样自动（batch 用计划值）
    __setLaunchPlanForTest(key, makePlan({ ctxTokens: 16384, batch: 2048 }));
    const planned = argsFor(rt);
    expect(valueOf(planned, "--ctx-size")).toBe("16384");
    expect(valueOf(planned, "--batch-size")).toBe("2048");
  });

  test("按模型的 parallel / KV 类型进计划 key（计价用的是真正发出去的值）", () => {
    setChatSettings();
    const key = buildLaunchPlanKeyFromSettings(chatModel, (k) => SETTINGS[k] ?? "", false, {
      modelParams: { parallel: 4, cacheTypeK: "f16", cacheTypeV: "f16" },
    });
    expect(key.parallel).toBe(4);
    expect(key.cacheTypeK).toBe("f16");
    expect(key.cacheTypeV).toBe("f16");
    expect(key.ctxOverride).toBeNull();
  });

  test("采样：按模型 > 全局；min-p / presence-penalty 也发；嵌入实例整组不发", () => {
    setChatSettings();
    setModelParams(chatModel, { sampling: { temperature: 0.6, topK: 20, minP: 0, presencePenalty: 1.5 } });
    const args = argsFor(new LlamaRuntime({ model: chatModel, port: "18433" }));
    expect(valueOf(args, "--temp")).toBe("0.6");
    expect(valueOf(args, "--top-k")).toBe("20");
    expect(valueOf(args, "--min-p")).toBe("0");
    expect(valueOf(args, "--presence-penalty")).toBe("1.5");
    // 没覆盖的字段仍按全局
    expect(valueOf(args, "--top-p")).toBe("0.8");
    expect(args).toContain("--repeat-last-n");

    setModelParams(embedModel, { sampling: { temperature: 0.6 }, ctxSize: 4096 });
    const emb = new LlamaRuntime({ model: embedModel, port: "18434", purpose: "embedding" }).buildArgs(
      { kind: "local", path: embedModel, alias: "wemm-emb" },
      DEFAULT_CUSTOM_SERVER_ARGS,
      embedModel,
    );
    expect(emb).not.toContain("--temp");
    expect(emb).not.toContain("--min-p");
    // 嵌入实例的按模型启动参数照样生效（ctx 同时决定嵌入 batch）
    expect(valueOf(emb, "--ctx-size")).toBe("4096");
    expect(valueOf(emb, "--batch-size")).toBe("4096");
  });

  test("hf 引用模型也按 target 读按模型参数", () => {
    setChatSettings();
    const ref = "unsloth/Some-Model-GGUF:Q4_K_M";
    setModelParams(ref, { ctxSize: 12288, extraArgs: "--foo bar" });
    const args = new LlamaRuntime({ model: ref, port: "18435" }).buildArgs(
      { kind: "hf", ref },
      DEFAULT_CUSTOM_SERVER_ARGS,
      ref,
    );
    expect(valueOf(args, "--ctx-size")).toBe("12288");
    expect(args.slice(-2)).toEqual(["--foo", "bar"]);
    clearModelParams(ref);
  });

  test("思考开关：认 --reasoning 发 --reasoning on|off；不认时关思考回落 chat-template-kwargs；auto 不发", () => {
    setChatSettings();
    const rt = () => new LlamaRuntime({ model: chatModel, port: "18436" });
    setModelParams(chatModel, { thinking: "off" });
    // 没探过 → 按不支持：关思考走 kwargs
    let args = argsFor(rt());
    expect(args).not.toContain("--reasoning");
    expect(valueOf(args, "--chat-template-kwargs")).toBe('{"enable_thinking":false}');
    // 探到支持 → --reasoning off
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate", reasoning: true });
    args = argsFor(rt());
    expect(valueOf(args, "--reasoning")).toBe("off");
    expect(args).not.toContain("--chat-template-kwargs");
    setModelParams(chatModel, { thinking: "on" });
    expect(valueOf(argsFor(rt()), "--reasoning")).toBe("on");
    // 不支持时「开」不发（模板默认就是开）
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate", reasoning: false });
    args = argsFor(rt());
    expect(args).not.toContain("--reasoning");
    expect(args).not.toContain("--chat-template-kwargs");
    setModelParams(chatModel, { thinking: "auto" });
    setCachedServerHelpSupport(resolvedBin(), { loadMode: "load-mode", flashAttn: "tristate", reasoning: true });
    expect(argsFor(rt())).not.toContain("--reasoning");
  });

  test("追加参数：按 shell 引号切分，全局在前、按模型在后；复制的命令切回来与 argv 相等", () => {
    setChatSettings();
    SETTINGS.SERVER_EXTRA_ARGS = `--chat-template-kwargs '{"enable_thinking": false}' --seed 1`;
    setModelParams(chatModel, { extraArgs: `--seed 42 --system-prompt "hi \\"there\\""` });
    const rt = new LlamaRuntime({ model: chatModel, port: "18437" });
    const args = argsFor(rt);
    expect(args.slice(-8)).toEqual([
      "--chat-template-kwargs",
      '{"enable_thinking": false}',
      "--seed",
      "1",
      "--seed",
      "42",
      "--system-prompt",
      'hi "there"',
    ]);
    const cmd = rt.buildCommandLine(chatModel);
    expect(splitShellArgs(cmd).slice(1)).toEqual(args);
  });

  test("needsRestart：没在跑 → false", () => {
    setChatSettings();
    expect(new LlamaRuntime({ model: chatModel, port: "18438" }).needsRestart()).toBe(false);
  });
});

// ---------- T4e：启动后回读实测值（预测 → 实测闭环） ----------

describe("start / 回读实测值（T4e）", () => {
  const originalFetch = globalThis.fetch;

  type LogEventInput = Parameters<typeof realAppLog.logEvent>[0];
  let logEvents: LogEventInput[];
  let propsBehavior: "ok" | "fail";
  let fetchedUrls: string[] = [];
  let faLogLine: string;
  let realSpawn: typeof realProc.spawnServerProcess;
  let realProbe: typeof realFlashAttn.probeServerHelp;

  beforeAll(async () => {
    // 让 checkBinary 能找到“已安装”的 llama-server：在临时数据目录的托管路径造一个空文件。
    // （probeServerHelp 已桩掉，这个文件不会被真正执行。）
    const binPath = llamaCppBinaryPath();
    mkdirSync(dirname(binPath), { recursive: true });
    if (!existsSync(binPath)) writeFileSync(binPath, "#!/bin/sh\n");

    // 桩只叠覆盖、其余保持真实（mock-hygiene）；app-log 收进数组供断言。
    await mockModulePartial<typeof import("../app-log")>("./app-log", {
      logEvent: (input: LogEventInput) => {
        logEvents.push(input);
        return {
          ...input,
          level: input.level ?? "info",
          seq: logEvents.length,
          ts: Date.now(),
          pid: 1,
        };
      },
    });
    logEvents = [];

    // 假 fetch：/health 永远 OK；/props 按用例行为（JSON / 抛错）。
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      fetchedUrls.push(url);
      if (url.endsWith("/health")) return new Response(null, { status: 200 });
      if (url.endsWith("/props")) {
        if (propsBehavior === "fail") throw new Error("connection refused (fake)");
        return new Response(
          JSON.stringify({
            default_generation_settings: { params: {}, n_ctx: 65536 },
            total_slots: 3,
          }),
          { status: 200 },
        );
      }
      throw new Error(`unexpected fetch in test: ${url}`);
    }) as typeof fetch;

    // 假子进程：把启动日志（含 FA 措辞）灌进 appendLog，exited 永不兑现（实例一直活着）。
    // stdout 用真 ReadableStream 把 faLogLine 吐出（pumpServerOutput 会读它），
    // stderr 空流。
    realSpawn = realProc.spawnServerProcess;
    mock.module("./proc", () => ({
      ...realProc,
      spawnServerProcess: () => {
        const makeStream = () =>
          new ReadableStream<Uint8Array>({
            start(controller) {
              const text = faLogLine;
              if (text) controller.enqueue(new TextEncoder().encode(text));
              controller.close();
            },
          });
        return {
          pid: 4242,
          exited: new Promise<number>(() => {}),
          stdout: makeStream(),
          stderr: makeStream(),
          kill: () => {},
        } as never;
      },
    }));

    // --help 探测桩（不真 spawn，避免测试里拉起外部二进制）。
    realProbe = realFlashAttn.probeServerHelp;
    mock.module("./llama-flash-attn", () => ({
      ...realFlashAttn,
      probeServerHelp: async () => ({ loadMode: "unknown", flashAttn: "none" as const }),
    }));
  });

  beforeEach(() => {
    logEvents = [];
    propsBehavior = "ok";
    faLogLine = "";
    fetchedUrls = [];
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
    mock.module("./proc", () => ({ ...realProc, spawnServerProcess: realSpawn }));
    mock.module("./llama-flash-attn", () => ({
      ...realFlashAttn,
      probeServerHelp: realProbe,
    }));
    mock.module("../app-log", () => ({ ...realAppLog }));
  });

  /** 起一个“模型文件存在的”实例并等启动完成（含回读）。 */
  async function startServer(port: number): Promise<{
    result: { ok: boolean; error?: string };
    status: string;
    logs: string;
  }> {
    const rt = new LlamaRuntime({ model: chatModel, port: String(port) });
    const result = await rt.start();
    await Bun.sleep(100); // 等假子进程的 appendLog 微任务兑现（FA 措辞进日志）
    return { result, status: rt.getStatus(), logs: rt.getLogs() };
  }

  const eventsBy = (name: string) => logEvents.filter((e) => e.event === name);

  test("/props 成功 → launch_plan.measured 的 actualCtxTotal = n_ctx × total_slots", async () => {
    propsBehavior = "ok";
    faLogLine = "llama_context: flash_attn            = enabled\n";
    // 让回读能带上预测字段：开自动并注入一份计划
    SETTINGS.SERVER_AUTO_TUNE = "1";
    seedPlan(chatModel, makePlan({ ctxTokens: 131072, ctxPerSlot: 43690 }));

    const { result, status, logs } = await startServer(18500);
    expect(result.ok).toBe(true);
    expect(status).toBe("running");
    expect(logs).toContain("flash_attn            = enabled");

    const measured = eventsBy("launch_plan.measured");
    expect(measured.length).toBe(1);
    const detail = (measured[0]?.detail ?? {}) as Record<string, number | string | null | undefined>;
    expect(detail.actualCtxPerSlot).toBe(65536);
    expect(detail.actualSlots).toBe(3);
    expect(detail.actualCtxTotal).toBe(196608); // 65536 × 3
    expect(detail.flashAttn).toBe("on");
    expect(detail.predictedCtx).toBe(131072);
    expect(detail.predictedPerSlot).toBe(43690);
    // 预测 131072 vs 实测 196608 相差 > 5% → 额外一条 mismatch
    const mismatch = eventsBy("launch_plan.mismatch");
    expect(mismatch.length).toBe(1);
    const mDetail = (mismatch[0]?.detail ?? {}) as Record<string, number>;
    expect(mDetail.predictedCtx).toBe(131072);
    expect(mDetail.actualCtxTotal).toBe(196608);
  });

  test("/props 失败 → 启动仍成功、状态 running，回读异常只记 readback_failed", async () => {
    propsBehavior = "fail";
    faLogLine = "";
    const { result, status } = await startServer(18501);
    expect(result.ok).toBe(true);
    expect(status).toBe("running");
    // fetch 抛错在回读内部被吞 → 记一条 readback_failed（warn），启动不受影响
    expect(eventsBy("launch_plan.readback_failed").length).toBe(1);
  });

  test("日志含 flash_attn = enabled → SERVER_FLASH_ATTN_EFFECTIVE 被写成 on", async () => {
    propsBehavior = "ok";
    faLogLine = "llama_context: flash_attn            = enabled\n";
    // 先放一个旧值，验证它会被覆盖
    realSettings.updateSettings({ SERVER_FLASH_ATTN_EFFECTIVE: "off" });
    const { result } = await startServer(18502);
    expect(result.ok).toBe(true);
    expect(realSettings.getSetting("SERVER_FLASH_ATTN_EFFECTIVE")).toBe("on");
  });

  test("嵌入实例（无 overrides.port）：健康检查与回读轮询的是 --port 那个嵌入端口", async () => {
    propsBehavior = "ok";
    SETTINGS.SERVER_PORT = "18600";
    SETTINGS.EMBEDDING_PORT = "18655";
    const rt = new LlamaRuntime({ model: embedModel, purpose: "embedding" });
    expect(rt.buildCommandLine()).toContain("--port 18655");
    const result = await rt.start();
    expect(result.ok).toBe(true);
    expect(fetchedUrls).toContain("http://localhost:18655/health");
    expect(fetchedUrls).toContain("http://localhost:18655/props");
    expect(fetchedUrls.some((u) => u.includes(":18600/"))).toBe(false);
  });

  test("日志判断不出 FA → 设置保持原值不变", async () => {
    propsBehavior = "ok";
    faLogLine = "llama_model_loader: - full model: qwen\n"; // 无任何 FA 措辞
    realSettings.updateSettings({ SERVER_FLASH_ATTN_EFFECTIVE: "off" });
    const { result } = await startServer(18503);
    expect(result.ok).toBe(true);
    expect(realSettings.getSetting("SERVER_FLASH_ATTN_EFFECTIVE")).toBe("off");
  });

  test("needsRestart：跑着时改了按模型参数 → true（不自动重启）；改回去 → false", async () => {
    const { setModelParams, clearModelParams } =
      require("../db/model-params") as typeof import("../db/model-params");
    propsBehavior = "ok";
    faLogLine = "";
    setChatSettings();
    const rt = new LlamaRuntime({ model: chatModel, port: "18504" });
    expect((await rt.start()).ok).toBe(true);
    expect(rt.needsRestart()).toBe(false);
    setModelParams(chatModel, { sampling: { temperature: 1.1 } });
    expect(rt.needsRestart()).toBe(true);
    expect(rt.getStatus()).toBe("running");
    clearModelParams(chatModel);
    expect(rt.needsRestart()).toBe(false);
    // 不 stop：假子进程的 exited 永不兑现，stop 会等满 5s；也不能 kill 假 pid。与上面几条一样留着。
  });
});

describe("buildArgs / --fit、MoE 专家下放、--no-context-shift（按 --help 探测）", () => {
  const { setCachedServerHelpSupport, clearServerHelpSupportCache, defaultLlamaServerBinary } =
    require("./llama-flash-attn") as typeof import("./llama-flash-attn");
  const { cpuMoeOverrideTensor } = require("./llama") as typeof import("./llama");

  /** 与 buildArgs 同一条同步路径规则（托管安装可能已被上面的 start 用例造出来）。 */
  const probe = (over: Partial<import("./llama-flash-attn").ServerHelpSupport> = {}) =>
    setCachedServerHelpSupport(defaultLlamaServerBinary(), {
      loadMode: "load-mode",
      flashAttn: "tristate",
      kvUnified: true,
      ...over,
    });

  const chatArgs = (rt: InstanceType<typeof LlamaRuntime>) =>
    rt.buildArgs({ kind: "local", path: chatModel, alias: "e2e-chat" }, DEFAULT_CUSTOM_SERVER_ARGS);

  const valueOf = (args: string[], flag: string): string | undefined => {
    const i = args.indexOf(flag);
    return i >= 0 ? args[i + 1] : undefined;
  };

  const moePlan = (fits: boolean) =>
    makePlan({
      gpuLayers: null,
      fits,
      moeOffload: { cpuMoeLayers: 3, moeLayers: 48, blockCount: 48, cpuExpertBytes: 1024 ** 3, fallbackGpuLayers: 30 },
      reasons: [{ code: "budget.vram" }, { code: "gpu.moe-cpu-offload" }],
    });

  beforeEach(() => {
    setChatSettings();
    SETTINGS.SERVER_AUTO_TUNE = "1";
  });
  afterEach(() => clearServerHelpSupportCache());

  test("没探过：计划装得下也不发 --fit / --n-gpu-layers（与加这些开关之前逐字节一致）", () => {
    clearServerHelpSupportCache();
    seedPlan(chatModel, makePlan({ gpuLayers: null, fits: true }));
    const args = chatArgs(new LlamaRuntime({ model: chatModel, port: "18700" }));
    expect(args).not.toContain("--fit");
    expect(args).not.toContain("--n-gpu-layers");
    expect(args).not.toContain("--no-context-shift");
  });

  test("认 --fit + 计划证明装得下 → -ngl 999 --fit off（别让 llama.cpp 为留余量再挪层）；复制的命令一致", () => {
    probe({ fit: true });
    seedPlan(chatModel, makePlan({ gpuLayers: null, fits: true }));
    const rt = new LlamaRuntime({ model: chatModel, port: "18701" });
    const args = chatArgs(rt);
    expect(valueOf(args, "--fit")).toBe("off");
    expect(valueOf(args, "--n-gpu-layers")).toBe("999");
    expect(rt.buildCommandLine(chatModel)).toContain("--n-gpu-layers 999 --fit off");
  });

  test("认 --fit + 证明不了（按层卸载）→ 只发 --fit on，不发层数（交给 llama.cpp 按真实显存放）", () => {
    probe({ fit: true });
    seedPlan(chatModel, makePlan({ gpuLayers: 20, fits: true }));
    const args = chatArgs(new LlamaRuntime({ model: chatModel, port: "18702" }));
    expect(valueOf(args, "--fit")).toBe("on");
    expect(args).not.toContain("--n-gpu-layers");
  });

  test("不认 --fit → 按层卸载回落计划层数（老行为）", () => {
    probe({ fit: false });
    seedPlan(chatModel, makePlan({ gpuLayers: 20, fits: true }));
    const args = chatArgs(new LlamaRuntime({ model: chatModel, port: "18703" }));
    expect(valueOf(args, "--n-gpu-layers")).toBe("20");
    expect(args).not.toContain("--fit");
  });

  test("MoE 专家下放：认 --n-cpu-moe → 层全留 GPU + --n-cpu-moe N（+ --fit off）", () => {
    probe({ fit: true, nCpuMoe: true, overrideTensor: true });
    seedPlan(chatModel, moePlan(true));
    const args = chatArgs(new LlamaRuntime({ model: chatModel, port: "18704" }));
    expect(valueOf(args, "--n-gpu-layers")).toBe("999");
    expect(valueOf(args, "--n-cpu-moe")).toBe("3");
    expect(valueOf(args, "--fit")).toBe("off");
    expect(args).not.toContain("--override-tensor");
  });

  test("MoE 专家下放：只认 -ot → 用正则把前 N 层专家张量放 CPU", () => {
    probe({ nCpuMoe: false, overrideTensor: true });
    seedPlan(chatModel, moePlan(true));
    const args = chatArgs(new LlamaRuntime({ model: chatModel, port: "18705" }));
    expect(valueOf(args, "--override-tensor")).toBe("blk\\.(0|1|2)\\.ffn_.*_exps\\.=CPU");
    expect(valueOf(args, "--override-tensor")).toBe(cpuMoeOverrideTensor(3));
    expect(args).not.toContain("--n-cpu-moe");
    expect(args).not.toContain("--fit");
  });

  test("MoE：两种下放开关都不认 → 回落按层卸载的层数", () => {
    probe({});
    seedPlan(chatModel, moePlan(true));
    const args = chatArgs(new LlamaRuntime({ model: chatModel, port: "18706" }));
    expect(valueOf(args, "--n-gpu-layers")).toBe("30");
    expect(args).not.toContain("--n-cpu-moe");
  });

  test("MoE 计划也装不下 + 认 --fit → 整组交给 --fit on", () => {
    probe({ fit: true, nCpuMoe: true });
    seedPlan(chatModel, moePlan(false));
    const args = chatArgs(new LlamaRuntime({ model: chatModel, port: "18707" }));
    expect(valueOf(args, "--fit")).toBe("on");
    expect(args).not.toContain("--n-cpu-moe");
    expect(args).not.toContain("--n-gpu-layers");
  });

  test("用户固定了层数 → 只听用户的：不下放专家、不发 --fit", () => {
    probe({ fit: true, nCpuMoe: true });
    SETTINGS.SERVER_GPU_LAYERS = "28";
    seedPlan(chatModel, moePlan(true));
    const args = chatArgs(new LlamaRuntime({ model: chatModel, port: "18708" }));
    expect(valueOf(args, "--n-gpu-layers")).toBe("28");
    expect(args).not.toContain("--n-cpu-moe");
    expect(args).not.toContain("--fit");
  });

  test("--no-context-shift：认才发，只给聊天实例（嵌入实例没有「对话」可丢）", () => {
    SETTINGS.SERVER_AUTO_TUNE = "0";
    probe({ noContextShift: false });
    expect(chatArgs(new LlamaRuntime({ model: chatModel, port: "18709" }))).not.toContain("--no-context-shift");
    probe({ noContextShift: true });
    expect(chatArgs(new LlamaRuntime({ model: chatModel, port: "18709" }))).toContain("--no-context-shift");
    const emb = new LlamaRuntime({ model: embedModel, port: "18710", purpose: "embedding" });
    expect(emb.buildCommandLine()).not.toContain("--no-context-shift");
  });
});

describe("start / 显存不足降级重试", () => {
  const originalFetch = globalThis.fetch;
  type LogEventInput = Parameters<typeof realAppLog.logEvent>[0];
  let logEvents: LogEventInput[] = [];
  /** 每次 spawn 依次取一条剧本：exit = null 表示活着（健康检查通过），数字表示打完日志就以此退出。 */
  let script: { log: string; exit: number | null }[] = [];
  let spawned: string[][] = [];
  let alive = false;
  let realSpawn: typeof realProc.spawnServerProcess;
  let realProbe: typeof realFlashAttn.probeServerHelp;

  const OOM_LOG =
    "llama_model_load: loading model\n" +
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216.00 MiB on device 0: cudaMalloc failed: out of memory\n" +
    "llama_init_from_model: failed to initialize the context\n" +
    "srv    load_model: failed to load model\n";

  beforeAll(async () => {
    const binPath = llamaCppBinaryPath();
    mkdirSync(dirname(binPath), { recursive: true });
    if (!existsSync(binPath)) writeFileSync(binPath, "#!/bin/sh\n");

    await mockModulePartial<typeof import("../app-log")>("./app-log", {
      logEvent: (input: LogEventInput) => {
        logEvents.push(input);
        return { ...input, level: input.level ?? "info", seq: logEvents.length, ts: Date.now(), pid: 1 };
      },
    });

    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/health")) {
        if (!alive) throw new Error("connection refused (fake)");
        return new Response(null, { status: 200 });
      }
      if (url.endsWith("/props")) return new Response(JSON.stringify({}), { status: 200 });
      throw new Error(`unexpected fetch in test: ${url}`);
    }) as typeof fetch;

    realSpawn = realProc.spawnServerProcess;
    mock.module("./proc", () => ({
      ...realProc,
      spawnServerProcess: (cmd: string[]) => {
        spawned.push(cmd);
        const step = script.shift() ?? { log: "", exit: null };
        alive = step.exit === null;
        const stream = (text: string) =>
          new ReadableStream<Uint8Array>({
            start(controller) {
              if (text) controller.enqueue(new TextEncoder().encode(text));
              controller.close();
            },
          });
        return {
          // pid 0：killProcessTree 不会去碰真实进程组
          pid: 0,
          exited:
            step.exit === null
              ? new Promise<number>(() => {})
              : new Promise<number>((resolve) => setTimeout(() => resolve(step.exit as number), 5)),
          stdout: stream(step.log),
          stderr: stream(""),
          kill: () => {},
        } as never;
      },
    }));

    realProbe = realFlashAttn.probeServerHelp;
    mock.module("./llama-flash-attn", () => ({
      ...realFlashAttn,
      probeServerHelp: async () => ({
        loadMode: "unknown",
        flashAttn: "none" as const,
        fit: true,
        nCpuMoe: true,
        overrideTensor: true,
        noContextShift: true,
      }),
    }));
  });

  beforeEach(() => {
    logEvents = [];
    script = [];
    spawned = [];
    alive = false;
    setChatSettings();
    SETTINGS.SERVER_AUTO_TUNE = "0";
    SETTINGS.SERVER_CTX_SIZE = "32768";
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
    mock.module("./proc", () => ({ ...realProc, spawnServerProcess: realSpawn }));
    mock.module("./llama-flash-attn", () => ({ ...realFlashAttn, probeServerHelp: realProbe }));
    mock.module("../app-log", () => ({ ...realAppLog }));
    realFlashAttn.clearServerHelpSupportCache();
  });

  /** 测试用的快节奏实例（轮询 / 退出等待 / 重试间隔都调小）。 */
  function fastRuntime(port: string, retryDelayMs = 0) {
    const rt = new LlamaRuntime({ model: chatModel, port });
    Object.assign(rt as unknown as Record<string, number>, { healthPollMs: 10, exitGraceMs: 20, retryDelayMs });
    return rt;
  }

  /** spawn 的 argv 里 llama-server 之后那一段（darwin 下前面有 script -q /dev/null 包装）。 */
  const argsOf = (cmd: string[]) => cmd.slice(cmd.indexOf(llamaCppBinaryPath()) + 1);
  const valueOf = (args: string[], flag: string) => args[args.indexOf(flag) + 1];
  const events = (name: string) => logEvents.filter((e) => e.event === name);

  test("OOM 两次后起来：上下文减半 → KV 降档，每次都进日志；复制的命令 = 最后一次实际 argv", async () => {
    script = [
      { log: OOM_LOG, exit: 1 },
      { log: OOM_LOG, exit: 1 },
      { log: "main: server is listening\n", exit: null },
    ];
    const statuses: string[] = [];
    const rt = fastRuntime("18720");
    rt.onStatusChange((s) => statuses.push(s));
    const result = await rt.start();
    expect(result.ok).toBe(true);
    expect(rt.getStatus()).toBe("running");
    expect(spawned.length).toBe(3);

    const [a1, a2, a3] = spawned.map(argsOf);
    expect(valueOf(a1!, "--ctx-size")).toBe("32768");
    expect(valueOf(a2!, "--ctx-size")).toBe("16384");
    expect(valueOf(a3!, "--ctx-size")).toBe("16384");
    expect(valueOf(a2!, "--cache-type-k")).toBe("q8_0");
    expect(valueOf(a3!, "--cache-type-k")).toBe("q4_0");
    expect(valueOf(a3!, "--cache-type-v")).toBe("q4_0");
    // 聊天实例 + 认 --no-context-shift → 发
    expect(a1).toContain("--no-context-shift");

    const logs = rt.getLogs();
    expect(logs).toContain("[omni] 显存不足，第 1 次降级重试：上下文 32768 → 16384");
    expect(logs).toContain("[omni] 显存不足，第 2 次降级重试：KV 缓存 q8_0/q8_0 → q4_0/q4_0");
    expect(events("launch.degrade.retry").length).toBe(2);
    expect(events("launch.degrade.succeeded").length).toBe(1);
    // 重试之间不闪「错误」
    expect(statuses).not.toContain("error");

    const degraded = rt.getDegradedLaunch();
    expect(degraded?.attempts).toBe(2);
    expect(degraded?.adjust).toEqual({ ctxCap: 16384, cacheTypeK: "q4_0", cacheTypeV: "q4_0" });
    // 「复制的命令」与实际发出去的 argv 是同一份（降级调整也叠进去了）；也不需要重启
    expect(rt.buildArgs({ kind: "local", path: chatModel, alias: "e2e-chat" }, DEFAULT_CUSTOM_SERVER_ARGS, chatModel)).toEqual(a3!);
    expect(rt.needsRestart()).toBe(false);
    // 设置没被改
    expect(SETTINGS.SERVER_CTX_SIZE).toBe("32768");
    expect(SETTINGS.SERVER_CACHE_TYPE_K).toBe("q8_0");
  });

  test("一直 OOM：最多重试 3 次（第 3 次交给 --fit on），然后按原错误报失败", async () => {
    script = [0, 1, 2, 3, 4].map(() => ({ log: OOM_LOG, exit: 1 }));
    const rt = fastRuntime("18721");
    const result = await rt.start();
    expect(result.ok).toBe(false);
    expect(result.error).toContain("out of memory");
    expect(spawned.length).toBe(4);
    expect(argsOf(spawned[3]!)).toContain("--fit");
    expect(valueOf(argsOf(spawned[3]!), "--fit")).toBe("on");
    expect(rt.getStatus()).toBe("error");
    expect(rt.getLastError()).toContain("out of memory");
    expect(events("launch.degrade.exhausted").length).toBe(1);
    expect(rt.getDegradedLaunch()).toBeNull();
  });

  test("不是显存问题（架构不认识）→ 不重试", async () => {
    script = [{ log: "llama_model_load: error loading model: unknown model architecture: 'foo'\n", exit: 1 }];
    const rt = fastRuntime("18722");
    const result = await rt.start();
    expect(result.ok).toBe(false);
    expect(result.error).toContain("unknown model architecture");
    expect(spawned.length).toBe(1);
    expect(events("launch.degrade.retry").length).toBe(0);
    expect(rt.getStatus()).toBe("error");
  });

  test("按模型固定了上下文：照样临时调小，日志说清楚不改设置", async () => {
    const { setModelParams, clearModelParams } =
      require("../db/model-params") as typeof import("../db/model-params");
    setModelParams(chatModel, { ctxSize: 16384 });
    try {
      script = [
        { log: OOM_LOG, exit: 1 },
        { log: "", exit: null },
      ];
      const rt = fastRuntime("18723");
      expect((await rt.start()).ok).toBe(true);
      expect(valueOf(argsOf(spawned[1]!), "--ctx-size")).toBe("8192");
      expect(rt.getLogs()).toContain("固定了上下文长度");
    } finally {
      clearModelParams(chatModel);
    }
  });

  test("重试等待中用户点了停止 → 不再起下一次，状态是已停止而不是错误", async () => {
    script = [
      { log: OOM_LOG, exit: 1 },
      { log: "", exit: null },
    ];
    const rt = fastRuntime("18724", 300);
    const pending = rt.start();
    // 等到第一次失败、进入重试等待
    for (let i = 0; i < 100 && !rt.getLogs().includes("第 1 次降级重试"); i++) await Bun.sleep(10);
    expect(rt.getLogs()).toContain("第 1 次降级重试");
    await rt.stop();
    const result = await pending;
    expect(result.ok).toBe(false);
    expect(spawned.length).toBe(1);
    expect(rt.getStatus()).toBe("stopped");
    expect(rt.getLogs()).toContain("放弃降级重试");
    expect(rt.getDegradedLaunch()).toBeNull();
  });
});
