/**
 * llama.cpp 启动时显存不足（OOM）的**降级重试**：纯函数部分（选下一步、拼说明）。
 *
 * 为什么要有：规划器的估算再准也是估算 —— 别的进程临时占了显存、驱动碎片、compute buffer
 * 比公式大一截，都会让「算着装得下」的启动在分配时失败。以前失败只会整条换备选模型
 * （SERVER_FALLBACK_MODELS），而用户要的其实是「同一个模型，窗口小一点也行」。
 *
 * 做法：同一次 start() 里最多重试 3 次，每次在上一次的基础上再退一步，顺序是
 *   ① 上下文减半（不低于下限 4096 / SERVER_AUTO_TUNE_MIN_CTX）—— KV 与窗口线性相关，
 *      减半最直接，而且对「能不能用」影响最小；
 *   ② KV 缓存量化降一档（f16 → q8_0 → q4_0）—— 再省一半左右的 KV；
 *   ③ GPU 这一侧让步：能交给 `--fit on` 就交给 llama.cpp 按真实空闲显存自己放（比我们估得准）；
 *      MoE 专家下放模式下多放几层专家到 CPU；否则层数砍到 75%。
 * 三步都用过了还有次数，就从头再挑一个还能退的（通常是上下文再减半）。
 *
 * 调整是**运行时临时的**（LlamaRuntime.launchAdjust）：不写设置、不改按模型参数；下次启动
 * 从计划重新开始（显存可能已经空出来了）。buildArgs 最后一层叠上它，所以「复制的命令」
 * 与实际发出去的 argv 仍然是同一份。
 */

import type { DegradedLaunch, LaunchAdjust } from "../../shared/launch-planner";

export type { DegradedLaunch, LaunchAdjust };

/** 最多重试几次（不含第一次启动）。 */
export const MAX_DEGRADE_RETRIES = 3;

export type DegradeStepKind = "ctx" | "kv" | "gpu";

/** 本次启动「显存相关旋钮」的实际取值（已经叠过当前 LaunchAdjust），选下一步用。 */
export type MemoryKnobs = {
  ctx: number;
  /** 用户按模型固定了上下文（仍允许临时调小，但日志要说清楚） */
  ctxPinned: boolean;
  cacheTypeK: string | null;
  cacheTypeV: string | null;
  /** 实际发出去的 --n-gpu-layers（null = 没发，引擎 / --fit 决定） */
  gpuLayers: number | null;
  /** 用户固定了 GPU 层数（全局或按模型） */
  gpuLayersPinned: boolean;
  /** 实际发出去的专家下放层数（null = 没有） */
  nCpuMoe: number | null;
  fit: "on" | "off" | null;
  /** 模型层数（未知 = null） */
  blockCount: number | null;
  /** 这台 llama-server 认不认 --fit */
  fitSupported: boolean;
  /** flash attention 确定是关的：llama.cpp 不接受量化 V cache，只能降 K */
  flashAttnOff: boolean;
  /** 上下文下限 */
  minCtx: number;
};

export type DegradeStep = {
  kind: DegradeStepKind;
  /** 叠加后的完整调整（直接替换 runtime 上的那份） */
  adjust: LaunchAdjust;
  /** 这一步的中文说明（进服务器日志 / app.log） */
  summary: string;
};

/** KV 量化降一档：f16 / bf16 / f32 / 未设（引擎默认 f16）→ q8_0 → q4_0；已经 4bit 就不再降。 */
export function lowerKvType(type: string | null): string | null {
  const t = (type ?? "f16").toLowerCase();
  if (t === "f16" || t === "bf16" || t === "f32") return "q8_0";
  if (t === "q8_0" || t === "q5_0" || t === "q5_1" || t === "q4_1" || t === "iq4_nl") return "q4_0";
  return null;
}

/** 对齐到 256（llama.cpp KV cell 的粒度），不低于 floor。 */
function halveCtx(ctx: number, floor: number): number {
  const half = Math.floor(ctx / 2 / 256) * 256;
  return Math.max(floor, half);
}

type Candidate = { kind: DegradeStepKind; adjust: LaunchAdjust; summary: string };

function ctxCandidate(k: MemoryKnobs, prev: LaunchAdjust): Candidate | null {
  const floor = Math.max(1, k.minCtx);
  if (k.ctx <= floor) return null;
  const next = halveCtx(k.ctx, floor);
  if (next >= k.ctx) return null;
  return {
    kind: "ctx",
    adjust: { ...prev, ctxCap: next },
    summary:
      `上下文 ${k.ctx} → ${next}` +
      (k.ctxPinned ? "（该模型固定了上下文长度，本次临时调小，不改你的设置）" : ""),
  };
}

function kvCandidate(k: MemoryKnobs, prev: LaunchAdjust): Candidate | null {
  const nextK = lowerKvType(k.cacheTypeK);
  // FA 关着时 llama.cpp 拒绝量化 V（启动直接报错，而且不是显存错，重试链会就此中断）
  const nextV = k.flashAttnOff ? null : lowerKvType(k.cacheTypeV);
  if (nextK === null && nextV === null) return null;
  const adjust: LaunchAdjust = { ...prev };
  if (nextK !== null) adjust.cacheTypeK = nextK;
  if (nextV !== null) adjust.cacheTypeV = nextV;
  const from = `${k.cacheTypeK ?? "f16"}/${k.cacheTypeV ?? "f16"}`;
  const to = `${nextK ?? k.cacheTypeK ?? "f16"}/${nextV ?? k.cacheTypeV ?? "f16"}`;
  return { kind: "kv", adjust, summary: `KV 缓存 ${from} → ${to}` };
}

function gpuCandidate(k: MemoryKnobs, prev: LaunchAdjust): Candidate | null {
  const pinnedNote = k.gpuLayersPinned ? "（GPU 层数是你固定的，本次临时调小，不改你的设置）" : "";
  // MoE 专家下放模式：多放几层专家（每次加 1/4 层数），层与 KV 仍留在 GPU
  if (k.nCpuMoe !== null && k.blockCount !== null && k.nCpuMoe < k.blockCount && k.fit !== "on") {
    const next = Math.min(k.blockCount, k.nCpuMoe + Math.max(1, Math.ceil(k.blockCount / 4)));
    return {
      kind: "gpu",
      adjust: { ...prev, nCpuMoe: next },
      summary: `MoE 专家放 CPU 的层数 ${k.nCpuMoe} → ${next}`,
    };
  }
  // 交给 llama.cpp 按真实空闲显存自己拟合（它看得到我们看不到的碎片与别的进程）
  if (k.fitSupported && k.fit !== "on" && !k.gpuLayersPinned) {
    const adjust: LaunchAdjust = { ...prev, fit: "on" };
    delete adjust.gpuLayers;
    delete adjust.nCpuMoe;
    return { kind: "gpu", adjust, summary: "GPU 层数交给 llama.cpp 按空闲显存自动拟合（--fit on）" };
  }
  // 按层砍到 75%：当前发的层数，没发就按「全部层」算（需要知道层数）
  const current = k.gpuLayers !== null && k.gpuLayers < 999 ? k.gpuLayers : k.blockCount;
  if (current === null || current <= 0) return null;
  const next = Math.floor(current * 0.75);
  if (next >= current) return null;
  return {
    kind: "gpu",
    adjust: { ...prev, gpuLayers: next },
    summary: `GPU 层数 ${current} → ${next}${pinnedNote}`,
  };
}

const ORDER: DegradeStepKind[] = ["ctx", "kv", "gpu"];

/**
 * 选下一步降级：按 ctx → kv → gpu 的顺序挑第一个**还没用过**且还能退的；都用过了就
 * 按同样顺序挑第一个还能退的（上下文可以再减半、层数可以再砍）。一步都退不了 → null
 * （调用方就此放弃，按原错误报失败）。
 */
export function nextDegradeStep(
  knobs: MemoryKnobs,
  prev: LaunchAdjust,
  used: ReadonlySet<DegradeStepKind>,
): DegradeStep | null {
  const build = (kind: DegradeStepKind): Candidate | null =>
    kind === "ctx" ? ctxCandidate(knobs, prev) : kind === "kv" ? kvCandidate(knobs, prev) : gpuCandidate(knobs, prev);
  for (const kind of ORDER) {
    if (used.has(kind)) continue;
    const c = build(kind);
    if (c !== null) return c;
  }
  for (const kind of ORDER) {
    const c = build(kind);
    if (c !== null) return c;
  }
  return null;
}
