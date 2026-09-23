/**
 * 内置模型家族采样推荐表（主进程与界面共用，纯数据 + 纯函数）。
 *
 * 为什么要有：以前所有本地模型都吃同一套 OCR 口味的采样（temp 0.2 / repeat 1.12），
 * 推理模型在这组参数下会死循环、复读，或者思考链被压得很短。各家模型卡都给了推荐值，
 * 这里按名字把它们认出来（优先级见 shared/model-params.ts：按模型设置 > 模型自带 >
 * 本表 > 全局设置）。
 *
 * 匹配规则：模型名先归一化（小写，`_` / 空格 → `-`），每条预设有若干匹配式，
 * 一个匹配式是一组子串、**全部**出现才算命中，特异度 = 子串长度之和；
 * 全表取特异度最高的一条（最长匹配胜，"Qwen3-Coder-30B" 命中 qwen3-coder 而不是 qwen3），
 * 同分取表里靠前的。
 *
 * 变体：混合思考模型（Qwen3 等）两种模式的推荐值不一样，分别放 thinking / nonThinking；
 * `default` 是「不强制模式」时用的那一份（缺省 = thinking，因为这类模型模板默认会思考）。
 * 每个变体都是完整的一份，不和别的变体合并。某项没写 = 模型卡没说，落回下一级（全局设置）。
 *
 * 数值来源尽量取模型卡 / 官方 generation_config.json，其次 Unsloth 文档与 Unsloth Studio 的
 * inference_defaults.json（studio/backend/assets/configs/）。每条都注明出处。
 */
import type { SamplingParams, ThinkingMode } from "./model-params";

export type SamplingPreset = {
  /** 界面展示名（「按 Qwen3 推荐」）。 */
  label: string;
  /** 匹配式：字符串 = 单个子串；数组 = 这几个子串都要出现。全部小写、已按归一化规则写。 */
  patterns: (string | string[])[];
  default?: SamplingParams;
  thinking?: SamplingParams;
  nonThinking?: SamplingParams;
};

// 常用的几组值，避免同一组数字散在十几处
const QWEN3_THINK: SamplingParams = { temperature: 0.6, topP: 0.95, topK: 20, minP: 0, repeatPenalty: 1.0 };
const QWEN3_INSTRUCT: SamplingParams = { temperature: 0.7, topP: 0.8, topK: 20, minP: 0, repeatPenalty: 1.0 };
const GEMMA: SamplingParams = { temperature: 1.0, topP: 0.95, topK: 64, minP: 0, repeatPenalty: 1.0 };
// 老的 OCR 档案（shared/model-profiles.ts 的 DEFAULT_SERVER_ARGS）：保持原值，OCR 输出要稳定、少复读。
const OCR_LEGACY: SamplingParams = { temperature: 0.2, topP: 0.9, topK: 40, repeatPenalty: 1.12 };

export const SAMPLING_PRESETS: Readonly<Record<string, SamplingPreset>> = {
  // ---------- Qwen ----------
  // Qwen3.8：Unsloth 文档 / unsloth/Qwen3.8-27B 模型卡（thinking presence 0，instruct presence 1.5）
  "qwen3.8": {
    label: "Qwen3.8",
    patterns: ["qwen3.8"],
    thinking: { temperature: 1.0, topP: 0.95, topK: 20, minP: 0, presencePenalty: 0, repeatPenalty: 1.0 },
    nonThinking: { temperature: 0.7, topP: 0.8, topK: 20, minP: 0, presencePenalty: 1.5, repeatPenalty: 1.0 },
  },
  // Qwen3.6：unsloth.ai/docs/models/qwen3.6（thinking 通用档；精确编码档 temp 0.6 不单列）
  "qwen3.6": {
    label: "Qwen3.6",
    patterns: ["qwen3.6"],
    thinking: { temperature: 1.0, topP: 0.95, topK: 20, minP: 0, presencePenalty: 0, repeatPenalty: 1.0 },
    nonThinking: { temperature: 0.7, topP: 0.8, topK: 20, minP: 0, presencePenalty: 1.5, repeatPenalty: 1.0 },
  },
  // Qwen3.5：unsloth.ai/docs/models/qwen3.5（两种模式通用档都带 presence_penalty 1.5）
  "qwen3.5": {
    label: "Qwen3.5",
    patterns: ["qwen3.5"],
    thinking: { temperature: 1.0, topP: 0.95, topK: 20, minP: 0, presencePenalty: 1.5, repeatPenalty: 1.0 },
    nonThinking: { temperature: 0.7, topP: 0.8, topK: 20, minP: 0, presencePenalty: 1.5, repeatPenalty: 1.0 },
  },
  // Qwen3-Coder：Qwen/Qwen3-Coder-30B-A3B-Instruct 模型卡（repetition_penalty 1.05；没有思考模式）
  "qwen3-coder": {
    label: "Qwen3-Coder",
    patterns: ["qwen3-coder"],
    default: { temperature: 0.7, topP: 0.8, topK: 20, minP: 0, repeatPenalty: 1.05 },
  },
  // Qwen3-Next：Qwen3-Next-80B-A3B-Instruct / -Thinking 模型卡（Instruct 与 Thinking 是两个模型）
  "qwen3-next": {
    label: "Qwen3-Next",
    patterns: ["qwen3-next"],
    default: QWEN3_INSTRUCT,
  },
  "qwen3-next-thinking": {
    label: "Qwen3-Next Thinking",
    patterns: [["qwen3-next", "thinking"]],
    default: QWEN3_THINK,
  },
  // Qwen3-VL：Qwen3-VL-*-Instruct / -Thinking 模型卡（Instruct presence 1.5；Thinking temp 1.0）
  "qwen3-vl": {
    label: "Qwen3-VL",
    patterns: ["qwen3-vl"],
    default: { temperature: 0.7, topP: 0.8, topK: 20, minP: 0, presencePenalty: 1.5, repeatPenalty: 1.0 },
  },
  "qwen3-vl-thinking": {
    label: "Qwen3-VL Thinking",
    patterns: [["qwen3-vl", "thinking"]],
    default: { temperature: 1.0, topP: 0.95, topK: 20, minP: 0, presencePenalty: 0, repeatPenalty: 1.0 },
  },
  // Qwen3-2507：Instruct-2507 / Thinking-2507 是分开的单模式模型（各自模型卡）
  "qwen3-instruct-2507": {
    label: "Qwen3 Instruct-2507",
    patterns: [["qwen3", "instruct-2507"]],
    default: QWEN3_INSTRUCT,
  },
  "qwen3-thinking-2507": {
    label: "Qwen3 Thinking-2507",
    patterns: [["qwen3", "thinking-2507"]],
    default: QWEN3_THINK,
  },
  // Qwen3（混合思考）：Qwen/Qwen3-8B 等模型卡「Best Practices」
  qwen3: {
    label: "Qwen3",
    patterns: ["qwen3"],
    thinking: QWEN3_THINK,
    nonThinking: QWEN3_INSTRUCT,
  },
  // QwQ-32B 模型卡：Temperature 0.6, TopP 0.95, MinP 0, TopK 20~40（取 40，与 Unsloth 表一致）
  qwq: {
    label: "QwQ",
    patterns: ["qwq"],
    default: { temperature: 0.6, topP: 0.95, topK: 40, minP: 0, repeatPenalty: 1.0 },
  },
  // Qwen2 / 2.5（含 Coder / Math）：官方 generation_config.json（0.7 / 0.8 / 20 / 1.05）
  "qwen2.5": {
    label: "Qwen2.5",
    patterns: ["qwen2.5", "qwen2"],
    default: { temperature: 0.7, topP: 0.8, topK: 20, repeatPenalty: 1.05 },
  },

  // ---------- DeepSeek ----------
  // DeepSeek-R1 及蒸馏版模型卡：temperature 0.5~0.7（推荐 0.6）、top_p 0.95
  "deepseek-r1": {
    label: "DeepSeek-R1",
    patterns: ["deepseek-r1"],
    default: { temperature: 0.6, topP: 0.95, repeatPenalty: 1.0 },
  },
  // DeepSeek-V3.1 / V3.2：Unsloth 文档（temp 0.6, top_p 0.95）
  "deepseek-v3": {
    label: "DeepSeek-V3",
    patterns: ["deepseek-v3"],
    default: { temperature: 0.6, topP: 0.95, repeatPenalty: 1.0 },
  },
  // DeepSeek-V3-0324 模型卡：API temperature 1.0 映射到模型 0.3，本地直接用 0.3
  "deepseek-v3-0324": {
    label: "DeepSeek-V3-0324",
    patterns: ["deepseek-v3-0324"],
    default: { temperature: 0.3, topP: 0.95, repeatPenalty: 1.0 },
  },
  // DeepSeek-V4：Unsloth Studio inference_defaults.json（1.0 / 1.0）
  "deepseek-v4": {
    label: "DeepSeek-V4",
    patterns: ["deepseek-v4"],
    default: { temperature: 1.0, topP: 1.0, minP: 0, repeatPenalty: 1.0 },
  },

  // ---------- Gemma ----------
  // Gemma 2 / 3 / 3n / 4：Gemma 团队推荐（Unsloth Gemma 3 / Gemma 4 文档）：1.0 / 0.95 / 64 / min_p 0
  gemma: {
    label: "Gemma",
    patterns: ["gemma-2", "gemma2", "gemma-3", "gemma3", "gemma-4", "gemma4", "medgemma"],
    default: GEMMA,
  },

  // ---------- Llama ----------
  // Llama 3.x：Meta 官方 generation_config.json（temperature 0.6, top_p 0.9）
  "llama-3": {
    label: "Llama 3",
    patterns: ["llama-3", "llama3"],
    default: { temperature: 0.6, topP: 0.9, repeatPenalty: 1.0 },
  },
  // Llama 4：Unsloth Llama 4 文档（temp 0.6, min_p 0.01, top_p 0.9）
  "llama-4": {
    label: "Llama 4",
    patterns: ["llama-4", "llama4"],
    default: { temperature: 0.6, topP: 0.9, minP: 0.01, repeatPenalty: 1.0 },
  },

  // ---------- Mistral ----------
  // Mistral Small 3.x 模型卡：temperature 0.15
  "mistral-small": {
    label: "Mistral Small",
    patterns: ["mistral-small"],
    default: { temperature: 0.15, topP: 0.95, repeatPenalty: 1.0 },
  },
  // Mistral Nemo 模型卡：建议较低温度 0.3
  "mistral-nemo": {
    label: "Mistral Nemo",
    patterns: ["mistral-nemo"],
    default: { temperature: 0.3, repeatPenalty: 1.0 },
  },
  // Ministral：Unsloth Studio inference_defaults.json（0.15 / 0.95）
  ministral: {
    label: "Ministral",
    patterns: ["ministral"],
    default: { temperature: 0.15, topP: 0.95, repeatPenalty: 1.0 },
  },
  // Devstral 模型卡（vLLM 示例 temperature 0.15）
  devstral: {
    label: "Devstral",
    patterns: ["devstral"],
    default: { temperature: 0.15, repeatPenalty: 1.0 },
  },
  // Magistral 模型卡：top_p 0.95, temperature 0.7
  magistral: {
    label: "Magistral",
    patterns: ["magistral"],
    default: { temperature: 0.7, topP: 0.95, repeatPenalty: 1.0 },
  },

  // ---------- Phi ----------
  // Phi-4：Unsloth Studio inference_defaults.json（0.8 / 0.95）
  "phi-4": {
    label: "Phi-4",
    patterns: ["phi-4", "phi4"],
    default: { temperature: 0.8, topP: 0.95, repeatPenalty: 1.0 },
  },
  // Phi-4-reasoning / -mini-reasoning 模型卡：temperature 0.8, top_k 50, top_p 0.95
  "phi-4-reasoning": {
    label: "Phi-4 Reasoning",
    patterns: [["phi-4", "reasoning"], ["phi4", "reasoning"]],
    default: { temperature: 0.8, topP: 0.95, topK: 50, repeatPenalty: 1.0 },
  },

  // ---------- GLM / 智谱 ----------
  // GLM-4.5 / 4.6 / 4.7 / 5：Unsloth 文档（temperature 1.0, top_p 0.95）
  glm: {
    label: "GLM",
    patterns: ["glm-4", "glm4", "glm-5", "glm5"],
    default: { temperature: 1.0, topP: 0.95, repeatPenalty: 1.0 },
  },

  // ---------- 其他 ----------
  // gpt-oss：OpenAI 模型卡 / Unsloth 文档（temperature 1.0, top_p 1.0, top_k 0 = 不截断）；
  // min_p 置 0：官方只给了这三项，别让全局的 min_p 0.05 悄悄截尾。
  "gpt-oss": {
    label: "gpt-oss",
    patterns: ["gpt-oss"],
    default: { temperature: 1.0, topP: 1.0, topK: 0, minP: 0, repeatPenalty: 1.0 },
  },
  // Kimi K2 Instruct 模型卡：temperature 0.6
  kimi: {
    label: "Kimi",
    patterns: ["kimi"],
    default: { temperature: 0.6, repeatPenalty: 1.0 },
  },
  // Kimi K2-Thinking 模型卡：temperature 1.0
  "kimi-k2-thinking": {
    label: "Kimi K2 Thinking",
    patterns: [["kimi-k2", "thinking"]],
    default: { temperature: 1.0, repeatPenalty: 1.0 },
  },
  // Kimi K2.5 模型卡：Thinking 1.0 / Instant 0.6，top_p 0.95
  "kimi-k2.5": {
    label: "Kimi K2.5",
    patterns: ["kimi-k2.5"],
    thinking: { temperature: 1.0, topP: 0.95, repeatPenalty: 1.0 },
    nonThinking: { temperature: 0.6, topP: 0.95, repeatPenalty: 1.0 },
  },
  // SmolLM3 模型卡：temperature 0.6, top_p 0.95
  smollm3: {
    label: "SmolLM3",
    patterns: ["smollm3"],
    default: { temperature: 0.6, topP: 0.95, repeatPenalty: 1.0 },
  },
  // MiniMax-M2 系列模型卡：temperature 1.0, top_p 0.95, top_k 40
  minimax: {
    label: "MiniMax",
    patterns: ["minimax"],
    default: { temperature: 1.0, topP: 0.95, topK: 40, repeatPenalty: 1.0 },
  },
  // Granite 4.0：Unsloth Granite-4 文档（temperature 0.0, top_p 1.0, top_k 0，偏确定性输出）
  "granite-4": {
    label: "Granite 4",
    patterns: ["granite-4", "granite4"],
    default: { temperature: 0, topP: 1.0, topK: 0, repeatPenalty: 1.0 },
  },
  // Nemotron：Unsloth Studio inference_defaults.json / Nemotron 3 模型卡（推理档 1.0 / 1.0）
  nemotron: {
    label: "Nemotron",
    patterns: ["nemotron"],
    default: { temperature: 1.0, topP: 1.0, repeatPenalty: 1.0 },
  },
  // LFM2：Liquid 模型卡（temperature 0.3, min_p 0.15, repetition_penalty 1.05）
  lfm2: {
    label: "LFM2",
    patterns: ["lfm2"],
    default: { temperature: 0.3, minP: 0.15, repeatPenalty: 1.05 },
  },

  // ---------- OCR（原 model-profiles 档案，值不变） ----------
  lightonocr: { label: "LightOnOCR", patterns: ["lightonocr"], default: OCR_LEGACY },
  chandra: { label: "Chandra OCR", patterns: ["chandra"], default: OCR_LEGACY },
  glmocr: { label: "GLM-OCR", patterns: ["glm-ocr", "glmocr"], default: OCR_LEGACY },
  // DeepSeek-OCR：Unsloth Studio inference_defaults.json（temperature 0，OCR 要确定性）
  "deepseek-ocr": {
    label: "DeepSeek-OCR",
    patterns: ["deepseek-ocr"],
    default: { temperature: 0, topP: 0.95, repeatPenalty: 1.0 },
  },
};

/** 名字归一化：小写，`_` / 空白 → `-`（GGUF 文件名里 `Qwen3_8B`、`gemma 3` 都见过）。 */
export function normalizeModelName(name: string): string {
  return name.toLowerCase().replace(/[_\s]+/g, "-");
}

/** 模型名 → 命中的预设（最长匹配胜，同分取表中靠前者）；未命中返回 null。 */
export function matchSamplingPreset(modelName: string): { id: string; preset: SamplingPreset } | null {
  const name = normalizeModelName(modelName);
  if (!name) return null;
  let best: { id: string; preset: SamplingPreset; score: number } | null = null;
  for (const [id, preset] of Object.entries(SAMPLING_PRESETS)) {
    for (const pattern of preset.patterns) {
      const parts = typeof pattern === "string" ? [pattern] : pattern;
      if (!parts.every((p) => name.includes(p))) continue;
      const score = parts.reduce((n, p) => n + p.length, 0);
      if (best === null || score > best.score) best = { id, preset, score };
    }
  }
  return best === null ? null : { id: best.id, preset: best.preset };
}

/**
 * 名字认不出来时按 GGUF 的 general.architecture 兜底：微调模型常换名字（Qwopus3.5 是
 * Qwen3.5 的微调），架构名却照抄底座。只收「架构 = 单一家族」的，llama / deepseek2 这类
 * 被多个家族共用的架构不收 —— 猜错比不猜更糟。
 */
const ARCH_PRESETS: Readonly<Record<string, string>> = {
  qwen35: "qwen3.5",
  qwen35moe: "qwen3.5",
  qwen3next: "qwen3-next",
  qwen3vl: "qwen3-vl",
  qwen3vlmoe: "qwen3-vl",
  gemma3: "gemma",
  gemma3n: "gemma",
  gemma4: "gemma",
  "gpt-oss": "gpt-oss",
  gptoss: "gpt-oss",
  smollm3: "smollm3",
  lfm2: "lfm2",
};

export function matchSamplingPresetByArch(arch: string | null | undefined): { id: string; preset: SamplingPreset } | null {
  const id = arch ? ARCH_PRESETS[arch.toLowerCase()] : undefined;
  const preset = id ? SAMPLING_PRESETS[id] : undefined;
  return id && preset ? { id, preset } : null;
}

/**
 * 按思考模式挑变体：on / off 取对应变体（没有就用 default）；
 * auto 取 default，混合模型没写 default 时取 thinking（模板默认开思考）。
 */
export function presetVariant(preset: SamplingPreset, mode: ThinkingMode = "auto"): SamplingParams {
  if (mode === "on") return preset.thinking ?? preset.default ?? preset.nonThinking ?? {};
  if (mode === "off") return preset.nonThinking ?? preset.default ?? preset.thinking ?? {};
  return preset.default ?? preset.thinking ?? preset.nonThinking ?? {};
}

/** 这个预设有没有为该模式单列变体（有的话它比模型自带的默认值更贴切，见 model-sampling.ts）。 */
export function presetHasExplicitVariant(preset: SamplingPreset, mode: ThinkingMode): boolean {
  if (mode === "on") return preset.thinking !== undefined;
  if (mode === "off") return preset.nonThinking !== undefined;
  return false;
}
