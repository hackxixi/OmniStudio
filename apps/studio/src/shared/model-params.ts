/**
 * 按模型保存的参数（覆盖全局设置）与采样参数的来源标注。
 *
 * 模型身份 = 启动时的 target 字符串（`StartParams.model`：本地文件 / 仓库目录 / HF repo id），
 * 与已服务模型注册表同一口径。所有字段可选：缺省 = 不覆盖，落回下一级。
 *
 * 优先级（每个字段独立取）：
 *   启动参数：按模型设置 > 自动规划（SERVER_AUTO_TUNE）> 全局设置 > 档案默认
 *   采样参数：按模型设置 > 模型自带（generation_config.json / GGUF general.sampling.*）
 *             > 内置模型家族推荐表 > 全局设置
 */

export type SamplingParams = {
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  presencePenalty?: number;
  repeatPenalty?: number;
};

export const SAMPLING_FIELDS = [
  "temperature",
  "topP",
  "topK",
  "minP",
  "presencePenalty",
  "repeatPenalty",
] as const satisfies readonly (keyof SamplingParams)[];

export type SamplingField = (typeof SAMPLING_FIELDS)[number];

/** 思考模式：auto = 交给模型模板默认；on / off = 强制。 */
export type ThinkingMode = "auto" | "on" | "off";

export type ModelParams = {
  /** 固定上下文长度；自动规划开着时也作为 ctxOverride 交给规划器（其余参数照样自动）。 */
  ctxSize?: number;
  parallel?: number;
  /** -1 = 交给引擎 / 自动规划；>=0 = 固定层数。 */
  gpuLayers?: number;
  cacheTypeK?: string;
  cacheTypeV?: string;
  flashAttn?: "auto" | "on" | "off";
  thinking?: ThinkingMode;
  sampling?: SamplingParams;
  /** 追加给引擎的原始参数（按 shell 规则切分），放在最后、可覆盖前面的同名参数。 */
  extraArgs?: string;
};

/** 采样值的来源，界面上逐项标出来（「为什么是 0.6」）。 */
export type SamplingSource = "model-override" | "model-metadata" | "family-preset" | "global";

export type ResolvedSampling = {
  values: Required<SamplingParams>;
  sources: Record<SamplingField, SamplingSource>;
  /** 命中的家族预设 id（未命中 = null），界面显示「按 Qwen3 推荐」。 */
  preset: string | null;
};
