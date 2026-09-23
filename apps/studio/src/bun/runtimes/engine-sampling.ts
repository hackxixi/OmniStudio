/**
 * MLX / vLLM 的「服务端默认采样 + 思考开关」参数（llama.cpp 那套在 llama.ts / llama-flash-attn.ts）。
 *
 * 为什么要进启动参数：请求里没带采样值时（OpenAI 兼容客户端、网关转发的第三方请求），引擎用
 * 自己的默认 —— mlx_lm.server 默认 temp 0（贪心），vLLM 默认读 generation_config 或 1.0，
 * 与界面上显示的「这个模型按 0.6 采样」对不上。所以把解析好的值作为服务端默认交给引擎。
 *
 * 开关都按 `--help` 探测后才发（版本差异大：mlx-lm 较老的版本没有 --min-p / --chat-template-args，
 * vLLM 老版没有 --override-generation-config）；没探过 / 探测失败一律不发，行为与加这套之前一致。
 * SGLang 没有服务端级的数值采样开关（只有 --sampling-defaults model|openai），不在这里处理。
 */
import { resolveSampling } from "../model-sampling";
import type { ModelParams, ResolvedSampling, ThinkingMode } from "../../shared/model-params";

type SamplingValues = ResolvedSampling["values"];

/**
 * 解析某个模型的最终采样值（按模型 > 模型自带 > 家族推荐 > 全局）。解析器抛错 → null
 * （这时一个采样参数都不发，引擎用自己的默认 —— 与加这套之前一致，不能因此挡住启动）。
 */
export function resolveSamplingOrNull(target: string, mp: ModelParams | null): SamplingValues | null {
  try {
    return resolveSampling(target, { override: mp?.sampling, thinking: mp?.thinking }).values;
  } catch {
    return null;
  }
}

// —— MLX（mlx_lm.server） ——

export type MlxHelpSupport = {
  temp: boolean;
  topP: boolean;
  topK: boolean;
  minP: boolean;
  /** `--chat-template-args '<json>'`（传给 chat template 的 kwargs，关思考用）。 */
  chatTemplateArgs: boolean;
};

/** 开关名后面不能再跟 `-` / 单词字符（`--temp` 不能被 `--temperature-xxx` 误认）。 */
function hasFlag(help: string, flag: string): boolean {
  const escaped = flag.replace(/[-]/g, "\\-");
  return new RegExp(`(^|[\\s,\\[])${escaped}(?![-\\w])`, "m").test(help);
}

export function parseMlxHelp(help: string): MlxHelpSupport {
  return {
    temp: hasFlag(help, "--temp"),
    topP: hasFlag(help, "--top-p"),
    topK: hasFlag(help, "--top-k"),
    minP: hasFlag(help, "--min-p"),
    chatTemplateArgs: hasFlag(help, "--chat-template-args"),
  };
}

/**
 * MLX 的采样 / 思考参数。mlx_lm.server 没有 repetition / presence penalty 的服务端默认，那两项不发。
 * 思考只有「关」能表达（`enable_thinking: false`），开 / auto 交给模板默认。
 */
export function mlxSamplingArgs(
  values: SamplingValues | null,
  thinking: ThinkingMode | undefined,
  support: MlxHelpSupport | null,
): string[] {
  if (support === null) return [];
  const args: string[] = [];
  if (values !== null) {
    const push = (ok: boolean, flag: string, v: number) => {
      if (ok && Number.isFinite(v)) args.push(flag, String(v));
    };
    push(support.temp, "--temp", values.temperature);
    push(support.topP, "--top-p", values.topP);
    push(support.topK, "--top-k", values.topK);
    push(support.minP, "--min-p", values.minP);
  }
  if (thinking === "off" && support.chatTemplateArgs) {
    args.push("--chat-template-args", JSON.stringify({ enable_thinking: false }));
  }
  return args;
}

// —— vLLM ——

export type VllmHelpSupport = {
  /** `--override-generation-config '<json>'`（服务端默认采样，盖过模型的 generation_config）。 */
  overrideGenerationConfig: boolean;
  /** `--default-chat-template-kwargs '<json>'`（服务端默认模板 kwargs，关思考用）。 */
  defaultChatTemplateKwargs: boolean;
};

export function parseVllmHelp(help: string): VllmHelpSupport {
  return {
    overrideGenerationConfig: hasFlag(help, "--override-generation-config"),
    defaultChatTemplateKwargs: hasFlag(help, "--default-chat-template-kwargs"),
  };
}

/**
 * vLLM 的采样 / 思考参数：采样整组塞进一个 JSON（键名是 vLLM 的 SamplingParams 字段）。
 * top_k：llama.cpp 口径 0 = 不限，vLLM 老版只认 -1 表示不限（0 会报错），这里换算成 -1。
 */
export function vllmSamplingArgs(
  values: SamplingValues | null,
  thinking: ThinkingMode | undefined,
  support: VllmHelpSupport | null,
): string[] {
  if (support === null) return [];
  const args: string[] = [];
  if (values !== null && support.overrideGenerationConfig) {
    const cfg: Record<string, number> = {};
    const set = (key: string, v: number) => {
      if (Number.isFinite(v)) cfg[key] = v;
    };
    set("temperature", values.temperature);
    set("top_p", values.topP);
    set("top_k", values.topK <= 0 ? -1 : values.topK);
    set("min_p", values.minP);
    set("repetition_penalty", values.repeatPenalty);
    set("presence_penalty", values.presencePenalty);
    if (Object.keys(cfg).length > 0) args.push("--override-generation-config", JSON.stringify(cfg));
  }
  if (thinking === "off" && support.defaultChatTemplateKwargs) {
    args.push("--default-chat-template-kwargs", JSON.stringify({ enable_thinking: false }));
  }
  return args;
}

// —— 进程级探测缓存（按「启动命令前缀」分：托管 venv 与系统 python 版本可能不同） ——

const mlxCache = new Map<string, MlxHelpSupport>();
const vllmCache = new Map<string, VllmHelpSupport>();

export function cachedMlxHelpSupport(key: string | null): MlxHelpSupport | null {
  return key === null ? null : (mlxCache.get(key) ?? null);
}
export function setMlxHelpSupport(key: string, support: MlxHelpSupport): void {
  mlxCache.set(key, support);
}
export function cachedVllmHelpSupport(key: string | null): VllmHelpSupport | null {
  return key === null ? null : (vllmCache.get(key) ?? null);
}
export function setVllmHelpSupport(key: string, support: VllmHelpSupport): void {
  vllmCache.set(key, support);
}
export function clearEngineHelpCache(): void {
  mlxCache.clear();
  vllmCache.clear();
}
