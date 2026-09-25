import { describe, expect, test } from "bun:test";

import { mlxSamplingArgs, parseMlxHelp, parseVllmHelp, vllmSamplingArgs } from "./engine-sampling";

const VALUES = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0,
  presencePenalty: 1.5,
  repeatPenalty: 1.05,
};

describe("MLX（mlx_lm.server）", () => {
  const NEW_HELP = [
    "usage: mlx_lm.server [-h] [--model MODEL] ...",
    "  --temp TEMP           Default sampling temperature (default: 0.0)",
    "  --top-p TOP_P         Default nucleus sampling top-p (default: 1.0)",
    "  --top-k TOP_K         Default top-k sampling (default: 0, disables top-k)",
    "  --min-p MIN_P         Default min-p sampling (default: 0.0, disables min-p)",
    "  --chat-template-args CHAT_TEMPLATE_ARGS",
    "  --prompt-cache-bytes PROMPT_CACHE_BYTES",
  ].join("\n");

  test("新版 --help：四个采样开关 + chat-template-args 都认", () => {
    expect(parseMlxHelp(NEW_HELP)).toEqual({ temp: true, topP: true, topK: true, minP: true, chatTemplateArgs: true });
  });

  test("按支持情况发：采样值 + 关思考；没有 penalty 类开关", () => {
    const args = mlxSamplingArgs(VALUES, "off", parseMlxHelp(NEW_HELP));
    expect(args).toEqual([
      "--temp", "0.6",
      "--top-p", "0.95",
      "--top-k", "20",
      "--min-p", "0",
      "--chat-template-args", '{"enable_thinking":false}',
    ]);
  });

  test("老版只有 --temp / --top-p：只发这两个；不认的开关一律不发", () => {
    const old = parseMlxHelp("  --temp TEMP\n  --top-p TOP_P\n  --max-tokens MAX_TOKENS\n");
    expect(old.minP).toBe(false);
    expect(mlxSamplingArgs(VALUES, "off", old)).toEqual(["--temp", "0.6", "--top-p", "0.95"]);
  });

  test("没探过（support = null）/ 采样解析失败（values = null）：不发", () => {
    expect(mlxSamplingArgs(VALUES, "off", null)).toEqual([]);
    expect(mlxSamplingArgs(null, "on", parseMlxHelp(NEW_HELP))).toEqual([]);
  });

  test("--temp 不会被 --temperature-xxx 误认", () => {
    expect(parseMlxHelp("  --temperature-schedule X\n").temp).toBe(false);
  });
});

describe("vLLM", () => {
  const HELP = [
    "  --override-generation-config OVERRIDE_GENERATION_CONFIG",
    "                        Overrides or sets generation config. e.g. {\"temperature\": 0.5}",
    "  --default-chat-template-kwargs DEFAULT_CHAT_TEMPLATE_KWARGS",
  ].join("\n");

  test("--override-generation-config：整组 JSON，键名是 vLLM 的；top_k 0 → -1", () => {
    const args = vllmSamplingArgs({ ...VALUES, topK: 0 }, "off", parseVllmHelp(HELP));
    expect(args[0]).toBe("--override-generation-config");
    expect(JSON.parse(args[1]!)).toEqual({
      temperature: 0.6,
      top_p: 0.95,
      top_k: -1,
      min_p: 0,
      repetition_penalty: 1.05,
      presence_penalty: 1.5,
    });
    expect(args.slice(2)).toEqual(["--default-chat-template-kwargs", '{"enable_thinking":false}']);
  });

  test("老版没有这两个开关 / 没探过：不发", () => {
    const old = parseVllmHelp("  --max-model-len MAX_MODEL_LEN\n  --generation-config GENERATION_CONFIG\n");
    expect(old).toEqual({ overrideGenerationConfig: false, defaultChatTemplateKwargs: false });
    expect(vllmSamplingArgs(VALUES, "off", old)).toEqual([]);
    expect(vllmSamplingArgs(VALUES, "off", null)).toEqual([]);
  });

  test("思考 auto / on 不发模板 kwargs", () => {
    expect(vllmSamplingArgs(VALUES, "on", parseVllmHelp(HELP))).toHaveLength(2);
    expect(vllmSamplingArgs(VALUES, undefined, parseVllmHelp(HELP))).toHaveLength(2);
  });
});
