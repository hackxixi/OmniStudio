import { afterEach, describe, expect, test } from "bun:test";

import { clearModelParams, setModelParams } from "../db/model-params";
import { updateSettings } from "../db/settings";
import { clearEngineHelpCache, setMlxHelpSupport, setVllmHelpSupport } from "./engine-sampling";
import { MlxRuntime } from "./mlx";
import { SglangRuntime } from "./sglang";
import { splitShellArgs } from "./shell-args";
import { VllmRuntime } from "./vllm";

/**
 * vLLM / SGLang / MLX 的按模型参数：真设置库（test-preload 临时目录），只测命令行拼装。
 * 模型用 HF repo id（不存在的本地路径），key 原样。
 */

const MODEL = "Qwen/Qwen3-8B";

/** argv 里 flag 的最后一次取值（同名参数后者生效）。 */
function valueOf(argv: string[], flag: string): string | undefined {
  const i = argv.lastIndexOf(flag);
  return i === -1 ? undefined : argv[i + 1];
}

const argvOf = (cmd: string) => splitShellArgs(cmd);

afterEach(() => {
  clearModelParams(MODEL);
  clearEngineHelpCache();
  updateSettings({ VLLM_EXTRA_ARGS: "", SGLANG_EXTRA_ARGS: "", MLX_EXTRA_ARGS: "" });
});

describe("vLLM", () => {
  test("按模型 ctx → --max-model-len；追加参数按引号切分、按模型在全局之后", () => {
    updateSettings({ VLLM_EXTRA_ARGS: `--seed 1 --chat-template-content-format "string"` });
    setModelParams(MODEL, { ctxSize: 32768, extraArgs: `--seed 42 --override-pooler-config '{"a": 1}'` });
    const argv = argvOf(new VllmRuntime({ model: MODEL, port: "18500" }).buildCommandLine());
    expect(valueOf(argv, "--max-model-len")).toBe("32768");
    expect(valueOf(argv, "--seed")).toBe("42");
    expect(valueOf(argv, "--chat-template-content-format")).toBe("string");
    expect(valueOf(argv, "--override-pooler-config")).toBe('{"a": 1}');
  });

  test("没探过 --help：不发采样参数；探到支持：发 --override-generation-config（按模型的值）+ 关思考", () => {
    setModelParams(MODEL, { thinking: "off", sampling: { temperature: 0.3, topK: 0 } });
    const rt = new VllmRuntime({ model: MODEL, port: "18501" });
    expect(argvOf(rt.buildCommandLine())).not.toContain("--override-generation-config");

    // start() 探测后记下的 key（这里直接注入，不跑真 vllm --help）
    (rt as unknown as { helpKey: string }).helpKey = "cli:/x/vllm";
    setVllmHelpSupport("cli:/x/vllm", { overrideGenerationConfig: true, defaultChatTemplateKwargs: true });
    const argv = argvOf(rt.buildCommandLine());
    const cfg = JSON.parse(valueOf(argv, "--override-generation-config")!) as Record<string, number>;
    expect(cfg.temperature).toBe(0.3);
    expect(cfg.top_k).toBe(-1);
    expect(Object.keys(cfg).sort()).toEqual(
      ["min_p", "presence_penalty", "repetition_penalty", "temperature", "top_k", "top_p"],
    );
    expect(valueOf(argv, "--default-chat-template-kwargs")).toBe('{"enable_thinking":false}');
  });

  test("没有按模型参数：--max-model-len 走全局设置", () => {
    const argv = argvOf(new VllmRuntime({ model: MODEL, port: "18502" }).buildCommandLine());
    expect(valueOf(argv, "--max-model-len")).toBe("8192");
  });
});

describe("SGLang", () => {
  test("按模型 ctx → --context-length；追加参数；不发任何采样参数", () => {
    setModelParams(MODEL, { ctxSize: 65536, extraArgs: "--enable-metrics", sampling: { temperature: 0.1 } });
    const argv = argvOf(new SglangRuntime({ model: MODEL, port: "18503" }).buildCommandLine());
    expect(valueOf(argv, "--context-length")).toBe("65536");
    expect(argv[argv.length - 1]).toBe("--enable-metrics");
    expect(argv.join(" ")).not.toMatch(/--temp|generation-config/);
  });
});

describe("MLX", () => {
  test("探到支持：按模型采样 → --temp/--top-p/--top-k/--min-p，关思考 → --chat-template-args；追加参数在最后", () => {
    setModelParams(MODEL, { thinking: "off", sampling: { temperature: 0.6, minP: 0.02 }, extraArgs: "--max-tokens 4096" });
    const rt = new MlxRuntime({ model: MODEL, port: "18504" });
    // 没探过：一个采样参数都不发
    expect(argvOf(rt.buildCommandLine())).not.toContain("--temp");

    const inner = rt as unknown as { binary: string; binaryMode: string };
    inner.binary = "/x/mlx_lm.server";
    inner.binaryMode = "server";
    setMlxHelpSupport("server:/x/mlx_lm.server", {
      temp: true,
      topP: true,
      topK: true,
      minP: true,
      chatTemplateArgs: true,
    });
    const argv = argvOf(rt.buildCommandLine());
    expect(valueOf(argv, "--temp")).toBe("0.6");
    expect(valueOf(argv, "--min-p")).toBe("0.02");
    expect(argv).toContain("--top-p");
    expect(argv).toContain("--top-k");
    expect(valueOf(argv, "--chat-template-args")).toBe('{"enable_thinking":false}');
    expect(argv.slice(-2)).toEqual(["--max-tokens", "4096"]);
  });

  test("needsRestart：没在跑 → false", () => {
    expect(new MlxRuntime({ model: MODEL }).needsRestart()).toBe(false);
    expect(new VllmRuntime({ model: MODEL }).needsRestart()).toBe(false);
    expect(new SglangRuntime({ model: MODEL }).needsRestart()).toBe(false);
  });
});
