import { afterEach, describe, expect, test } from "bun:test";

import {
  cachedFlashAttnSupport,
  cachedKvUnifiedSupport,
  cachedMemoryFlagSupport,
  cachedServerHelpSupport,
  clearServerHelpSupportCache,
  flashAttnArgs,
  cachedReasoningSupport,
  parseServerHelpSupport,
  reasoningArgs,
  setCachedServerHelpSupport,
} from "./llama-flash-attn";

/**
 * 真机证据（本机 llama.cpp 的 `--help` 原文）：
 *   `-fa, --flash-attn [on|off|auto]    set Flash Attention use ('on', 'off', or 'auto', default: 'auto')`
 * 老一些的 build 里只有不带值的 `-fa, --flash-attn`；更老的没有。
 */

describe("parseServerHelpSupport / flashAttn", () => {
  test("新版三态：同一行含 [on|off|auto] → tristate", () => {
    const help =
      "Options:\n" +
      "  -f,   --flash-attn [on|off|auto]    set Flash Attention use ('on', 'off', or 'auto', default: 'auto')\n";
    expect(parseServerHelpSupport(help).flashAttn).toBe("tristate");
  });

  test("新版三态（任务里给的原始措辞：-fa, --flash-attn [on|off|auto]）", () => {
    const help =
      "-fa, --flash-attn [on|off|auto]    set Flash Attention use ('on', 'off', or 'auto', default: 'auto')\n" +
      "      (env: LLAMA_ARG_FLASH_ATTN)\n";
    expect(parseServerHelpSupport(help).flashAttn).toBe("tristate");
  });

  test("老版布尔：只有 -fa, --flash-attn，没有取值列表 → boolean", () => {
    expect(
      parseServerHelpSupport("-fa, --flash-attn           whether to use Flash Attention").flashAttn,
    ).toBe("boolean");
    expect(
      parseServerHelpSupport("  -fa, --flash-attn\n      enable Flash Attention (default: off)").flashAttn,
    ).toBe("boolean");
  });

  test("没有 flash-attn → none", () => {
    expect(parseServerHelpSupport("").flashAttn).toBe("none");
    expect(parseServerHelpSupport("Usage: llama-server [options]").flashAttn).toBe("none");
  });

  test("load-mode 与 flash-attn 同一次解析（合并探测）", () => {
    const help =
      "-lm,   --load-mode MODE    model loading mode (default: auto)\n" +
      "  -fa, --flash-attn [on|off|auto]    set Flash Attention use\n";
    const support = parseServerHelpSupport(help);
    expect(support.loadMode).toBe("load-mode");
    expect(support.flashAttn).toBe("tristate");
  });
});

describe("flashAttnArgs", () => {
  test("tristate：三态原样发（auto 与不传等价但显式发出去，让 llama-server 日志打印实际值）", () => {
    expect(flashAttnArgs("auto", "tristate")).toEqual(["--flash-attn", "auto"]);
    expect(flashAttnArgs("on", "tristate")).toEqual(["--flash-attn", "on"]);
    expect(flashAttnArgs("off", "tristate")).toEqual(["--flash-attn", "off"]);
    expect(flashAttnArgs(null, "tristate")).toEqual(["--flash-attn", "auto"]);
  });

  test("boolean：只有 on 发（不带值），auto / off 不发", () => {
    expect(flashAttnArgs("on", "boolean")).toEqual(["--flash-attn"]);
    expect(flashAttnArgs("auto", "boolean")).toEqual([]);
    expect(flashAttnArgs("off", "boolean")).toEqual([]);
  });

  test("none：一个参数都不发（与加这个开关前逐字节一致）", () => {
    for (const v of ["auto", "on", "off", null]) {
      expect(flashAttnArgs(v, "none")).toEqual([]);
    }
  });

  test("非法值（手改设置行）：tristate 时回落 auto，boolean / none 不发", () => {
    expect(flashAttnArgs("--evil", "tristate")).toEqual(["--flash-attn", "auto"]);
    expect(flashAttnArgs("--evil", "boolean")).toEqual([]);
    expect(flashAttnArgs("--evil", "none")).toEqual([]);
  });
});

describe("support cache", () => {
  afterEach(() => clearServerHelpSupportCache());

  test("未探测 → null（同步路径按 none 处理）", () => {
    expect(cachedFlashAttnSupport("/bin/x")).toBeNull();
    expect(cachedServerHelpSupport("/bin/x")).toBeNull();
  });

  test("setCachedServerHelpSupport 后两个缓存可读；unknown load-mode 不落", () => {
    setCachedServerHelpSupport("/bin/x", { loadMode: "unknown", flashAttn: "tristate" });
    expect(cachedFlashAttnSupport("/bin/x")).toBe("tristate");
    const support = cachedServerHelpSupport("/bin/x");
    expect(support).not.toBeNull();
    // loadMode 是 unknown → 合并读回落 unknown（等价于「load-mode 还没探过」）
    expect(support!.loadMode).toBe("unknown");
    expect(support!.flashAttn).toBe("tristate");
  });

  test("load-mode 成功探测后 cachedServerHelpSupport 直接返回它", () => {
    setCachedServerHelpSupport("/bin/y", { loadMode: "load-mode", flashAttn: "boolean" });
    const support = cachedServerHelpSupport("/bin/y");
    expect(support?.loadMode).toBe("load-mode");
    expect(support?.flashAttn).toBe("boolean");
  });
});

describe("parseServerHelpSupport / kvUnified", () => {
  test("新版：-kvu, --kv-unified 那一行 → true", () => {
    const help =
      "-kvu,  --kv-unified                     use single unified KV buffer shared across all sequences\n" +
      "                                        (env: LLAMA_ARG_KV_UNIFIED)\n";
    expect(parseServerHelpSupport(help).kvUnified).toBe(true);
  });

  test("只有反向开关 --no-kv-unified 不算（不能据此发 --kv-unified）", () => {
    expect(parseServerHelpSupport("  --no-kv-unified    disable unified KV\n").kvUnified).toBe(false);
  });

  test("没有这个开关 → false", () => {
    expect(parseServerHelpSupport("").kvUnified).toBe(false);
    expect(parseServerHelpSupport("-fa, --flash-attn [on|off|auto]\n").kvUnified).toBe(false);
  });
});

describe("kvUnified support cache", () => {
  afterEach(() => clearServerHelpSupportCache());

  test("未探测 → null；写入后可读；clear 后回到 null", () => {
    expect(cachedKvUnifiedSupport("/bin/k")).toBeNull();
    setCachedServerHelpSupport("/bin/k", { loadMode: "load-mode", flashAttn: "tristate", kvUnified: true });
    expect(cachedKvUnifiedSupport("/bin/k")).toBe(true);
    expect(cachedServerHelpSupport("/bin/k")?.kvUnified).toBe(true);
    clearServerHelpSupportCache();
    expect(cachedKvUnifiedSupport("/bin/k")).toBeNull();
  });

  test("旧调用方不带 kvUnified → 不落缓存，合并读按不支持", () => {
    setCachedServerHelpSupport("/bin/k2", { loadMode: "load-mode", flashAttn: "none" });
    expect(cachedKvUnifiedSupport("/bin/k2")).toBeNull();
    expect(cachedServerHelpSupport("/bin/k2")?.kvUnified).toBe(false);
  });
});

describe("parseServerHelpSupport / reasoning", () => {
  test("新版：-rea, --reasoning [on|off|auto] → 支持（本机 llama-server --help 原文）", () => {
    const help =
      "--reasoning-format FORMAT               controls whether thought tags are allowed\n" +
      "-rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat ('on', 'off', or 'auto', default:\n" +
      "--reasoning-budget N                    token budget for thinking\n";
    expect(parseServerHelpSupport(help).reasoning).toBe(true);
  });

  test("老版只有 --reasoning-format / --reasoning-budget：同名前缀不算数", () => {
    const help =
      "--reasoning-format FORMAT               controls whether thought tags are allowed\n" +
      "--reasoning-budget N                    token budget for thinking\n" +
      "--chat-template-kwargs STRING           sets additional params for the json template parser\n";
    expect(parseServerHelpSupport(help).reasoning).toBe(false);
  });

  test("缓存：没探过 = null；注入后同步可读；清掉恢复未探测", () => {
    clearServerHelpSupportCache();
    expect(cachedReasoningSupport("/x/llama-server")).toBeNull();
    setCachedServerHelpSupport("/x/llama-server", { loadMode: "load-mode", flashAttn: "tristate", reasoning: true });
    expect(cachedReasoningSupport("/x/llama-server")).toBe(true);
    expect(cachedServerHelpSupport("/x/llama-server")?.reasoning).toBe(true);
    clearServerHelpSupportCache();
    expect(cachedReasoningSupport("/x/llama-server")).toBeNull();
  });
});

describe("reasoningArgs", () => {
  test("支持 --reasoning：on / off 原样发，auto / 没设不发", () => {
    expect(reasoningArgs("on", true)).toEqual(["--reasoning", "on"]);
    expect(reasoningArgs("off", true)).toEqual(["--reasoning", "off"]);
    expect(reasoningArgs("auto", true)).toEqual([]);
    expect(reasoningArgs(undefined, true)).toEqual([]);
  });

  test("不支持：关思考回落 chat-template-kwargs，开不发；非法值不发", () => {
    expect(reasoningArgs("off", false)).toEqual(["--chat-template-kwargs", '{"enable_thinking":false}']);
    expect(reasoningArgs("on", false)).toEqual([]);
    expect(reasoningArgs("--evil", true)).toEqual([]);
  });
});

describe("parseServerHelpSupport / 显存开关（--fit / --n-cpu-moe / -ot / --no-context-shift）", () => {
  // 本机 llama-server（build 11005）--help 的原文行
  const HELP = [
    "-ot,   --override-tensor <tensor name pattern>=<buffer type>,...",
    "-cmoe, --cpu-moe                        keep all Mixture of Experts (MoE) weights in the CPU",
    "-ncmoe, --n-cpu-moe N                   keep the Mixture of Experts (MoE) weights of the first N layers in the",
    "-fit,  --fit [on|off]                   whether to adjust unset arguments to fit in device memory ('on' or",
    "-fitt, --fit-target MiB0,MiB1,MiB2,...",
    "-fitc, --fit-ctx N                      minimum ctx size that can be set by --fit option, default: 4096",
    "--context-shift, --no-context-shift     whether to use context shift on infinite text generation (default:",
  ].join("\n");

  test("新版：四个都认", () => {
    const s = parseServerHelpSupport(HELP);
    expect(s.fit).toBe(true);
    expect(s.nCpuMoe).toBe(true);
    expect(s.overrideTensor).toBe(true);
    expect(s.noContextShift).toBe(true);
  });

  test("同名前缀的兄弟开关不算（--fit-target / --n-cpu-moe-draft / --override-tensor-draft）", () => {
    const s = parseServerHelpSupport(
      "-fitt, --fit-target MiB\n--spec-draft-n-cpu-moe, -ncmoed, --n-cpu-moe-draft N\n" +
        "--spec-draft-override-tensor, -otd, --override-tensor-draft <p>=<b>\n--context-shift  enable\n",
    );
    expect(s.fit).toBe(false);
    expect(s.nCpuMoe).toBe(false);
    expect(s.overrideTensor).toBe(false);
    expect(s.noContextShift).toBe(false);
  });

  test("缓存：没探过 = 全不支持；注入后同步读得到", () => {
    clearServerHelpSupportCache();
    expect(cachedMemoryFlagSupport("/x/llama-server")).toEqual({
      fit: false,
      nCpuMoe: false,
      overrideTensor: false,
      noContextShift: false,
    });
    setCachedServerHelpSupport("/x/llama-server", { loadMode: "load-mode", flashAttn: "tristate", fit: true, noContextShift: true });
    expect(cachedMemoryFlagSupport("/x/llama-server")).toEqual({
      fit: true,
      nCpuMoe: false,
      overrideTensor: false,
      noContextShift: true,
    });
    clearServerHelpSupportCache();
  });
});
