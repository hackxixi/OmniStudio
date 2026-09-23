/**
 * 从「即将发出去 / 正在运行的」启动命令（argv）里还原出界面关心的几个参数的实际取值。
 *
 * 「启动」区的输入框留空 = 不覆盖（自动规划 / 全局），用户看不到最终给了多少；
 * 模型参数抽屉把命令里解析出来的值挂在「自动 / 跟随全局」后面（如「跟随全局 (q8_0)」）。
 *
 * 命令字符串由 bun 侧用 shell-args 的引号感知切分（`splitShellArgs`）变成 argv 再传进来 ——
 * shared 不能 import bun 目录（那边会连带初始化数据库 / electrobun）。
 */

export type EffectiveLaunch = {
  ctxSize?: number;
  parallel?: number;
  gpuLayers?: number;
  cacheTypeK?: string;
  cacheTypeV?: string;
  flashAttn?: string;
};

/**
 * `--flag value` 与 `--flag=value` 两种写法都认（llama.cpp 用前者，vLLM 用后者），
 * 同一参数出现多次取**最后一次**（两个引擎的解析器都是后者覆盖前者，命令里追加参数
 * 也拼在最后，所以「最后」就是实际生效的那次）。
 */
function lastFlagValue(argv: string[], flags: string[]): string | undefined {
  for (let i = argv.length - 1; i >= 0; i--) {
    const a = argv[i]!;
    for (const f of flags) {
      if (a === f) {
        const v = argv[i + 1];
        if (v !== undefined) return v;
      } else if (a.startsWith(f + "=")) {
        return a.slice(f.length + 1);
      }
    }
  }
  return undefined;
}

function num(v: string | undefined): number | undefined {
  if (v === undefined) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

/** argv → 实际生效值。认不出（MLX / SGLang 等）就留空字段，界面不加括号。 */
export function effectiveFromArgv(argv: string[]): EffectiveLaunch {
  const out: EffectiveLaunch = {};

  out.ctxSize = num(
    lastFlagValue(argv, ["-c", "--ctx-size", "--max-model-len"]),
  );
  out.parallel = num(lastFlagValue(argv, ["-np", "--parallel", "--max-num-seqs"]));
  out.gpuLayers = num(
    lastFlagValue(argv, ["-ngl", "--n-gpu-layers", "--gpu-layers"]),
  );
  const ctk = lastFlagValue(argv, ["-ctk", "--cache-type-k"]);
  if (ctk) out.cacheTypeK = ctk;
  const ctv = lastFlagValue(argv, ["-ctv", "--cache-type-v"]);
  if (ctv) out.cacheTypeV = ctv;

  // flash attention 特殊：老版是布尔开关，`-fa` 不带值就是开（--flash-attn 同理）。
  // 判「带不带值」得看下一个 token 是不是也以 `-` 开头（引擎开关/参数都以 `-` 开头），
  // 而不是判是否为空 —— `--flash-attn --temp 0.8` 里下一个 token 是另一个参数，不算值。
  for (let i = argv.length - 1; i >= 0; i--) {
    const a = argv[i]!;
    if (a === "-fa" || a === "--flash-attn") {
      const v = argv[i + 1];
      out.flashAttn = v !== undefined && !v.startsWith("-") ? v : "on";
      break;
    }
    if (a.startsWith("--flash-attn=")) {
      out.flashAttn = a.slice("--flash-attn=".length);
      break;
    }
  }

  return out;
}
