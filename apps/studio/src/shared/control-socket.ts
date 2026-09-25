import { join } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";

/**
 * Unix domain socket 路径的字节上限。`sun_path` 字段 macOS 104 字节、
 * 其他平台 108 字节，最后 1 字节必须留给结尾的 `\0`，所以可用长度
 * 分别只剩 103 / 107。超限时 `Bun.serve({ unix })` 直接抛
 * ENAMETOOLONG，控制通道起不来 —— 主进程与 CLI 共用本文件里的算法，
 * 保证两边算出的是同一个路径。
 */
export const SOCKET_PATH_MAX_BYTES: number = process.platform === "darwin" ? 103 : 107;

const SOCKET_NAME = "omni-control.sock";

/** 路径的字节长度（UTF-8；中文路径每字符 3 字节，按字符串长度算会漏判）。 */
function byteLength(s: string): number {
  return Buffer.byteLength(s, "utf8");
}

function hashSuffix(dataDir: string): string {
  return createHash("sha256").update(dataDir, "utf8").digest("hex").slice(0, 16);
}

/**
 * 数据目录对应的控制 socket 路径：默认 `<dataDir>/omni-control.sock`；
 * 超过平台上限时改用 `<os.tmpdir()>/omni-<sha256(dataDir) 前 16 位>.sock`
 * （同一数据目录总落到同一个文件，主进程与 CLI 算得一致）。
 * `override`（`OMNI_CONTROL_SOCKET`）优先且原样返回。
 *
 * tmpdir 本身若也超长（极少见），退到 `/tmp/omni-<hash>.sock`。
 */
export function controlSocketPathFor(
  dataDir: string,
  opts?: { override?: string; tmpDir?: string; platform?: string },
): string {
  if (opts?.override) return opts.override;

  const limit = opts?.platform === "darwin" ? 103 : 107;
  const defaultPath = join(dataDir, SOCKET_NAME);
  if (byteLength(defaultPath) <= limit) return defaultPath;

  const base = opts?.tmpDir ?? tmpdir();
  const name = `omni-${hashSuffix(dataDir)}.sock`;
  const p1 = join(base, name);
  if (byteLength(p1) <= limit) return p1;
  return join("/tmp", name);
}
