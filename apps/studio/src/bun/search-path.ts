import { homedir } from "os";
import path from "path";

/**
 * 子进程 / `Bun.which` 用的 PATH 拼接，按平台走 —— 以前十来处各自 `join(":")`：
 * Windows 的分隔符是 `;`，用 `:` 拼会把原 PATH 的第一项和 `/opt/homebrew/bin` 这类
 * POSIX 目录粘成一个无效条目（`C:\Windows\system32` 就这么丢了），GUI 进程里找命令
 * 时好时坏。这里统一：分隔符按平台、Windows 上丢掉 POSIX 绝对目录、空条目不进 PATH
 * （POSIX 上的空条目等于「当前目录」，不该悄悄混进去）。
 *
 * 平台 / env / home 都可注入，Windows 路径在 Linux / macOS 的单测里也能断言。
 */

export type SearchPathOptions = {
  platform?: NodeJS.Platform;
  env?: Record<string, string | undefined>;
  /** 用户主目录；不传时 POSIX 读 `HOME`，Windows 读 `USERPROFILE` → `os.homedir()`。 */
  home?: string;
};

export function pathDelimiter(platform: NodeJS.Platform = process.platform): string {
  return platform === "win32" ? ";" : ":";
}

/** 读 PATH：Windows 上变量名是 `Path`，传进来的 env 若是普通对象就不会大小写不敏感。 */
export function currentPath(env: Record<string, string | undefined> = process.env): string {
  if (env.PATH !== undefined) return env.PATH;
  const key = Object.keys(env).find((k) => k.toUpperCase() === "PATH");
  return key ? (env[key] ?? "") : "";
}

function resolveHome(opts: SearchPathOptions, platform: NodeJS.Platform): string {
  if (opts.home !== undefined) return opts.home;
  const env = opts.env ?? process.env;
  if (platform === "win32") return env.USERPROFILE || homedir();
  return env.HOME ?? "";
}

/** 把 `extra` 目录放到当前 PATH 前面，按平台分隔；Windows 上 POSIX 绝对目录直接丢掉。 */
export function prependPath(extra: string[], opts: SearchPathOptions = {}): string {
  const platform = opts.platform ?? process.platform;
  const dirs = extra.filter((d) => d && !(platform === "win32" && d.startsWith("/")));
  const current = currentPath(opts.env ?? process.env);
  return [...dirs, ...(current ? [current] : [])].join(pathDelimiter(platform));
}

/**
 * 查找引擎 / 工具（llama-server、python、uv、tesseract、ffmpeg……）用的 PATH：
 * macOS 的 GUI 进程不继承 shell 的 PATH，只查 `Bun.which` 会漏掉 Homebrew 与 `~/.local/bin`
 * （uv 在三个平台上都装到这里）。
 */
export function toolSearchPath(opts: SearchPathOptions = {}): string {
  const platform = opts.platform ?? process.platform;
  const home = resolveHome(opts, platform);
  const p = platform === "win32" ? path.win32 : path.posix;
  const homeDirs = home ? [p.join(home, ".local", "bin"), p.join(home, "bin")] : [];
  const system = platform === "win32" ? [] : ["/opt/homebrew/bin", "/usr/local/bin"];
  return prependPath([...system, ...homeDirs], { ...opts, platform });
}

/**
 * Agent 的 bash 工具与 MCP stdio 服务器（npx / uvx / node）的 PATH：GUI 启动的进程
 * 常缺 Homebrew / nvm / bun / cargo 目录，补上常见路径免得莫名 "command not found"。
 */
export function agentShellPath(opts: SearchPathOptions = {}): string {
  const platform = opts.platform ?? process.platform;
  const home = resolveHome(opts, platform);
  if (platform === "win32") {
    const homeDirs = home ? [path.win32.join(home, ".bun", "bin"), path.win32.join(home, ".cargo", "bin")] : [];
    return prependPath(homeDirs, { ...opts, platform });
  }
  return prependPath(
    [
      "/usr/local/bin",
      "/opt/homebrew/bin",
      "/opt/homebrew/sbin",
      "/usr/bin",
      "/bin",
      "/usr/sbin",
      "/sbin",
      `${home}/.nvm/versions/node/*/bin`,
      `${home}/.bun/bin`,
      `${home}/.cargo/bin`,
    ],
    { ...opts, platform },
  );
}

/**
 * 返回一份把 PATH 换成 `value` 的 env：先删掉所有大小写变体（`Path` / `path`）。
 * Windows 上 `{ ...process.env, PATH }` 可能同时带着 `Path` 和 `PATH` 两份，
 * 子进程拿到哪一份没有保证。
 */
export function withPath(
  env: Record<string, string | undefined>,
  value: string,
): Record<string, string | undefined> {
  const out: Record<string, string | undefined> = {};
  for (const [k, v] of Object.entries(env)) {
    if (k.toUpperCase() !== "PATH") out[k] = v;
  }
  out.PATH = value;
  return out;
}
