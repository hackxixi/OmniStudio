import { existsSync } from "fs";
import { spawn } from "bun";
import { appBundlePath, resolveControlSocket } from "./data-dir";

export type ControlResult = {
  connected: boolean;
  ok: boolean;
  data?: any;
  error?: string;
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * 向运行中的应用发控制命令（HTTP over unix socket）。
 * `connected=false` 表示应用没在运行（没有活着的控制通道）。
 *
 * socket 用 `resolveControlSocket()` 现探：dev / canary / stable 各有一份数据目录，
 * 只看安装包或"socket 文件在不在"会在应用真的跑着时说"未运行"（残留 socket 更骗人）。
 */
export async function controlRequest(
  cmd: string,
  payload?: Record<string, unknown>,
  timeoutMs = 20000,
): Promise<ControlResult> {
  const socketPath = await resolveControlSocket();
  if (!socketPath || !existsSync(socketPath)) {
    return { connected: false, ok: false, error: "应用未运行（控制 socket 不存在）" };
  }
  try {
    const res = await fetch("http://control", {
      unix: socketPath,
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ cmd, payload }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return { connected: true, ok: false, error: `控制服务返回 HTTP ${res.status}` };
    const body = (await res.json()) as { ok: boolean; data?: unknown; error?: string };
    return { connected: true, ok: body.ok, data: body.data, error: body.error };
  } catch (err) {
    return { connected: false, ok: false, error: String(err) };
  }
}

export async function isAppRunning(): Promise<boolean> {
  const r = await controlRequest("ping", undefined, 2000);
  return r.connected && r.ok;
}

/**
 * 按平台拼「拉起应用」的命令。`wait` = 这条命令会很快返回（open / start 只负责交给系统），
 * 可以等它的退出码；为 false 时直接运行的就是应用本身，只能 detached 放走。
 * 返回 null = 不知道应用装在哪（非 macOS 目前没有固定安装位置可猜，需要 --app-path）。
 */
export function launchCommand(
  target: string | null,
  platform: NodeJS.Platform = process.platform,
): { cmd: string[]; wait: boolean } | null {
  if (platform === "darwin") return { cmd: target ? ["open", target] : ["open", "-a", "OmniStudio"], wait: true };
  if (!target) return null;
  // `start` 的第一个带引号参数是窗口标题，空串占位，否则带空格的路径会被当成标题。
  if (platform === "win32") return { cmd: ["cmd", "/c", "start", "", target], wait: true };
  return { cmd: [target], wait: false };
}

/** 唤起已安装的 OmniStudio（macOS `open` / Windows `start` 拉起 + 轮询控制 socket 就绪）。 */
export async function launchApp(appPath?: string): Promise<boolean> {
  const target = appPath && existsSync(appPath) ? appPath : appBundlePath();
  const launch = launchCommand(target);
  const hint =
    process.platform === "darwin" ? "--app-path <OmniStudio.app>" : "--app-path <OmniStudio 可执行文件路径>";
  if (!launch) {
    console.error(`找不到 OmniStudio 的安装位置，请指定 ${hint}`);
    return false;
  }
  const proc = spawn(launch.cmd, {
    stdout: "ignore",
    stderr: launch.wait ? "pipe" : "ignore",
    detached: !launch.wait,
  });
  if (!launch.wait) {
    proc.unref();
    return true;
  }
  const code = await proc.exited;
  if (code !== 0) {
    console.error(`无法启动 OmniStudio 应用，请确认已安装，或指定 ${hint}`);
    return false;
  }
  return true;
}

/** 确保应用在运行；没运行就拉起并等待控制 socket 就绪。 */
export async function ensureAppRunning(opts?: {
  appPath?: string;
  waitMs?: number;
}): Promise<boolean> {
  if (await isAppRunning()) return true;
  const launched = await launchApp(opts?.appPath);
  if (!launched) return false;
  const timeout = opts?.waitMs ?? 20000;
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (await isAppRunning()) return true;
    await sleep(400);
  }
  return false;
}
