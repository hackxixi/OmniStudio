/**
 * Windows 上的「杀整棵进程树」与「按 pid 取映像名」。
 *
 * POSIX 这边靠 detached 进程组 + `process.kill(-pgid)`；Windows 没有进程组信号，
 * 负 pid 直接抛错，以前就退回成只杀直接子进程 —— python 启动器、shell 拉起的
 * 真正干活的进程留成孤儿，占着显存 / 端口直到重启。`taskkill /T` 按父子关系
 * 走整棵树；Windows 也没有 SIGTERM 的等价物（控制台程序收不到 WM_CLOSE），
 * 所以一律 `/F`。
 *
 * 命令执行可注入，参数与输出解析在非 Windows 的单测里也能断言。
 */

export type SyncRun = (cmd: string[]) => { exitCode: number | null; stdout: string };

const defaultRun: SyncRun = (cmd) => {
  const out = Bun.spawnSync(cmd, { stdout: "pipe", stderr: "ignore", windowsHide: true });
  return { exitCode: out.exitCode, stdout: out.stdout ? out.stdout.toString() : "" };
};

export function taskkillTreeArgs(pid: number): string[] {
  return ["taskkill", "/PID", String(pid), "/T", "/F"];
}

/** 杀掉以 pid 为根的整棵进程树；返回 taskkill 是否成功（进程不存在时为 false）。 */
export function killWindowsTree(pid: number, run: SyncRun = defaultRun): boolean {
  try {
    return run(taskkillTreeArgs(pid)).exitCode === 0;
  } catch {
    return false;
  }
}

export function tasklistArgs(pid: number): string[] {
  return ["tasklist", "/FI", `PID eq ${pid}`, "/FO", "CSV", "/NH"];
}

/**
 * 解析 `tasklist /FO CSV /NH` 的输出，取指定 pid 那一行的映像名（如 `llama-server.exe`）。
 * 没有匹配行（进程不存在时 tasklist 打印一行 INFO 提示）→ null。
 */
export function parseTasklistImage(output: string, pid: number): string | null {
  for (const line of output.split(/\r?\n/)) {
    const cells = [...line.matchAll(/"((?:[^"]|"")*)"/g)].map((m) => (m[1] ?? "").replace(/""/g, '"'));
    if (cells.length >= 2 && cells[1] === String(pid) && cells[0]) return cells[0];
  }
  return null;
}

/** 取 pid 对应进程的映像名；进程不存在 / 查询失败 → null。 */
export function windowsImageName(pid: number, run: SyncRun = defaultRun): string | null {
  try {
    const res = run(tasklistArgs(pid));
    if (res.exitCode !== 0) return null;
    return parseTasklistImage(res.stdout, pid);
  } catch {
    return null;
  }
}
