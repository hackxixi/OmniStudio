/**
 * 托管引擎的落盘位置 —— 「应用自己装的引擎住在哪」的唯一真源。
 *
 * 三条约束都在这里收口，别处不要再拼路径：
 *  - 与既有的媒体引擎同规矩：全部落在 `<dataDir>/engines/<名字>/`（cloudflared 用
 *    `current/` 存二进制、mflux 直接把 venv 放根目录），备份 / 排障时只看一个地方；
 *  - 解析顺序统一是**托管目录 → 系统 PATH**：应用自己装的版本自洽可复现，
 *    用户自己装过的仍然能用（只是不接管、不删除）；
 *  - 谁装的谁删得掉：卸载只动托管目录，PATH 上的东西一律不碰。
 */
import path, { join } from "path";

import { getDataDir } from "./paths";

const isWindows = process.platform === "win32";

/**
 * venv 里某个可执行文件的路径。POSIX 是 `bin/<name>`；Windows 是 `Scripts\<name>.exe`，
 * 且没有 `python3` / `pip3` 这种带版本号的解释器名（venv 只放 `python.exe` / `pip.exe`），
 * 所以这两个在 Windows 上映射成不带 3 的名字。所有托管 venv（推理引擎、PaddleOCR……）都走这里。
 */
export function venvExecutable(venvDir: string, name: string, platform: NodeJS.Platform = process.platform): string {
  if (platform !== "win32") return path.posix.join(venvDir, "bin", name);
  const base = name === "python3" ? "python" : name === "pip3" ? "pip" : name;
  return path.win32.join(venvDir, "Scripts", `${base}.exe`);
}

/** Python 引擎（mlx-lm / vLLM / SGLang）的 venv 目录：`<dataDir>/engines/<id>`。 */
export type PythonEngineId = "mlx-lm" | "vllm" | "sglang";

export function pythonEngineDir(id: PythonEngineId): string {
  return getDataDir("engines", id);
}

/** venv 里的可执行文件路径（Windows 是 `Scripts/`，其余是 `bin/`）。 */
export function pythonEngineBin(id: PythonEngineId, name: string): string {
  return venvExecutable(pythonEngineDir(id), name);
}

/** venv 的解释器：跑 `-m <module>` / `-c "import …"` 都用它。 */
export function pythonEnginePython(id: PythonEngineId): string {
  return venvExecutable(pythonEngineDir(id), "python3");
}

/** llama.cpp 托管安装的根（与 cloudflared 同构：暂存 → 原子 rename 到 current）。 */
export function llamaCppRootDir(): string {
  return getDataDir("engines", "llama.cpp");
}

export function llamaCppCurrentDir(): string {
  return join(llamaCppRootDir(), "current");
}

/** llama-server 可执行文件（托管安装的目标路径，也是 runtimes 优先解析的那个）。 */
export function llamaCppBinaryPath(): string {
  return join(llamaCppCurrentDir(), isWindows ? "llama-server.exe" : "llama-server");
}

/** 安装标记（记下装的是哪个官方构建，排障与「要不要升级」都读它）。 */
export function engineVersionFilePath(id: "llama.cpp" | PythonEngineId): string {
  return id === "llama.cpp"
    ? join(llamaCppCurrentDir(), "VERSION")
    : join(pythonEngineDir(id), "VERSION");
}
