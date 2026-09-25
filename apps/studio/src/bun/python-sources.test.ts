import { afterEach, describe, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import path from "path";

import type { SourcePlan } from "../shared/net-sources";
import type { CommandRunner } from "./command-runner";
import {
  installEnv,
  installFromIndexes,
  installUv,
  managedUvDir,
  managedUvPath,
  planPypiIndexes,
  pypiIndexArgs,
  pypiIndexLabel,
  pythonChildEnv,
  resolveUv,
  uvInstallOrder,
  uvReleaseAsset,
  type UvDownloader,
} from "./python-sources";

/**
 * Python 侧下载源：索引轮换顺序、uv 安装的路径选择。
 * 全部用固定计划 + 假 runner / 假下载器：不联网、不真的装东西。
 */
function planOf(mode: "cn" | "global", extra: Partial<SourcePlan> = {}): SourcePlan {
  return {
    mode,
    decidedBy: "setting",
    cnLocale: mode === "cn",
    modelSource: mode === "cn" ? "modelscope" : "huggingface",
    hfEndpoints: mode === "cn" ? ["https://hf-mirror.com", "https://huggingface.co"] : ["https://huggingface.co"],
    pypiIndexes:
      mode === "cn"
        ? ["https://mirrors.aliyun.com/pypi/simple", "https://pypi.tuna.tsinghua.edu.cn/simple", "https://pypi.org/simple"]
        : ["https://pypi.org/simple", "https://mirrors.aliyun.com/pypi/simple"],
    githubPrefixes: mode === "cn" ? ["https://gh-proxy.com/", "https://ghfast.top/", ""] : ["", "https://gh-proxy.com/"],
    homebrewEnv:
      mode === "cn"
        ? {
            HOMEBREW_API_DOMAIN: "https://mirrors.ustc.edu.cn/homebrew-bottles/api",
            HOMEBREW_BOTTLE_DOMAIN: "https://mirrors.ustc.edu.cn/homebrew-bottles",
          }
        : {},
    probes: [],
    at: Date.now(),
    ...extra,
  };
}

const tmpDirs: string[] = [];
function tempDir(): string {
  const dir = mkdtempSync(path.join(tmpdir(), "omni-uv-test-"));
  tmpDirs.push(dir);
  return dir;
}

afterEach(() => {
  rmSync(managedUvDir(), { recursive: true, force: true });
  for (const dir of tmpDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function writeExe(file: string): void {
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, "#!/bin/sh\necho uv 0.0.0\n");
  chmodSync(file, 0o755);
}

describe("PyPI 索引", () => {
  test("索引名给人看：阿里云 / 清华 / 官方，未知主机带上主机名", () => {
    expect(pypiIndexLabel("https://mirrors.aliyun.com/pypi/simple")).toBe("阿里云 PyPI 镜像");
    expect(pypiIndexLabel("https://pypi.tuna.tsinghua.edu.cn/simple")).toBe("清华 PyPI 镜像");
    expect(pypiIndexLabel("https://pypi.org/simple")).toBe("官方 PyPI 源");
    expect(pypiIndexLabel("https://pypi.example.com/simple")).toBe("PyPI 镜像（pypi.example.com）");
  });

  test("官方源不带参数（尊重用户自己的 pip.conf），镜像用 --index-url（pip 与 uv pip 都认）", () => {
    expect(pypiIndexArgs("https://pypi.org/simple/")).toEqual([]);
    expect(pypiIndexArgs("https://mirrors.aliyun.com/pypi/simple/")).toEqual([
      "--index-url",
      "https://mirrors.aliyun.com/pypi/simple",
    ]);
  });

  test("空计划兜底官方源；重复项去掉", () => {
    expect(planPypiIndexes(planOf("global", { pypiIndexes: [] }))).toEqual(["https://pypi.org/simple"]);
    expect(
      planPypiIndexes(planOf("global", { pypiIndexes: ["https://pypi.org/simple/", "https://pypi.org/simple"] })),
    ).toEqual(["https://pypi.org/simple"]);
  });

  test("按计划顺序逐个试：首选成功就不再换", async () => {
    const tried: string[] = [];
    const logs: string[] = [];
    const result = await installFromIndexes({
      plan: planOf("cn"),
      what: "mflux",
      log: (l) => logs.push(l),
      run: async (_args, index) => {
        tried.push(index);
        return 0;
      },
    });
    expect(result).toEqual({ code: 0, index: "https://mirrors.aliyun.com/pypi/simple" });
    expect(tried).toEqual(["https://mirrors.aliyun.com/pypi/simple"]);
    expect(logs).toEqual(["用 阿里云 PyPI 镜像安装 mflux…"]);
  });

  test("失败依次换下一个，全部失败返回最后的退出码", async () => {
    const tried: string[][] = [];
    const logs: string[] = [];
    const result = await installFromIndexes({
      plan: planOf("cn"),
      what: "laya-mlx",
      log: (l) => logs.push(l),
      run: async (args) => {
        tried.push(args);
        return 2;
      },
    });
    expect(result.code).toBe(2);
    expect(tried).toEqual([
      ["--index-url", "https://mirrors.aliyun.com/pypi/simple"],
      ["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"],
      [],
    ]);
    expect(logs[1]).toBe("阿里云 PyPI 镜像安装失败（退出码 2），改用 清华 PyPI 镜像重试…");
    expect(logs[2]).toBe("清华 PyPI 镜像安装失败（退出码 2），改用 官方 PyPI 源重试…");
  });

  test("安装环境：带镜像变量，但默认索引变量拿掉（索引由参数逐次决定）", () => {
    const env = installEnv(planOf("cn"));
    expect(env.PIP_INDEX_URL).toBeUndefined();
    expect(env.UV_DEFAULT_INDEX).toBeUndefined();
    expect(env.UV_PYTHON_INSTALL_MIRROR).toStartWith("https://gh-proxy.com/https://github.com/astral-sh/");
    expect(env.HF_ENDPOINT).toBe("https://hf-mirror.com");
    expect(env.HOMEBREW_BOTTLE_DOMAIN).toBe("https://mirrors.ustc.edu.cn/homebrew-bottles");
  });

  test("Python 子进程环境带上 HF 端点候选（脚本失败时逐个换）", () => {
    expect(pythonChildEnv(planOf("cn")).OMNI_HF_ENDPOINTS).toBe("https://hf-mirror.com,https://huggingface.co");
    expect(pythonChildEnv(planOf("global")).OMNI_HF_ENDPOINTS).toBe("https://huggingface.co");
  });
});

describe("uv 安装", () => {
  test("官方单文件包名按平台 / 架构拼", () => {
    expect(uvReleaseAsset("darwin", "arm64")).toBe("uv-aarch64-apple-darwin.tar.gz");
    expect(uvReleaseAsset("darwin", "x64")).toBe("uv-x86_64-apple-darwin.tar.gz");
    expect(uvReleaseAsset("linux", "x64")).toBe("uv-x86_64-unknown-linux-gnu.tar.gz");
    expect(uvReleaseAsset("win32", "x64")).toBe("uv-x86_64-pc-windows-msvc.zip");
    expect(uvReleaseAsset("linux", "s390x")).toBeNull();
  });

  test("顺序：国内先镜像（单文件包 → PyPI → brew），海外 brew / 官方脚本在前", () => {
    expect(uvInstallOrder(planOf("cn"), true)).toEqual(["github", "pypi", "brew"]);
    expect(uvInstallOrder(planOf("cn"), false)).toEqual(["github", "pypi"]);
    expect(uvInstallOrder(planOf("global"), true)).toEqual(["brew", "script", "github", "pypi"]);
    expect(uvInstallOrder(planOf("global"), false)).toEqual(["script", "github", "pypi"]);
  });

  test("托管目录里的 uv 优先于 PATH", () => {
    const bin = tempDir();
    writeExe(path.join(bin, "uv"));
    expect(resolveUv(bin)).toBe(path.join(bin, "uv"));
    writeExe(managedUvPath());
    expect(resolveUv(bin)).toBe(managedUvPath());
  });

  test("国内：经 GitHub 加速镜像下单文件包，装进托管目录（不碰 brew）", async () => {
    const calls: string[][] = [];
    const runner: CommandRunner = {
      run: (cmd) => {
        calls.push(cmd);
        // 假解包：tar -xf <file> -C <dir> → 在目标目录里放出官方包的目录结构
        if (cmd[0] === "tar") writeExe(path.join(cmd[4]!, "uv-aarch64-apple-darwin", "uv"));
        return { code: 0, stdout: "", stderr: "" };
      },
      runStreaming: async (cmd) => {
        calls.push(cmd);
        return 0;
      },
    };
    let seenUrls: string[] = [];
    const download: UvDownloader = async ({ urls, accept, dest }) => {
      seenUrls = urls;
      // 假装落盘了一个包：accept 里的解包由假 runner 接住
      writeFileSync(dest, "x");
      const err = await accept(dest);
      return err ? { ok: false, error: err } : { ok: true };
    };
    const result = await installUv({
      plan: planOf("cn"),
      log: () => {},
      searchPath: tempDir(),
      source: "systemone",
      runner,
      platform: "darwin",
      arch: "arm64",
      download,
      findBrew: () => "/opt/homebrew/bin/brew",
    });
    expect(result).toEqual({ ok: true, path: managedUvPath(), via: "github" });
    expect(seenUrls[0]).toBe(
      "https://gh-proxy.com/https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz",
    );
    expect(seenUrls[seenUrls.length - 1]).toBe(
      "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz",
    );
    expect(calls.some((c) => c.includes("brew") || c[0]!.endsWith("brew"))).toBe(false);
    // 临时解包目录与下载文件都清掉了
    expect(existsSync(path.join(managedUvDir(), `.staging-${process.pid}`))).toBe(false);
  });

  test("单文件包下不下来 → 用已有 Python 从 PyPI 镜像 pip install uv", async () => {
    const streamed: { cmd: string[]; env?: Record<string, string> }[] = [];
    const runner: CommandRunner = {
      run: () => ({ code: 0, stdout: "", stderr: "" }),
      runStreaming: async (cmd, _onLine, opts) => {
        streamed.push({ cmd, env: opts?.env });
        if (cmd[1] === "-m" && cmd[2] === "venv") {
          writeExe(path.join(cmd[3]!, "bin", "pip"));
          return 0;
        }
        if (cmd.includes("install")) {
          writeExe(path.join(path.dirname(cmd[0]!), "uv"));
          return 0;
        }
        return 1;
      },
    };
    const result = await installUv({
      plan: planOf("cn"),
      log: () => {},
      searchPath: tempDir(),
      source: "systemone",
      runner,
      platform: "darwin",
      arch: "arm64",
      download: async () => ({ ok: false, error: "all mirrors down" }),
      findAnyPython: () => "/usr/local/bin/python3",
      findBrew: () => null,
    });
    expect(result.ok).toBe(true);
    expect(result.via).toBe("pypi");
    expect(existsSync(managedUvPath())).toBe(true);
    const pip = streamed.find((s) => s.cmd.includes("install"))!;
    expect(pip.cmd.join(" ")).toContain("install --disable-pip-version-check uv --index-url https://mirrors.aliyun.com/pypi/simple");
    expect(pip.env?.PIP_INDEX_URL).toBeUndefined();
    // 临时 venv 用完就删
    expect(existsSync(path.join(managedUvDir(), ".pip-venv"))).toBe(false);
  });

  test("国内镜像都不行 → 最后才 brew，且带上 Homebrew 镜像变量", async () => {
    const pathDir = tempDir();
    const brewCalls: { cmd: string[]; env?: Record<string, string> }[] = [];
    const runner: CommandRunner = {
      run: () => ({ code: 0, stdout: "", stderr: "" }),
      runStreaming: async (cmd, _onLine, opts) => {
        if (cmd[0] === "/opt/homebrew/bin/brew") {
          brewCalls.push({ cmd, env: opts?.env });
          writeExe(path.join(pathDir, "uv"));
          return 0;
        }
        return 1;
      },
    };
    const result = await installUv({
      plan: planOf("cn"),
      log: () => {},
      searchPath: pathDir,
      source: "systemone",
      runner,
      platform: "darwin",
      arch: "arm64",
      download: async () => ({ ok: false, error: "down" }),
      findAnyPython: () => null,
      findBrew: () => "/opt/homebrew/bin/brew",
    });
    expect(result).toEqual({ ok: true, path: path.join(pathDir, "uv"), via: "brew" });
    expect(brewCalls).toHaveLength(1);
    expect(brewCalls[0]!.cmd).toEqual(["/opt/homebrew/bin/brew", "install", "uv"]);
    expect(brewCalls[0]!.env?.HOMEBREW_BOTTLE_DOMAIN).toBe("https://mirrors.ustc.edu.cn/homebrew-bottles");
    expect(brewCalls[0]!.env?.HOMEBREW_API_DOMAIN).toBe("https://mirrors.ustc.edu.cn/homebrew-bottles/api");
  });

  test("海外：brew 在前；brew 失败换官方脚本", async () => {
    const pathDir = tempDir();
    const order: string[] = [];
    const runner: CommandRunner = {
      run: () => ({ code: 0, stdout: "", stderr: "" }),
      runStreaming: async (cmd) => {
        if (cmd[0] === "/usr/local/bin/brew") {
          order.push("brew");
          return 1;
        }
        if (cmd[0] === "/bin/sh") {
          order.push("script");
          writeExe(path.join(pathDir, "uv"));
          return 0;
        }
        return 1;
      },
    };
    const result = await installUv({
      plan: planOf("global"),
      log: () => {},
      searchPath: pathDir,
      source: "systemone",
      runner,
      download: async () => {
        throw new Error("不该走到 GitHub 下载");
      },
      findBrew: () => "/usr/local/bin/brew",
    });
    expect(order).toEqual(["brew", "script"]);
    expect(result.via).toBe("script");
  });

  test("全部失败：如实报错，带上每条路的原因", async () => {
    const result = await installUv({
      plan: planOf("cn"),
      log: () => {},
      searchPath: tempDir(),
      source: "systemone",
      runner: { run: () => ({ code: 1, stdout: "", stderr: "" }), runStreaming: async () => 1 },
      platform: "darwin",
      arch: "arm64",
      download: async () => ({ ok: false, error: "down" }),
      findAnyPython: () => null,
      findBrew: () => null,
    });
    expect(result.ok).toBe(false);
    expect(result.error).toContain("github");
    expect(result.error).toContain("pypi");
  });
});
