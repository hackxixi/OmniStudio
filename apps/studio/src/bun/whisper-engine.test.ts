import { describe, expect, test } from "bun:test";

import type { SourcePlan } from "../shared/net-sources";
import { compareCondaVersion, condaUrls, pickFromRepodata, resolveCondaFiles } from "./whisper-engine";

/**
 * whisper.cpp（macOS 走 conda-forge 预编译包）的版本解析与下载地址。
 * api.anaconda.org 在国内时通时不通 —— 问不到就读频道索引（镜像上的 current_repodata.json）。
 */
function planOf(mode: "cn" | "global"): SourcePlan {
  return {
    mode,
    decidedBy: "setting",
    cnLocale: mode === "cn",
    modelSource: "huggingface",
    hfEndpoints: ["https://huggingface.co"],
    pypiIndexes: ["https://pypi.org/simple"],
    githubPrefixes: [""],
    homebrewEnv: {},
    probes: [],
    at: Date.now(),
  };
}

describe("conda 版本与索引", () => {
  test("版本号按段数值比较", () => {
    expect(compareCondaVersion("1.9.3", "1.10.0")).toBe(-1);
    expect(compareCondaVersion("23.1.1", "22.1.8")).toBe(1);
    expect(compareCondaVersion("1.8", "1.8.0")).toBe(0);
  });

  test("从 repodata 挑最新：版本 → build 号 → 非 debug → 最新上传；basename 带子目录", () => {
    const repodata = {
      packages: { "libcxx-11.1.0-h168391b_0.tar.bz2": { name: "libcxx", version: "11.1.0", build_number: 0 } },
      "packages.conda": {
        "libcxx-21.1.8-hf598326_1.conda": { name: "libcxx", version: "21.1.8", build: "hf598326_1", build_number: 1 },
        "libcxx-23.1.1-debug_h1659566_0.conda": { name: "libcxx", version: "23.1.1", build: "debug_h1659566_0", build_number: 0, timestamp: 9 },
        "libcxx-23.1.1-h55c6f16_0.conda": { name: "libcxx", version: "23.1.1", build: "h55c6f16_0", build_number: 0, timestamp: 5 },
        "whisper.cpp-1.9.3-h1b8b2da_0.conda": { name: "whisper.cpp", version: "1.9.3", build: "h1b8b2da_0", build_number: 0 },
        "whisper.cpp-1.10.0-h2_0.conda": { name: "whisper.cpp", version: "1.10.0", build: "h2_0", build_number: 0 },
      },
    };
    expect(pickFromRepodata(repodata, "libcxx", "osx-arm64")).toEqual({
      pkg: "libcxx",
      version: "23.1.1",
      basename: "osx-arm64/libcxx-23.1.1-h55c6f16_0.conda",
    });
    expect(pickFromRepodata(repodata, "whisper.cpp", "osx-arm64")?.version).toBe("1.10.0");
    expect(pickFromRepodata(repodata, "llvm-openmp", "osx-arm64")).toBeNull();
  });

  test("下载地址：镜像是 <频道>/<子目录>/<文件>（不是 api 的 <包>/<版本>/… 结构）", () => {
    const urls = condaUrls({ pkg: "whisper.cpp", version: "1.9.3", basename: "osx-arm64/whisper.cpp-1.9.3-h1b8b2da_0.conda" });
    expect(urls).toEqual([
      "https://api.anaconda.org/download/conda-forge/whisper.cpp/1.9.3/osx-arm64/whisper.cpp-1.9.3-h1b8b2da_0.conda",
      "https://conda.anaconda.org/conda-forge/osx-arm64/whisper.cpp-1.9.3-h1b8b2da_0.conda",
      "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/osx-arm64/whisper.cpp-1.9.3-h1b8b2da_0.conda",
      "https://mirror.sjtu.edu.cn/anaconda/cloud/conda-forge/osx-arm64/whisper.cpp-1.9.3-h1b8b2da_0.conda",
    ]);
  });
});

describe("resolveCondaFiles", () => {
  const pkgs = ["whisper.cpp", "llvm-openmp", "libcxx"];
  const file = (pkg: string, v = "1.0") => ({ pkg, version: v, basename: `osx-arm64/${pkg}-${v}-h_0.conda` });

  test("API 都问得到：不读索引（与原来一致）；国内给 API 短超时", async () => {
    const timeouts: number[] = [];
    let indexCalls = 0;
    const out = await resolveCondaFiles(pkgs, planOf("cn"), {
      fromApi: async (pkg, t) => {
        timeouts.push(t);
        return file(pkg);
      },
      fromRepodata: async () => {
        indexCalls += 1;
        return new Map();
      },
    });
    expect(out.map((f) => f?.pkg)).toEqual(pkgs);
    expect(indexCalls).toBe(0);
    expect(timeouts.every((t) => t === 10_000)).toBe(true);
  });

  test("API 有一个问不到 → 读频道索引补齐（API 拿到的保留）", async () => {
    const out = await resolveCondaFiles(pkgs, planOf("global"), {
      fromApi: async (pkg, t) => {
        expect(t).toBe(30_000);
        return pkg === "libcxx" ? null : file(pkg);
      },
      fromRepodata: async (want) => new Map(want.map((pkg) => [pkg, file(pkg, "9.9")])),
    });
    expect(out.map((f) => f?.version)).toEqual(["1.0", "1.0", "9.9"]);
  });
});
