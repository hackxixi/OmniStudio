import { describe, expect, test } from "bun:test";

import { agentShellPath, currentPath, prependPath, toolSearchPath, withPath } from "./search-path";

describe("prependPath", () => {
  test("POSIX 用 : 拼接，空条目不进 PATH", () => {
    expect(prependPath(["/a", "", "/b"], { platform: "darwin", env: { PATH: "/usr/bin:/bin" } })).toBe(
      "/a:/b:/usr/bin:/bin",
    );
    expect(prependPath(["/a"], { platform: "linux", env: {} })).toBe("/a");
  });

  test("Windows 用 ; 拼接，丢掉 POSIX 绝对目录，原 PATH 第一项不被粘坏", () => {
    const env = { Path: "C:\\Windows\\system32;C:\\Windows" };
    expect(prependPath(["/opt/homebrew/bin", "C:\\tools"], { platform: "win32", env })).toBe(
      "C:\\tools;C:\\Windows\\system32;C:\\Windows",
    );
  });
});

describe("currentPath", () => {
  test("大小写不同的 Path 也读得到", () => {
    expect(currentPath({ Path: "C:\\x" })).toBe("C:\\x");
    expect(currentPath({ PATH: "/usr/bin" })).toBe("/usr/bin");
    expect(currentPath({})).toBe("");
  });
});

describe("toolSearchPath", () => {
  test("macOS：Homebrew + ~/.local/bin + ~/bin 在前", () => {
    expect(toolSearchPath({ platform: "darwin", home: "/Users/u", env: { PATH: "/usr/bin" } })).toBe(
      "/opt/homebrew/bin:/usr/local/bin:/Users/u/.local/bin:/Users/u/bin:/usr/bin",
    );
  });

  test("Windows：只补用户目录（反斜杠），分隔符为 ;", () => {
    expect(toolSearchPath({ platform: "win32", home: "C:\\Users\\u", env: { Path: "C:\\Windows" } })).toBe(
      "C:\\Users\\u\\.local\\bin;C:\\Users\\u\\bin;C:\\Windows",
    );
  });

  test("Windows 未传 home 时读 USERPROFILE", () => {
    expect(toolSearchPath({ platform: "win32", env: { USERPROFILE: "C:\\Users\\v", Path: "" } })).toBe(
      "C:\\Users\\v\\.local\\bin;C:\\Users\\v\\bin",
    );
  });
});

describe("agentShellPath", () => {
  test("POSIX 保留原有的系统目录与用户目录", () => {
    const p = agentShellPath({ platform: "linux", home: "/home/u", env: { PATH: "/x" } }).split(":");
    expect(p[0]).toBe("/usr/local/bin");
    expect(p).toContain("/home/u/.bun/bin");
    expect(p.at(-1)).toBe("/x");
  });

  test("Windows 不带任何 POSIX 目录", () => {
    const p = agentShellPath({ platform: "win32", home: "C:\\Users\\u", env: { Path: "C:\\Windows" } });
    expect(p).toBe("C:\\Users\\u\\.bun\\bin;C:\\Users\\u\\.cargo\\bin;C:\\Windows");
  });
});

describe("withPath", () => {
  test("删掉所有大小写变体，只留一份 PATH", () => {
    const env = withPath({ Path: "old", path: "old2", HOME: "/h" }, "new");
    expect(env).toEqual({ HOME: "/h", PATH: "new" });
  });
});
