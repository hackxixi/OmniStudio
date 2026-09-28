import { describe, expect, test } from "bun:test";

import { isSafeArchiveEntry, zipExtractArgs, zipListArgs } from "./installer";

describe("zip 列 / 解命令", () => {
  test("macOS / Linux 用 unzip", () => {
    expect(zipListArgs("/a.zip", "darwin")).toEqual(["unzip", "-Z1", "/a.zip"]);
    expect(zipExtractArgs("/a.zip", "/t", "linux")).toEqual(["unzip", "-q", "-o", "/a.zip", "-d", "/t"]);
  });

  test("Windows 用 System32 的 bsdtar（不走 PATH，避开 Git 的 GNU tar）", () => {
    const list = zipListArgs("C:\\a.zip", "win32");
    expect(list[0]).toMatch(/\\System32\\tar\.exe$/);
    expect(list.slice(1)).toEqual(["-tf", "C:\\a.zip"]);
    expect(zipExtractArgs("C:\\a.zip", "C:\\t", "win32").slice(1)).toEqual(["-xf", "C:\\a.zip", "-C", "C:\\t"]);
  });
});

describe("isSafeArchiveEntry", () => {
  test("拒绝绝对路径、盘符、.. 段（含反斜杠写法）", () => {
    expect(isSafeArchiveEntry("skill/SKILL.md")).toBe(true);
    expect(isSafeArchiveEntry("/etc/passwd")).toBe(false);
    expect(isSafeArchiveEntry("C:\\Windows\\x")).toBe(false);
    expect(isSafeArchiveEntry("a\\..\\..\\x")).toBe(false);
  });
});
