import { describe, expect, test } from "bun:test";

import { killWindowsTree, parseTasklistImage, taskkillTreeArgs, windowsImageName } from "./win-process";

describe("taskkill", () => {
  test("按 pid 杀整棵树并强制", () => {
    expect(taskkillTreeArgs(42)).toEqual(["taskkill", "/PID", "42", "/T", "/F"]);
  });

  test("退出码 0 才算成功，执行抛错不外泄", () => {
    expect(killWindowsTree(42, () => ({ exitCode: 0, stdout: "" }))).toBe(true);
    expect(killWindowsTree(42, () => ({ exitCode: 128, stdout: "" }))).toBe(false);
    expect(
      killWindowsTree(42, () => {
        throw new Error("ENOENT");
      }),
    ).toBe(false);
  });
});

describe("tasklist", () => {
  const out = '"llama-server.exe","4321","Console","1","1,234,567 K"\r\n';

  test("取匹配 pid 那一行的映像名", () => {
    expect(parseTasklistImage(out, 4321)).toBe("llama-server.exe");
    expect(parseTasklistImage(out, 1234)).toBeNull();
  });

  test("进程不存在时 tasklist 只打印 INFO 行", () => {
    expect(parseTasklistImage("INFO: No tasks are running which match the specified criteria.\r\n", 4321)).toBeNull();
  });

  test("查询失败返回 null", () => {
    expect(windowsImageName(4321, () => ({ exitCode: 0, stdout: out }))).toBe("llama-server.exe");
    expect(windowsImageName(4321, () => ({ exitCode: 1, stdout: "" }))).toBeNull();
  });
});
