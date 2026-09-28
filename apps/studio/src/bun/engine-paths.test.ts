import { describe, expect, test } from "bun:test";

import { venvExecutable } from "./engine-paths";

describe("venvExecutable", () => {
  test("POSIX：bin/<name>，名字原样", () => {
    expect(venvExecutable("/d/engines/paddleocr", "python3", "darwin")).toBe("/d/engines/paddleocr/bin/python3");
    expect(venvExecutable("/d/engines/vllm", "pip3", "linux")).toBe("/d/engines/vllm/bin/pip3");
  });

  test("Windows：Scripts\\<name>.exe，python3 / pip3 映射成不带版本号的名字", () => {
    const dir = "C:\\Users\\u\\AppData\\Local\\omni\\engines\\paddleocr";
    expect(venvExecutable(dir, "python3", "win32")).toBe(`${dir}\\Scripts\\python.exe`);
    expect(venvExecutable(dir, "pip3", "win32")).toBe(`${dir}\\Scripts\\pip.exe`);
    expect(venvExecutable(dir, "paddleocr", "win32")).toBe(`${dir}\\Scripts\\paddleocr.exe`);
  });
});
