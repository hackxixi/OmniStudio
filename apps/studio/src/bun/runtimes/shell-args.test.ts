import { describe, expect, test } from "bun:test";

import { shellJoin, shellQuote, splitShellArgs } from "./shell-args";

describe("splitShellArgs", () => {
  test("空白分隔，多余空白 / 换行不产生空参数", () => {
    expect(splitShellArgs("  --a 1\n\t--b   2 ")).toEqual(["--a", "1", "--b", "2"]);
    expect(splitShellArgs("")).toEqual([]);
  });

  test("单引号内全字面量（带空格的 JSON 不被拆开）", () => {
    expect(splitShellArgs(`--chat-template-kwargs '{"enable_thinking": false}'`)).toEqual([
      "--chat-template-kwargs",
      '{"enable_thinking": false}',
    ]);
    expect(splitShellArgs(`'a \\b'`)).toEqual(["a \\b"]);
  });

  test("双引号：认 \\\" \\\\ \\$ 转义，其余反斜杠保留", () => {
    expect(splitShellArgs(`"hi \\"there\\"" "a\\\\b" "\\$HOME" "x\\ny"`)).toEqual([
      'hi "there"',
      "a\\b",
      "$HOME",
      "x\\ny",
    ]);
  });

  test("引号外反斜杠转义下一个字符；引号可与普通字符拼成一个参数", () => {
    expect(splitShellArgs(`a\\ b --x=' y'z`)).toEqual(["a b", "--x= yz"]);
  });

  test("'' / \"\" 产生一个空参数", () => {
    expect(splitShellArgs(`--a '' --b ""`)).toEqual(["--a", "", "--b", ""]);
  });

  test("不做展开：$(...) / 通配符原样（argv 不经 shell）", () => {
    expect(splitShellArgs("$(rm -rf /) *.gguf")).toEqual(["$(rm", "-rf", "/)", "*.gguf"]);
  });

  test("引号没闭合：宽松收进当前参数，不抛", () => {
    expect(splitShellArgs(`--a 'b c`)).toEqual(["--a", "b c"]);
    expect(splitShellArgs(`--a "b c`)).toEqual(["--a", "b c"]);
  });
});

describe("shellQuote / shellJoin", () => {
  test("安全字符原样，其余单引号包裹", () => {
    expect(shellQuote("--ctx-size")).toBe("--ctx-size");
    expect(shellQuote("/a/b.gguf")).toBe("/a/b.gguf");
    expect(shellQuote("/Library/Application Support/m.gguf")).toBe("'/Library/Application Support/m.gguf'");
    expect(shellQuote("it's")).toBe(`'it'\\''s'`);
    expect(shellQuote("")).toBe("''");
  });

  test("往返：splitShellArgs(shellJoin(argv)) 与 argv 逐项相等", () => {
    const argv = [
      "/Users/x/Application Support/llama-server",
      "--chat-template-kwargs",
      '{"enable_thinking":false}',
      "it's",
      "",
      "a\\b",
      "$HOME",
    ];
    expect(splitShellArgs(shellJoin(argv))).toEqual(argv);
  });
});
