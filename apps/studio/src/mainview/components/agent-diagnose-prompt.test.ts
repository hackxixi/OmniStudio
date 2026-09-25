/**
 * 「让 Agent 解决」带过去的那一段话的格式 —— 这就是这个按钮的全部价值：
 * 带齐了，Agent 第一轮就能动手；带漏了，它只能先反问。
 */
import { expect, test } from "bun:test";

import { buildDiagnosisPrompt } from "./agent-diagnose-prompt";

test("说明 → 报错 → 环境 → 日志末尾，按这个顺序", () => {
  const prompt = buildDiagnosisPrompt({
    intro: "本地模型起不来，请诊断并修好。",
    error: "E gguf_init_from_reader: failed to read magic",
    context: ["引擎：llama.cpp", "模型：/models/Qwopus3.5-4B-Coder-MTP-GGUF"],
    logs: ["I load_model: loading model '/models/Qwopus3.5-4B-Coder-MTP-GGUF'", "E llama_server: exiting due to model loading error"],
  });
  const at = (s: string) => prompt.indexOf(s);
  expect(at("本地模型起不来")).toBe(0);
  expect(at("报错：E gguf_init_from_reader")).toBeGreaterThan(0);
  expect(at("引擎：llama.cpp")).toBeGreaterThan(at("报错："));
  expect(at("日志（末尾 2 行）")).toBeGreaterThan(at("模型："));
  expect(prompt).toContain("exiting due to model loading error");
});

test("日志只带末尾 40 行，空行不算", () => {
  const logs = Array.from({ length: 100 }, (_, i) => `line ${i + 1}`).concat(["", "   "]);
  const prompt = buildDiagnosisPrompt({ intro: "x", error: "y", logs });
  expect(prompt).toContain("日志（末尾 40 行）");
  expect(prompt).toContain("line 100");
  expect(prompt).toContain("line 61");
  expect(prompt).not.toContain("line 60\n");
});

test("没有日志就不出现日志段（但行动建议还在，它只是引用了上面的日志）", () => {
  const prompt = buildDiagnosisPrompt({ intro: "x", error: "y" });
  expect(prompt).not.toContain("日志（末尾");
  // 行动建议末尾依然带一句"先读上面给出的日志……"—— 它指的是调用方传入的现场，
  // 没有现场时这句话变成"读报错"（下面的用例单独验证它始终存在）。
});

test("日志里的 ANSI 颜色码与控制字符剥干净（真机第一次点时满屏 ⌧[34m）", () => {
  const prompt = buildDiagnosisPrompt({
    intro: "x",
    error: "y",
    logs: [
      "\x1b[0m\x1b[34m0.00.065.576\x1b[0m \x1b[32mI \x1b[0msrv  operator(): cleaning up before exit...\r",
      "\x1b[34m0.00.065.874\x1b[0m \x1b[31mE srv  llama_server: exiting due to model loading error",
      "\x1b[0m",
    ],
  });
  expect(prompt).not.toContain("\x1b");
  expect(prompt).not.toContain("\r");
  expect(prompt).toContain("0.00.065.874 E srv  llama_server: exiting due to model loading error");
  // 只剩颜色码的那一行清完就是空行，不该占一行。
  expect(prompt).toContain("日志（末尾 2 行）");
});

test("结尾始终带一句行动建议：先读已有日志、工作区外用 bash、别大范围搜源码", () => {
  // 这句是省步数的关键（真机上它先花十几步 glob/grep 找日志在哪，还被工作区权限拒了）。
  // 无论带不带现场、带不带日志，它都要在 —— 调用方忘了传路径时也至少能防住"搜源码"。
  const advice =
    "先读上面给出的日志与报错，工作区外的文件用 bash（cat / grep / sed）读取；判断清楚后直接给出结论与修复步骤，不要大范围搜索源码。";
  const cases = [
    buildDiagnosisPrompt({ intro: "x", error: "y" }),
    buildDiagnosisPrompt({ intro: "x", error: "y", context: ["平台：darwin arm64"] }),
    buildDiagnosisPrompt({ intro: "x", error: "y", logs: ["a", "b"] }),
  ];
  for (const prompt of cases) {
    expect(prompt.endsWith(advice)).toBe(true);
  }
});
