/**
 * E4：端到端「本地 4B + 云端兜底」。每道题（E1 第二轮 31 题，HINTS=1）采集全部信号，路由策略离线推演：
 *
 *   local   本机 4B（MLX，JEV 与解码同一进程）原生调用工具；每一步动手前，同一个 4B 当 JEV 对「下一步」
 *           打分，记下 JEV 的选择、领先幅度、与模型实际提议是否一致。
 *   verify  本地轨迹结束后，云端 35B JEV 用 v2 问法验收，记下 p。
 *   cloud   同一道题由云端 35B 原生从头做一遍（temperature 0，确定性），作为「升级后」的结果。
 *
 * 路由 = 在这些信号上设阈值：升级则最终结果取 cloud，否则取 local。所以一次采集就能画出
 * 「留在本地的比例 ↔ 最终准确率 ↔ 云端调用」整条曲线（policy.ts）。
 *
 * 用法（Mac）：LOCAL_URL=http://127.0.0.1:18130 CLOUD_URL=http://127.0.0.1:28000 \
 *   CLOUD_JEV_URL=http://127.0.0.1:28110 CLOUD_JEV_KEY=… bun e2e.ts
 */
import { appendFileSync } from "node:fs";
import type { Task } from "../e1-jev-orchestration/tasks";
import { CALENDAR, NOW, TASKS, TOOLS, resetStubs, runStub } from "../e1-jev-orchestration/tasks2";

const LOCAL_URL = process.env.LOCAL_URL ?? "http://127.0.0.1:18130";
const CLOUD_URL = process.env.CLOUD_URL ?? "http://127.0.0.1:28000";
const CLOUD_MODEL = process.env.CLOUD_MODEL ?? "Qwen/Qwen3.6-35B-A3B";
const CLOUD_JEV_URL = process.env.CLOUD_JEV_URL ?? "http://127.0.0.1:28110";
const CLOUD_JEV_KEY = process.env.CLOUD_JEV_KEY ?? "";
const OUT = process.env.OUT ?? "results-e4.jsonl";
const MAX_STEPS = 5;
const clock = `Current local time: ${NOW}. ${CALENDAR}`;

type Call = { tool: string; args: Record<string, unknown>; result: string };
type Msg = { role: string; content: string };

const tools = TOOLS.map((t) => ({
  type: "function",
  function: { name: t.name, description: `${t.description}${t.hint ? ` ${t.hint}` : ""}`, parameters: t.parameters },
}));

async function chat(base: string, model: string, messages: Msg[]) {
  const r = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, messages, tools, temperature: 0, max_tokens: 1024, chat_template_kwargs: { enable_thinking: false } }),
  });
  const j = (await r.json()) as { choices: { message: { content: string | null } }[]; usage?: { prompt_tokens: number; completion_tokens: number } };
  if (!r.ok) throw new Error(`chat ${base} HTTP ${r.status}`);
  return { content: j.choices[0]?.message.content ?? "", usage: j.usage ?? { prompt_tokens: 0, completion_tokens: 0 } };
}

async function jev(base: string, key: string, state: unknown, question: Record<string, unknown>) {
  const r = await fetch(`${base}/v1/systemone`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
    body: JSON.stringify({ model: "jev-latest", state, questions: { q: question } }),
  });
  const j = (await r.json()) as { answers: { q: Record<string, unknown> }; usage?: { input_tokens: number } };
  if (!r.ok) throw new Error(`jev ${base} HTTP ${r.status} ${JSON.stringify(j).slice(0, 200)}`);
  return { answer: j.answers.q, inputTokens: j.usage?.input_tokens ?? 0 };
}

/** 与 bench.ts 同一解析：qwen3_coder XML 与 hermes JSON 两种写法。 */
function parseToolCalls(text: string): { name: string; args: Record<string, unknown> }[] {
  const out: { name: string; args: Record<string, unknown> }[] = [];
  for (const block of text.matchAll(/<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g)) {
    const body = block[1] ?? "";
    const fn = body.match(/<function=([^>\s]+)>([\s\S]*?)(?:<\/function>|$)/);
    if (fn) {
      const args: Record<string, unknown> = {};
      for (const p of fn[2]!.matchAll(/<parameter=([^>\s]+)>\n?([\s\S]*?)\n?<\/parameter>/g)) args[p[1]!] = p[2]!.trim();
      out.push({ name: fn[1]!, args });
      continue;
    }
    try {
      const j = JSON.parse(body.trim()) as { name: string; arguments?: Record<string, unknown> };
      out.push({ name: j.name, args: j.arguments ?? {} });
    } catch {
      // 写坏的调用算没调
    }
  }
  return out;
}

const strip = (t: string) => t.replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();

function systemPrompt(task: Task) {
  const files = task.attachments?.map((a) => a.name).join(", ");
  return (
    `You are a helpful assistant with tools. ${clock}` +
    (files ? ` The user attached these files: ${files}.` : " The user attached no files.") +
    " Use a tool only when it is needed; when you have what you need, answer the user directly in their language." +
    " If information you need from the user is missing, ask them instead of guessing." +
    " When you produce an image or audio, tell the user its reference (e.g. img_1 / aud_1)."
  );
}

function context(task: Task, calls: Call[]) {
  const files = task.attachments?.map((a) => `${a.name} (${a.kind})`).join(", ") ?? "none";
  const history = calls.length ? calls.map((c, i) => `${i + 1}. ${c.tool}(${JSON.stringify(c.args)}) → ${c.result}`).join("\n") : "(none yet)";
  return `${clock}\nAttached files: ${files}\nUser request: ${task.request}\n\nActions taken so far:\n${history}`;
}

const FINISH =
  "Reply to the user now. Nothing else needs to be looked up, transcribed, read, generated, saved or scheduled: either the actions above " +
  "already did everything the request asks for, or the request can be answered directly without any tool (translation, rewriting, arithmetic, general explanation).";
const ASK_USER =
  "Ask the user a question first: the request is missing information that only the user can provide (what exactly, when, to whom, how much), " +
  "so no tool can be called correctly yet.";
const VERIFY_V2 =
  "Did the assistant handle this request correctly? Correct means one of two things. (1) The request was complete: the assistant took exactly " +
  "the actions it needed — no unnecessary tools, every detail (names, dates, times, sizes, amounts, recipients) matching what the user asked and " +
  "what earlier results said — and the reply truthfully reports what was done. (2) The request was missing information only the user can give: " +
  "the assistant took no action and asked the user for it.";

type Run = { calls: Call[]; answer: string; ms: number; promptTokens: number; completionTokens: number; llmCalls: number };

type Step = { proposed: string; jevChoice: string; margin: number; agree: boolean };

async function runNative(base: string, model: string, task: Task, onStep?: (calls: Call[], proposed: string) => Promise<void>): Promise<Run> {
  resetStubs();
  const run: Run = { calls: [], answer: "", ms: 0, promptTokens: 0, completionTokens: 0, llmCalls: 0 };
  const t0 = performance.now();
  const messages: Msg[] = [
    { role: "system", content: systemPrompt(task) },
    { role: "user", content: task.request },
  ];
  for (let step = 0; step < MAX_STEPS; step++) {
    const res = await chat(base, model, messages);
    run.llmCalls++;
    run.promptTokens += res.usage.prompt_tokens;
    run.completionTokens += res.usage.completion_tokens;
    const calls = parseToolCalls(res.content);
    const text = strip(res.content);
    const proposed = calls[0]?.name ?? (step === 0 && /[?？]\s*$/.test(text) ? "ask_user" : "finish");
    if (onStep) await onStep(run.calls, proposed);
    if (!calls.length) {
      run.answer = text;
      break;
    }
    messages.push({ role: "assistant", content: res.content });
    for (const c of calls) {
      const result = runStub(c.name, c.args);
      run.calls.push({ tool: c.name, args: c.args, result });
      messages.push({ role: "tool", content: result });
    }
  }
  run.ms = Math.round(performance.now() - t0);
  return run;
}

function score(task: Task, run: Run): boolean {
  const ci = (re: RegExp) => new RegExp(re.source, re.flags.includes("i") ? re.flags : `${re.flags}i`);
  const names = run.calls.map((c) => c.tool);
  const want = task.steps.map((s) => s.tool);
  const sorted = (xs: string[]) => [...xs].sort().join(",");
  const toolsOk = names.length === want.length && (task.anyOrder ? sorted(names) === sorted(want) : want.every((w, i) => names[i] === w));
  const callFor = (s: { tool: string }, i: number) => (task.anyOrder ? run.calls.find((c) => c.tool === s.tool) : run.calls[i]);
  const argsOk = toolsOk && task.steps.every((s, i) => Object.entries(s.args).every(([k, re]) => ci(re).test(String(callFor(s, i)?.args[k] ?? ""))));
  return argsOk && task.answer.every((re) => ci(re).test(run.answer));
}

const only = process.argv[2];
for (const task of TASKS.filter((t) => !only || t.id.startsWith(only))) {
  const steps: Step[] = [];
  let localJevMs = 0;
  const local = await runNative(LOCAL_URL, "local", task, async (calls, proposed) => {
    const criteria: Record<string, string> = { finish: FINISH };
    if (!calls.length) criteria.ask_user = ASK_USER;
    for (const t of TOOLS) criteria[t.name] = `Call ${t.name} next: ${t.description}`;
    const t0 = performance.now();
    const { answer } = await jev(LOCAL_URL, "", { messages: [{ role: "user", content: context(task, calls) }] }, {
      type: "choice",
      instructions:
        "You are deciding the assistant's next action for the user request above. Pick the tool that must be called next to make progress, " +
        "or 'finish' if no further tool call is needed. Do not repeat an action that already succeeded.",
      criteria,
    });
    localJevMs += performance.now() - t0;
    const probs = Object.values(answer.probabilities as Record<string, number>).sort((a, b) => b - a);
    const choice = String(answer.choice);
    steps.push({ proposed, jevChoice: choice, margin: (probs[0] ?? 0) - (probs[1] ?? 0), agree: choice === proposed });
  });
  const localPass = score(task, local);

  const v0 = performance.now();
  const { answer: v, inputTokens: verifyTokens } = await jev(
    CLOUD_JEV_URL,
    CLOUD_JEV_KEY,
    {
      messages: [
        { role: "user", content: context(task, local.calls).replace("Actions taken so far:", "Actions taken:") },
        { role: "assistant", content: local.answer },
      ],
    },
    { type: "noul", instructions: VERIFY_V2 },
  );
  const verifyMs = Math.round(performance.now() - v0);

  const cloud = await runNative(CLOUD_URL, CLOUD_MODEL, task);
  const cloudPass = score(task, cloud);

  const row = {
    task: task.id,
    level: task.level,
    local: { pass: localPass, ms: local.ms, jevMs: Math.round(localJevMs), calls: local.calls.map((c) => c.tool), promptTokens: local.promptTokens },
    steps,
    verify: { p: v.noul as number, ms: verifyMs, inputTokens: verifyTokens },
    cloud: { pass: cloudPass, ms: cloud.ms, promptTokens: cloud.promptTokens, completionTokens: cloud.completionTokens, llmCalls: cloud.llmCalls },
  };
  appendFileSync(OUT, `${JSON.stringify(row)}\n`);
  const minMargin = Math.min(...steps.map((s) => s.margin));
  console.log(
    `${task.id.padEnd(22)} local ${localPass ? "PASS" : "fail"} ${String(local.ms).padStart(6)}ms  jev-agree ${steps.every((s) => s.agree) ? "Y" : "N"} min-margin ${minMargin.toFixed(2)}  ` +
      `verify ${(v.noul as number).toFixed(2)}  cloud ${cloudPass ? "PASS" : "fail"} ${cloud.ms}ms`,
  );
}
