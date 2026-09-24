/**
 * E1：JEV 编排 + 小模型填参数，对比小模型 / 中等模型直接跑 Agent。
 *
 *   A   小模型原生工具调用（模型自己决定调哪个工具、何时结束）
 *   B   JEV 选下一步 + 判完成，小模型按 JSON Schema 约束解码填参数、写最终回答
 *   B0  同 B，但不按附件过滤工具（量出「只给可行选项」的贡献）
 *   C   中等模型原生工具调用（参照）
 *
 * 用法（在 ycs2 上）：bun bench.ts [A,B,B0,C] [任务 id 前缀]
 * 环境变量：SMALL_URL / SMALL_MODEL / MID_URL / MID_MODEL / JEV_URL / JEV_KEY_FILE / OUT
 */
import { appendFileSync, readFileSync } from "node:fs";
import { NOW, TASKS, TOOLS, resetStubs, runStub, type Task, type ToolSpec } from "./tasks";

const SMALL_URL = process.env.SMALL_URL ?? "http://127.0.0.1:30001";
const SMALL_MODEL = process.env.SMALL_MODEL ?? "Qwen/Qwen3.5-4B";
const MID_URL = process.env.MID_URL ?? "http://127.0.0.1:30000";
const MID_MODEL = process.env.MID_MODEL ?? "Qwen/Qwen3.6-35B-A3B";
const JEV_URL = process.env.JEV_URL ?? "http://127.0.0.1:18110";
const JEV_KEY = readFileSync(process.env.JEV_KEY_FILE ?? `${process.env.HOME}/jev/openjev.key`, "utf8").trim();
const OUT = process.env.OUT ?? "results.jsonl";
const MAX_STEPS = 5;

type Call = { tool: string; args: Record<string, unknown>; result: string };
type Decision = { choice: string; top: number; margin: number; options: number };
type Run = {
  arm: string;
  task: string;
  level: string;
  calls: Call[];
  answer: string;
  decisions: Decision[];
  verifier?: number;
  ms: number;
  llmCalls: number;
  jevCalls: number;
  error?: string;
};

// ---------------------------------------------------------------------------
// 模型调用
// ---------------------------------------------------------------------------

type Msg = { role: "system" | "user" | "assistant" | "tool"; content: string };

async function chat(
  base: string,
  model: string,
  messages: Msg[],
  extra: Record<string, unknown> = {},
): Promise<{ content: string; toolCalls: { name: string; arguments: string }[] }> {
  const r = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      temperature: 0,
      max_tokens: 1024,
      chat_template_kwargs: { enable_thinking: false },
      ...extra,
    }),
  });
  const j = (await r.json().catch(() => ({}))) as {
    choices?: { message?: { content?: string | null; tool_calls?: { function: { name: string; arguments: string } }[] } }[];
  };
  if (!r.ok) throw new Error(`chat HTTP ${r.status} ${JSON.stringify(j).slice(0, 300)}`);
  const m = j.choices?.[0]?.message;
  return {
    content: m?.content ?? "",
    toolCalls: (m?.tool_calls ?? []).map((t) => ({ name: t.function.name, arguments: t.function.arguments })),
  };
}

async function jev(state: unknown, questions: Record<string, unknown>) {
  const r = await fetch(`${JEV_URL}/v1/systemone`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${JEV_KEY}` },
    body: JSON.stringify({ state, model: "jev-latest", questions }),
  });
  const j = (await r.json().catch(() => ({}))) as { answers?: Record<string, Record<string, unknown>> };
  if (!r.ok) throw new Error(`jev HTTP ${r.status} ${JSON.stringify(j).slice(0, 300)}`);
  return j.answers ?? {};
}

// ---------------------------------------------------------------------------
// 原生工具调用（A / C）
// ---------------------------------------------------------------------------

/** 服务端没开工具解析器时，工具调用留在正文里：认 qwen3_coder 的 XML 与 hermes 的 JSON 两种写法。 */
export function parseToolCalls(text: string): { name: string; args: Record<string, unknown> }[] {
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
      const j = JSON.parse(body.trim()) as { name: string; arguments?: Record<string, unknown> | string };
      const args = typeof j.arguments === "string" ? JSON.parse(j.arguments) : (j.arguments ?? {});
      out.push({ name: j.name, args });
    } catch {
      // 写坏的调用：当作没调用（计入失败）
    }
  }
  return out;
}

function stripToolCalls(text: string): string {
  return text.replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

function systemPrompt(task: Task): string {
  const files = task.attachments?.map((a) => a.name).join(", ");
  return (
    `You are a helpful assistant with tools. Current local time: ${NOW}.` +
    (files ? ` The user attached these files: ${files}.` : " The user attached no files.") +
    " Use a tool only when it is needed; when you have what you need, answer the user directly in their language." +
    " When you produce an image or audio, tell the user its reference (e.g. img_1 / aud_1)."
  );
}

const openAiTools = TOOLS.map((t) => ({
  type: "function",
  function: { name: t.name, description: t.description, parameters: t.parameters },
}));

async function runNative(arm: string, base: string, model: string, task: Task): Promise<Run> {
  const run: Run = { arm, task: task.id, level: task.level, calls: [], answer: "", decisions: [], ms: 0, llmCalls: 0, jevCalls: 0 };
  const messages: Msg[] = [
    { role: "system", content: systemPrompt(task) },
    { role: "user", content: task.request },
  ];
  for (let step = 0; step < MAX_STEPS; step++) {
    const res = await chat(base, model, messages, { tools: openAiTools });
    run.llmCalls++;
    const calls = res.toolCalls.length
      ? res.toolCalls.map((c) => ({ name: c.name, args: safeJson(c.arguments) }))
      : parseToolCalls(res.content);
    if (!calls.length) {
      run.answer = stripToolCalls(res.content);
      return run;
    }
    messages.push({ role: "assistant", content: res.content });
    for (const c of calls) {
      const result = runStub(c.name, c.args);
      run.calls.push({ tool: c.name, args: c.args, result });
      messages.push({ role: "tool", content: result });
    }
  }
  run.error = "max steps";
  return run;
}

function safeJson(s: string): Record<string, unknown> {
  try {
    return JSON.parse(s) as Record<string, unknown>;
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// JEV 编排（B / B0）
// ---------------------------------------------------------------------------

function context(task: Task, calls: Call[]): string {
  const files = task.attachments?.map((a) => `${a.name} (${a.kind})`).join(", ") ?? "none";
  const history = calls.length
    ? calls.map((c, i) => `${i + 1}. ${c.tool}(${JSON.stringify(c.args)}) → ${c.result}`).join("\n")
    : "(none yet)";
  return `Current local time: ${NOW}\nAttached files: ${files}\nUser request: ${task.request}\n\nActions taken so far:\n${history}`;
}

const FINISH =
  "Reply to the user now. Nothing else needs to be looked up, transcribed, read, generated, saved or scheduled: " +
  "either the actions above already did everything the request asks for, or the request can be answered directly " +
  "without any tool (translation, rewriting, arithmetic, general explanation).";

function feasibleTools(task: Task, filter: boolean): ToolSpec[] {
  if (!filter) return TOOLS;
  const kinds = new Set(task.attachments?.map((a) => a.kind));
  return TOOLS.filter((t) => !t.needs || kinds.has(t.needs));
}

async function runJev(arm: string, task: Task, filter: boolean): Promise<Run> {
  const run: Run = { arm, task: task.id, level: task.level, calls: [], answer: "", decisions: [], ms: 0, llmCalls: 0, jevCalls: 0 };
  for (let step = 0; step < MAX_STEPS; step++) {
    const tools = feasibleTools(task, filter);
    const criteria: Record<string, string> = { finish: FINISH };
    for (const t of tools) criteria[t.name] = `Call ${t.name} next: ${t.description}`;
    const answers = await jev(
      { messages: [{ role: "user", content: context(task, run.calls) }] },
      {
        next: {
          type: "choice",
          instructions:
            "You are deciding the assistant's next action for the user request above. Pick the tool that must be " +
            "called next to make progress, or 'finish' if no further tool call is needed. Do not repeat an action " +
            "that already succeeded.",
          criteria,
        },
      },
    );
    run.jevCalls++;
    const a = answers.next as { choice: string; probabilities: Record<string, number> };
    const probs = Object.values(a.probabilities).sort((x, y) => y - x);
    run.decisions.push({ choice: a.choice, top: probs[0] ?? 0, margin: (probs[0] ?? 0) - (probs[1] ?? 0), options: probs.length });
    if (a.choice === "finish") break;
    const tool = TOOLS.find((t) => t.name === a.choice)!;
    const filled = await chat(
      SMALL_URL,
      SMALL_MODEL,
      [
        { role: "system", content: "You fill in the arguments of one tool call. Reply with the JSON arguments only." },
        {
          role: "user",
          content:
            `${context(task, run.calls)}\n\nThe next step is to call \`${tool.name}\` (${tool.description}). ` +
            "Fill in its arguments so that this call does what the user asked. Copy exact text, names and file names from the request; " +
            "write times as ISO 8601 local time.",
        },
      ],
      { response_format: { type: "json_schema", json_schema: { name: tool.name, schema: tool.parameters, strict: true } } },
    );
    run.llmCalls++;
    const args = safeJson(filled.content);
    run.calls.push({ tool: tool.name, args, result: runStub(tool.name, args) });
    if (step === MAX_STEPS - 1) run.error = "max steps";
  }
  const final = await chat(SMALL_URL, SMALL_MODEL, [
    { role: "system", content: `You are a helpful assistant. Current local time: ${NOW}.` },
    {
      role: "user",
      content:
        `${context(task, run.calls)}\n\nAll needed actions are done. Now write the reply to the user in the language of their request, ` +
        "using the results above. If an image or audio was produced, give its reference (e.g. img_1 / aud_1). " +
        "If there were no actions, just answer the request yourself.",
    },
  ]);
  run.llmCalls++;
  run.answer = stripToolCalls(final.content);
  // 完成度判定：与真值对照，看 JEV 当验收员靠不靠得住
  const verdict = await jev(
    {
      messages: [
        { role: "user", content: context(task, run.calls) },
        { role: "assistant", content: run.answer },
      ],
    },
    {
      done: {
        type: "noul",
        instructions:
          "Did the assistant fully accomplish the user's request — the right actions with the right details, and a reply that " +
          "correctly reports the result?",
      },
    },
  );
  run.jevCalls++;
  run.verifier = (verdict.done as { noul: number } | undefined)?.noul;
  return run;
}

// ---------------------------------------------------------------------------
// 打分
// ---------------------------------------------------------------------------

export function score(task: Task, run: Run) {
  const ci = (re: RegExp) => new RegExp(re.source, re.flags.includes("i") ? re.flags : `${re.flags}i`);
  const names = run.calls.map((c) => c.tool);
  const want = task.steps.map((s) => s.tool);
  const toolsOk = names.length === want.length && want.every((w, i) => names[i] === w);
  const argsOk =
    toolsOk &&
    task.steps.every((s, i) => Object.entries(s.args).every(([k, re]) => ci(re).test(String(run.calls[i]!.args[k] ?? ""))));
  const answerOk = task.answer.every((re) => ci(re).test(run.answer));
  return { toolsOk, argsOk, answerOk, pass: toolsOk && argsOk && answerOk && !run.error };
}

// ---------------------------------------------------------------------------

const arms = (process.argv[2] ?? "A,B,B0,C").split(",");
const only = process.argv[3];
const tasks = TASKS.filter((t) => !only || t.id.startsWith(only));

for (const arm of arms) {
  const rows: ReturnType<typeof score>[] = [];
  let ms = 0;
  for (const task of tasks) {
    resetStubs();
    const t0 = performance.now();
    let run: Run;
    try {
      run =
        arm === "A"
          ? await runNative(arm, SMALL_URL, SMALL_MODEL, task)
          : arm === "C"
            ? await runNative(arm, MID_URL, MID_MODEL, task)
            : await runJev(arm, task, arm === "B");
    } catch (e) {
      run = { arm, task: task.id, level: task.level, calls: [], answer: "", decisions: [], ms: 0, llmCalls: 0, jevCalls: 0, error: String(e) };
    }
    run.ms = Math.round(performance.now() - t0);
    ms += run.ms;
    const s = score(task, run);
    rows.push(s);
    appendFileSync(OUT, `${JSON.stringify({ ...run, ...s })}\n`);
    console.log(
      `${arm.padEnd(3)} ${task.id.padEnd(22)} ${s.pass ? "PASS" : "fail"}  tools=${+s.toolsOk} args=${+s.argsOk} answer=${+s.answerOk}` +
        `  [${run.calls.map((c) => c.tool).join(" → ")}]${run.error ? ` ERR ${run.error.slice(0, 80)}` : ""}  ${run.ms}ms`,
    );
  }
  const pct = (k: keyof ReturnType<typeof score>) => `${Math.round((rows.filter((r) => r[k]).length / rows.length) * 100)}%`;
  console.log(
    `== ${arm}: pass ${pct("pass")}  tools ${pct("toolsOk")}  args ${pct("argsOk")}  answer ${pct("answerOk")}  avg ${Math.round(ms / rows.length)}ms\n`,
  );
}
