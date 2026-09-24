/**
 * 事后给每一组的每条运行都过一遍 JEV 完成度判定（B 组在运行时已判过，这里统一重判），
 * 看「JEV 当验收员」能否兜住所有编排方式的错误，与编排方式无关。
 *
 * 用法：bun verify.ts results.jsonl
 */
import { readFileSync } from "node:fs";
import { NOW, TASKS } from "./tasks";

const JEV_URL = process.env.JEV_URL ?? "http://127.0.0.1:18110";
const JEV_KEY = readFileSync(process.env.JEV_KEY_FILE ?? `${process.env.HOME}/jev/openjev.key`, "utf8").trim();

type Row = {
  arm: string;
  task: string;
  pass: boolean;
  answer: string;
  calls: { tool: string; args: unknown; result: string }[];
};

const rows = readFileSync(process.argv[2] ?? "results.jsonl", "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l) as Row);

const out: { arm: string; task: string; pass: boolean; p: number }[] = [];
for (const r of rows) {
  const task = TASKS.find((t) => t.id === r.task)!;
  const files = task.attachments?.map((a) => `${a.name} (${a.kind})`).join(", ") ?? "none";
  const history = r.calls.length
    ? r.calls.map((c, i) => `${i + 1}. ${c.tool}(${JSON.stringify(c.args)}) → ${c.result}`).join("\n")
    : "(none)";
  const res = await fetch(`${JEV_URL}/v1/systemone`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${JEV_KEY}` },
    body: JSON.stringify({
      model: "jev-latest",
      state: {
        messages: [
          {
            role: "user",
            content: `Current local time: ${NOW}\nAttached files: ${files}\nUser request: ${task.request}\n\nActions taken:\n${history}`,
          },
          { role: "assistant", content: r.answer },
        ],
      },
      questions: {
        done: {
          type: "noul",
          instructions:
            "Did the assistant fully accomplish the user's request — the right actions with the right details, and a reply that " +
            "correctly reports what was actually done (no claims about actions that were never taken)?",
        },
      },
    }),
  });
  const j = (await res.json()) as { answers: { done: { noul: number } } };
  out.push({ arm: r.arm, task: r.task, pass: r.pass, p: j.answers.done.noul });
}

for (const threshold of [0.8, 0.85, 0.9]) {
  const failures = out.filter((o) => !o.pass);
  const caught = failures.filter((o) => o.p < threshold).length;
  const falseAlarms = out.filter((o) => o.pass && o.p < threshold).length;
  console.log(
    `threshold ${threshold}: caught ${caught}/${failures.length} failures, false alarms ${falseAlarms}/${out.length - failures.length} passes`,
  );
}
for (const o of out.filter((x) => !x.pass || x.p < 0.9)) console.log(`${o.arm.padEnd(3)} ${o.task.padEnd(22)} ${o.pass ? "PASS" : "fail"} p=${o.p.toFixed(3)}`);
