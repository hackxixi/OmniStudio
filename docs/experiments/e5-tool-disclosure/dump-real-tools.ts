/**
 * 把 Agent 真实装配的内置工具定义（名称 / 描述 / 参数）存成快照 real-tools.json，
 * 让 E5 在仓库代码变化后仍可复现。MCP 工具与 view_image 不在内（前者取决于用户配置，后者取决于模型是否支持视觉）。
 *
 * 用法（仓库根目录）：OMNI_DATA_DIR=$(mktemp -d) bun docs/experiments/e5-tool-disclosure/dump-real-tools.ts
 */
import { buildAgentTools } from "../../../apps/studio/src/bun/agent-tools";
import { buildMediaGenTools, buildMediaReadTools } from "../../../apps/studio/src/bun/media-tools";
import { buildMemoryAgentTools } from "../../../apps/studio/src/bun/memory";
import { buildNotesAgentTools } from "../../../apps/studio/src/bun/notes-tools";
import { buildSystemOneAgentTools } from "../../../apps/studio/src/bun/systemone-tools";

// 构造工具只读取 ctx 上少数字段；其余返回 undefined 即可
const ctx = new Proxy({ workspace: "/tmp/ws", sessionId: "e5", mode: "agent" } as Record<string, unknown>, {
  get: (t, k) => (typeof k === "string" && k in t ? t[k] : undefined),
}) as never;
type Built = { name: string; description: string; parameters: unknown };
const groups: Record<string, Built[]> = {
  core: buildAgentTools(ctx) as Built[],
  mediaRead: buildMediaReadTools() as Built[],
  notes: buildNotesAgentTools() as Built[],
  jev: buildSystemOneAgentTools() as Built[],
  mediaGen: buildMediaGenTools(ctx) as Built[],
  memory: buildMemoryAgentTools({ scope: "/tmp/ws" }) as Built[],
};
const out = Object.entries(groups).flatMap(([group, tools]) =>
  tools.map((t) => ({ group, name: t.name, description: t.description, parameters: t.parameters })),
);
await Bun.write(new URL("./real-tools.json", import.meta.url), `${JSON.stringify(out, null, 1)}\n`);
console.log(`${out.length} tools`);
