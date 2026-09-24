/**
 * E6 预置：给一个**独立数据目录**写好设置与数据，再用它启动真实 App（不碰用户自己的 dev / canary / stable 数据）。
 *
 * - 设置：聊天走「远程」模式指向本机 llama-server（候选 Qwen3.5-4B Q4_K_M），JEV 走 cloud 后端指向
 *   llama_jev_server（同一个 llama-server 的 JEV 接口）；工具策略由 STRATEGY 决定（classic / routed）。
 * - 数据：一条笔记（向量数据库选型）、一条记忆（喜欢爵士乐 / 花生过敏）、一张生图记录（橘猫宇航员），
 *   以及工作区里的 notes/todo.md、docs/plan.md、finance/budget-2026.xlsx —— 对应 E5 任务里要查的东西。
 *   知识库没有预置（导入要配向量模型）：知识库类任务只检查工具与查询词。
 *
 * 用法：OMNI_DATA_DIR=<目录> STRATEGY=routed WORKSPACE=<目录> bun seed.ts
 */
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { db } from "../../../apps/studio/src/bun/db";
import { imageRecords } from "../../../apps/studio/src/bun/db/schema";
import { updateSettings } from "../../../apps/studio/src/bun/db/settings";
import { saveMemory } from "../../../apps/studio/src/bun/memory";
import { saveNote } from "../../../apps/studio/src/bun/notes";

const dataDir = process.env.OMNI_DATA_DIR;
const workspace = process.env.WORKSPACE;
if (!dataDir || !workspace) throw new Error("need OMNI_DATA_DIR and WORKSPACE");
const strategy = process.env.STRATEGY === "routed" ? "routed" : "classic";

updateSettings({
  SERVER_MODE: "remote",
  VLLM_API_BASE: process.env.CHAT_BASE ?? "http://127.0.0.1:18131/v1",
  VLLM_API_KEY: "EMPTY",
  VLLM_MODEL_NAME: "qwen3.5-4b",
  CHAT_MODEL: "qwen3.5-4b",
  AGENT_TOOL_STRATEGY: strategy,
  AGENT_THINKING_LEVEL: "off",
  SYSTEMONE_BACKEND: "cloud",
  SYSTEMONE_CLOUD_BASE_URL: process.env.JEV_BASE ?? "http://127.0.0.1:18133",
  SYSTEMONE_CLOUD_API_KEY: "local-jev",
  MEMORY_ENABLED: "1",
});

const already = db.select().from(imageRecords).all().length > 0;
if (!already) {
  saveNote({ title: "向量数据库选型", body: "对比 Milvus / Qdrant / pgvector：部署复杂度、过滤性能、运维成本。\n\n结论：选 Qdrant（部署简单、过滤性能好）。" });
  saveMemory({ content: "用户喜欢爵士乐和猫；对花生过敏。", category: "preference", source: "manual" });
  // 1×1 PNG 就够：media_search 查的是记录里的提示词，不看像素
  const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "base64");
  mkdirSync(path.join(dataDir, "images", "gen"), { recursive: true });
  writeFileSync(path.join(dataDir, "images", "gen", "cat-astronaut.png"), png);
  db.insert(imageRecords)
    .values({ status: "done", source: "manual", prompt: "戴着宇航员头盔的橘猫", imagePath: "gen/cat-astronaut.png", width: 1024, height: 1024, model: "z-image-turbo" })
    .run();
}

const files: Record<string, string> = {
  "notes/todo.md": "- [x] 提交周报\n- [ ] 报销差旅发票\n- [ ] 预约牙医\n- [x] 续费域名\n",
  "docs/plan.md": "# 上线计划\n\n- 功能冻结：9 月 30 日\n- 上线日期：10 月 8 日\n- 负责人：张伟\n",
  "finance/budget-2026.xlsx": "placeholder spreadsheet",
};
for (const [rel, content] of Object.entries(files)) {
  mkdirSync(path.dirname(path.join(workspace, rel)), { recursive: true });
  writeFileSync(path.join(workspace, rel), content);
}
console.log(`seeded ${dataDir} (strategy=${strategy})`);
