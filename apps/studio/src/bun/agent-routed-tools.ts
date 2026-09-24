import { Type } from "typebox";

import { errorResult, textResult, type BuiltTool } from "./agent-tools";

/**
 * 「精简路由」工具策略（背景：docs/experiments/e5-tool-disclosure/README.md）。
 *
 * Agent 目前把全部工具一次交给模型：本地小窗口下，工具定义本身就要吃掉几千 token
 * 的前缀，每次未命中缓存都要重读。实验结论是**常驻少量核心工具 + 其余按组按需加载**
 * （开局由外部选组，模型自己调 `load_tools` 兜底）。本模块只做「工具集」这一层：
 * 把完整工具列表切成核心 / 分组，合成 recall / remember / find 三个工具，
 * 以及负责按需加载的元工具 `load_tools`。接入 agent.ts 与设置开关由别处完成。
 *
 * 组内顺序、组之间顺序都是**固定的**：同一组组合出来的工具列表前缀逐字节一致，
 * 推理引擎才能命中前缀缓存（E5 里"前缀快照拍在结构边界"的落地）。
 */

export type RoutedGroupId = "creation" | "dev" | "mcp";

/** 常驻核心工具（顺序即发给模型的顺序；输入里没有的跳过）。recall / remember / find 是本模块合成的新工具。 */
export const ROUTED_CORE_TOOL_NAMES = [
  "web_search",
  "web_fetch",
  "recall",
  "remember",
  "read_file",
  "write_file",
  "edit_file",
  "find",
  "note_read",
  "view_image",
  "ask_user",
  "read_skill",
] as const;

/** 精简策略下不给模型的工具（被合成工具取代的，或短任务用不上的流程控制类）。 */
export const ROUTED_DROPPED_TOOL_NAMES = [
  "knowledge_search",
  "note_search",
  "note_list",
  "memory_search",
  "memory_save",
  "memory_forget",
  "glob",
  "grep",
  "list_dir",
  "think",
  "checkpoint",
  "rewind",
  "todo_write",
  "get_context_remaining",
  "request_permissions",
  "goal",
  "write_plan",
  "task",
  "jev_evaluate",
] as const;

/** 组的固定顺序（拼工具列表时按这个顺序，保证同一组合的前缀逐字节一致，便于推理引擎的前缀缓存）。 */
export const ROUTED_GROUP_ORDER: readonly RoutedGroupId[] = ["creation", "dev", "mcp"];

/**
 * 每组一行英文说明（会进系统提示与 load_tools 描述）；
 * toolNames 为 "rest" 表示「其余所有未归类工具」（MCP 等）。
 */
export const ROUTED_GROUPS: Record<
  RoutedGroupId,
  { summary: string; toolNames: readonly string[] | "rest" }
> = {
  creation: {
    summary:
      "creation — generate images, speech (text-to-speech) and videos; find and reuse media generated earlier",
    toolNames: ["generate_image", "generate_speech", "generate_video", "media_search", "media_export"],
  },
  dev: {
    summary: "dev — run shell commands and apply multi-file code patches",
    toolNames: ["bash", "apply_patch"],
  },
  mcp: {
    summary: "connected apps — tools from connected MCP servers",
    toolNames: "rest",
  },
};

export type RoutedToolset = {
  core: BuiltTool[]; // 按 ROUTED_CORE_TOOL_NAMES 顺序
  groups: Partial<Record<RoutedGroupId, BuiltTool[]>>; // 空组不出现
};

/** 从工具结果里取全部文本（content 是文本/图片的数组）。 */
function resultText(result: { content: { type: string; text?: string }[] }): string {
  return result.content
    .filter((c): c is { type: "text"; text: string } => c.type === "text" && typeof c.text === "string")
    .map((c) => c.text)
    .join("\n");
}

/**
 * 合成：同时查知识库 / 笔记 / 记忆。parts 里缺哪个就不查哪个；三个都缺返回 null。
 *
 * 实验动机：三个近义检索工具在模型眼里是"选哪个"的难题；合并成一个之后，
 * 「用户之前告诉过我的东西」只有一种问法。
 */
export function createRecallTool(parts: {
  knowledge?: BuiltTool;
  notes?: BuiltTool;
  memory?: BuiltTool;
}): BuiltTool | null {
  const sources: Array<{ label: string; tool?: BuiltTool }> = [
    { label: "knowledge_base", tool: parts.knowledge },
    { label: "notes", tool: parts.notes },
    { label: "memory", tool: parts.memory },
  ];
  const present = sources.filter((s) => s.tool);
  if (present.length === 0) return null;
  return {
    name: "recall",
    label: "Recall",
    description:
      "Search what the user already has: the knowledge base (imported documents), the user's own notes, " +
      "and facts the user told you before. Results are labeled by source.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query — natural language is fine." }),
      source: Type.Optional(
        Type.Union([Type.Literal("any"), Type.Literal("knowledge_base"), Type.Literal("notes"), Type.Literal("memory")]),
      ),
    }),
    execute: async (_toolCallId, params: { query: string; source?: "any" | "knowledge_base" | "notes" | "memory" }, signal) => {
      const want = params.source ?? "any";
      const sections: string[] = [];
      for (const source of present) {
        if (want !== "any" && source.label !== want) continue;
        const tool = source.tool!;
        try {
          const result = await tool.execute(_toolCallId, { query: params.query }, signal);
          const text = resultText(result).trim();
          // 出错 / 空结果不影响其他来源：失败只写一行说明。
          if ("isError" in result && (result as { isError?: boolean }).isError) {
            sections.push(`[${source.label}] unavailable: ${text || "error"}`);
            continue;
          }
          // 空结果不占段：全部为空时统一回一句“什么都没查到”。
          if (text) sections.push(`[${source.label}]\n${text}`);
        } catch (e) {
          sections.push(`[${source.label}] unavailable: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
      return textResult(
        sections.length > 0 ? sections.join("\n\n") : "Nothing found in the knowledge base, notes or memory.",
      );
    },
  };
}

/** 合成：保存一条关于用户的长期记忆，委托给 memory_save。缺 memory_save 返回 null。 */
export function createRememberTool(memorySave?: BuiltTool): BuiltTool | null {
  if (!memorySave) return null;
  return {
    name: "remember",
    label: "Remember",
    description:
      "Save a lasting fact about the user (a preference, a personal detail) for future conversations. One sentence.",
    parameters: Type.Object({
      content: Type.String({ description: "The fact to remember, in one sentence." }),
    }),
    execute: async (_toolCallId, params: { content: string }, signal?) => {
      return memorySave.execute(_toolCallId, { content: params.content }, signal);
    },
  };
}

/** 合成：按文件名模式或文件内文字查找工作区文件，委托给 glob / grep。两个都缺返回 null。 */
export function createFindTool(parts: { glob?: BuiltTool; grep?: BuiltTool }): BuiltTool | null {
  if (!parts.glob && !parts.grep) return null;
  return {
    name: "find",
    label: "Find",
    description:
      "Find files in the workspace by file name pattern (e.g. *budget*) or by text inside files. Returns matching paths.",
    parameters: Type.Object({
      pattern: Type.String({
        description:
          "File name pattern with wildcards (* ?), or plain text to search inside files.",
      }),
      path: Type.Optional(Type.String({ description: "Directory to search from (default: workspace root)." })),
    }),
    execute: async (_toolCallId, params: { pattern: string; path?: string }, signal?) => {
      // 含通配符 = 按文件名找；否则是文字，文件名与内容都要搜一遍。
      const byName = params.pattern.includes("*") || params.pattern.includes("?");
      const sections: string[] = [];
      try {
        if (byName) {
          if (parts.glob) {
            const text = resultText(await parts.glob.execute(_toolCallId, { pattern: params.pattern, path: params.path }, signal)).trim();
            if (text) sections.push(text);
          }
        } else {
          if (parts.glob) {
            const text = resultText(
              await parts.glob.execute(_toolCallId, { pattern: `**/*${params.pattern}*`, path: params.path }, signal),
            ).trim();
            if (text) sections.push(`[file names]\n${text}`);
          }
          if (parts.grep) {
            const text = resultText(
              await parts.grep.execute(_toolCallId, { pattern: params.pattern, path: params.path }, signal),
            ).trim();
            if (text) sections.push(`[file contents]\n${text}`);
          }
        }
      } catch (e) {
        return errorResult(`find failed: ${e instanceof Error ? e.message : String(e)}`);
      }
      return textResult(sections.length > 0 ? sections.join("\n\n") : "Nothing found in the workspace.");
    },
  };
}

/**
 * 元工具：按组加载额外工具。
 * available 是当前还没加载、且非空的组；执行时调用 onLoad(group)，
 * 并在结果里列出该组工具名与一句话说明。
 */
export function createLoadToolsTool(available: RoutedGroupId[], onLoad: (group: RoutedGroupId) => void): BuiltTool {
  const lines = available
    .map((g) => `- ${g}: ${ROUTED_GROUPS[g].summary}`)
    .join("\n");
  return {
    name: "load_tools",
    label: "Load tools",
    description:
      "Load an extra group of tools that are not available yet:\n" +
      lines +
      "\nCall this first when none of your current tools can do what the user asked.",
    parameters: Type.Object({
      group: Type.Union(available.map((g) => Type.Literal(g))),
    }),
    execute: async (_toolCallId, params: { group: string }) => {
      const requested = params.group;
      if (!available.includes(requested as RoutedGroupId)) {
        return errorResult(
          `Unknown or already loaded group: ${requested}. Available groups: ${available.join(", ")}.`,
        );
      }
      const group = requested as RoutedGroupId;
      onLoad(group);
      const names = typeof ROUTED_GROUPS[group].toolNames === "string" ? "" : ` (${ROUTED_GROUPS[group].toolNames.join(", ")})`;
      return textResult(`Loaded group "${group}"${names}. ${ROUTED_GROUPS[group].summary}.`);
    },
  };
}

/**
 * 把 Agent 原有的完整工具列表切成核心 + 各组，并用原工具合成 recall / remember / find。
 *
 * 规则：
 * 1. 按名字建索引；核心里的 recall / remember / find 用对应原工具合成（合成失败即跳过）；
 * 2. 其余核心名字直接取原工具；
 * 3. ROUTED_GROUPS 里列出名字的组取对应原工具；"rest" 组收纳所有既不是核心、
 *    也不在 DROPPED、也不属于其他组的工具（按输入顺序）；
 * 4. 空组不放进 groups。
 */
export function partitionRoutedTools(all: BuiltTool[]): RoutedToolset {
  const byName = new Map<string, BuiltTool>();
  for (const tool of all) {
    if (!byName.has(tool.name)) byName.set(tool.name, tool);
  }

  // 合成工具用原工具做委托目标。
  const recall = createRecallTool({
    knowledge: byName.get("knowledge_search"),
    notes: byName.get("note_search"),
    memory: byName.get("memory_search"),
  });
  const remember = createRememberTool(byName.get("memory_save"));
  const find = createFindTool({ glob: byName.get("glob"), grep: byName.get("grep") });
  const synthByCoreName: Record<string, BuiltTool | null> = { recall, remember, find };

  const core: BuiltTool[] = [];
  for (const name of ROUTED_CORE_TOOL_NAMES) {
    const tool = synthByCoreName[name] ?? byName.get(name);
    if (tool) core.push(tool);
  }
  // 显式归组的核心名字（组只收这些，核心里的工具不算进组）。
  const explicitGroupNames = new Set<string>();
  const groups: Partial<Record<RoutedGroupId, BuiltTool[]>> = {};
  for (const group of ROUTED_GROUP_ORDER) {
    const spec = ROUTED_GROUPS[group];
    if (spec.toolNames === "rest") continue;
    const tools: BuiltTool[] = [];
    for (const name of spec.toolNames) {
      const tool = byName.get(name);
      if (tool) {
        tools.push(tool);
        explicitGroupNames.add(name);
      }
    }
    if (tools.length > 0) groups[group] = tools;
  }

  // "rest" 组：核心 / dropped / 其他组之外的全部（MCP 等），保持输入顺序。
  const coreNames: Set<string> = new Set(ROUTED_CORE_TOOL_NAMES as readonly string[]);
  const dropped: Set<string> = new Set(ROUTED_DROPPED_TOOL_NAMES as readonly string[]);
  const rest: BuiltTool[] = all.filter(
    (t) => !coreNames.has(t.name) && !dropped.has(t.name) && !explicitGroupNames.has(t.name),
  );
  if (rest.length > 0) groups.mcp = rest;

  return { core, groups };
}

/**
 * 拼出发给模型的工具列表：core（原顺序）+ 已加载的组（按 ROUTED_GROUP_ORDER）
 * + 若还有未加载的非空组则最后加 loadTools。
 *
 * loadTools 永远放在列表末尾：它的定义是"变动的"（available 随加载状态变化），
 * 放末尾才不会让前面的前缀缓存失效。
 */
export function assembleRoutedTools(set: RoutedToolset, loaded: RoutedGroupId[], loadTools: BuiltTool | null): BuiltTool[] {
  const loadedSet = new Set(loaded);
  const out = [...set.core];
  for (const group of ROUTED_GROUP_ORDER) {
    if (loadedSet.has(group) && set.groups[group] && set.groups[group]!.length > 0) {
      out.push(...set.groups[group]!);
    }
  }
  if (loadTools) out.push(loadTools);
  return out;
}
