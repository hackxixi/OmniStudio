/**
 * 精简路由工具集（agent-routed-tools.ts）的单测。
 *
 * 这里不装真实工具 —— 全部用假 BuiltTool（记录收到的参数、返回固定文本），
 * 钉住的是「切分 / 合成 / 拼装」这层的行为，而不是各原工具自己。
 */
import { describe, expect, test } from "bun:test";
import { Type } from "typebox";

import {
  ROUTED_CORE_TOOL_NAMES,
  ROUTED_DROPPED_TOOL_NAMES,
  ROUTED_GROUP_ORDER,
  assembleRoutedTools,
  createFindTool,
  createLoadToolsTool,
  createRecallTool,
  createRememberTool,
  partitionRoutedTools,
  type RoutedGroupId,
  type RoutedToolset,
} from "./agent-routed-tools";
import { type BuiltTool } from "./agent-tools";

/** 取工具结果的第一段文本（假工具只会返回文本段）。 */
function firstText(result: { content: { type: string; text?: string }[] }): string {
  return (result.content[0] as { text: string })!.text;
}

/** 假工具：记下每次收到的参数，返回一段固定文本（或抛错 / 报错）。 */
function fakeTool(
  name: string,
  reply = `result of ${name}`,
  opts: { throw?: string; isError?: boolean } = {},
): BuiltTool {
  const tool = {
    name,
    label: name,
    description: `fake ${name}`,
    parameters: Type.Object({}),
    calls: [] as unknown[],
    async execute(
      _toolCallId: string,
      params: unknown,
    ) {
      tool.calls.push(params);
      if (opts.throw) throw new Error(opts.throw);
      if (opts.isError) {
        return {
          content: [{ type: "text" as const, text: `oops ${name}` }],
          details: { error: `oops ${name}` },
          isError: true,
        };
      }
      return { content: [{ type: "text" as const, text: reply }], details: {} };
    },
  };
  return tool as unknown as BuiltTool;
}

/** 核心列表里能真实命中的那些名字（合成工具除外），按列表顺序。 */
const REAL_CORE_NAMES = ROUTED_CORE_TOOL_NAMES.filter((n) => !["recall", "remember", "find"].includes(n));
const GROUP_CREATION_NAMES = ["generate_image", "generate_speech", "generate_video", "media_search", "media_export"];
const GROUP_DEV_NAMES = ["bash", "apply_patch"];

/** 造一套"真实感"的完整工具列表：核心原工具 + 组内工具 + dropped + MCP 风格工具。 */
function makeAllTools() {
  const tools: BuiltTool[] = [];
  for (const name of [...REAL_CORE_NAMES, ...GROUP_CREATION_NAMES, ...GROUP_DEV_NAMES, ...ROUTED_DROPPED_TOOL_NAMES]) {
    tools.push(fakeTool(name));
  }
  // MCP 风格：不属于任何固定分类，应被 rest（mcp）组收纳。
  tools.push(fakeTool("mcp__jira__create_issue"), fakeTool("mcp__calendar__list_events"));
  return tools;
}

describe("partitionRoutedTools", () => {
  test("核心顺序与内容：合成工具在核心位置，其余按列表顺序取原工具", () => {
    const { core } = partitionRoutedTools(makeAllTools());
    const names = core.map((t) => t.name);
    expect(names).toEqual([...ROUTED_CORE_TOOL_NAMES]);
  });

  test("dropped 的工具不进入任何组（含 rest）", () => {
    const set = partitionRoutedTools(makeAllTools());
    const seen = new Set([...set.core, ...Object.values(set.groups).flat()].map((t) => t.name));
    for (const name of ROUTED_DROPPED_TOOL_NAMES) expect(seen.has(name)).toBe(false);
  });

  test("rest 组收纳 MCP 类工具（既非核心、非 dropped、非其他组），组内工具在固定组里", () => {
    const set = partitionRoutedTools(makeAllTools());
    expect(set.groups.mcp?.map((t) => t.name)).toEqual(["mcp__jira__create_issue", "mcp__calendar__list_events"]);
    expect(set.groups.creation?.map((t) => t.name)).toEqual(GROUP_CREATION_NAMES);
    expect(set.groups.dev?.map((t) => t.name)).toEqual(GROUP_DEV_NAMES);
  });

  test("空组不出现（没有 MCP 工具时 mcp 组缺失，其他组不受影响）", () => {
    const all = makeAllTools().filter((t) => !t.name.startsWith("mcp__"));
    const set = partitionRoutedTools(all);
    expect(set.groups.mcp).toBeUndefined();
    expect(set.groups.dev).toHaveLength(2);
  });
});

describe("recall", () => {
  const sources = () => ({
    knowledge: fakeTool("knowledge_search", "kb hit"),
    notes: fakeTool("note_search", "note hit"),
    memory: fakeTool("memory_search", "memory hit"),
  });

  test("缺全部来源时返回 null", () => {
    expect(createRecallTool({})).toBeNull();
    expect(createRecallTool({ notes: fakeTool("note_search") })).not.toBeNull();
  });

  test("默认 any：按来源顺序拼接并标注来源", async () => {
    const tool = createRecallTool(sources())!;
    const result = await tool.execute("t1", { query: "q" });
    const text = firstText(result);
    expect(text).toBe("[knowledge_base]\nkb hit\n\n[notes]\nnote hit\n\n[memory]\nmemory hit");
  });

  test("query 原样传给每个子工具", async () => {
    const s = sources() as unknown as Record<string, any>;
    const tool = createRecallTool({
      knowledge: s.knowledge as BuiltTool,
      notes: s.notes as BuiltTool,
      memory: s.memory as BuiltTool,
    })!;
    await tool.execute("t1", { query: "用户偏好" });
    expect(s.knowledge.calls).toEqual([{ query: "用户偏好" }]);
    expect(s.notes.calls).toEqual([{ query: "用户偏好" }]);
    expect(s.memory.calls).toEqual([{ query: "用户偏好" }]);
  });

  test("source 过滤：只查指定来源", async () => {
    const tool = createRecallTool({ notes: fakeTool("note_search", "note hit") } as any)!;
    const result = await tool.execute("t1", { query: "q", source: "memory" });
    expect(firstText(result)).toBe("Nothing found in the knowledge base, notes or memory.");
  });

  test("某来源抛错不影响其他来源", async () => {
    const tool = createRecallTool({
      knowledge: fakeTool("knowledge_search", "kb hit"),
      notes: fakeTool("note_search", "", { throw: "notes blew up" }),
      memory: fakeTool("memory_search", "memory hit", { isError: true }),
    })!;
    const result = await tool.execute("t1", { query: "q" });
    expect(firstText(result)).toBe(
      "[knowledge_base]\nkb hit\n\n[notes] unavailable: notes blew up\n\n[memory] unavailable: oops memory_search",
    );
  });

  test("signal 原样传给子工具", async () => {
    const seen: (AbortSignal | undefined)[] = [];
    const probe = fakeTool("memory_search", "memory hit");
    const original = probe.execute.bind(probe);
    (probe as any).execute = async (id: string, p: unknown, signal?: AbortSignal) => {
      seen.push(signal);
      return original(id, p, signal);
    };
    const tool = createRecallTool({ memory: probe })!;
    const signal = new AbortController().signal;
    await tool.execute("t1", { query: "q" }, signal);
    expect(seen).toEqual([signal]);
  });
});

describe("remember", () => {
  test("委托 memory_save（参数原样透传）", async () => {
    const save = fakeTool("memory_save", "saved") as any;
    const tool = createRememberTool(save)!;
    const result = await tool.execute("t1", { content: "用户喜欢深色主题" });
    expect(firstText(result)).toBe("saved");
    expect(save.calls).toEqual([{ content: "用户喜欢深色主题" }]);
  });

  test("缺 memory_save 返回 null", () => {
    expect(createRememberTool()).toBeNull();
  });
});

describe("find", () => {
  test("通配符 pattern：只调 glob", async () => {
    const glob = fakeTool("glob", "a/budget.csv\nb/budget.json") as any;
    const grep = fakeTool("grep", "grep should not be called") as any;
    const tool = createFindTool({ glob, grep })!;
    const result = await tool.execute("t1", { pattern: "*budget*" });
    expect(firstText(result)).toBe("a/budget.csv\nb/budget.json");
    expect(glob.calls).toEqual([{ pattern: "*budget*", path: undefined }]);
    expect(grep.calls).toEqual([]);
  });

  test("普通文字 pattern：glob（包装成 **/*<文字>*）与 grep 并行，结果分两段标注", async () => {
    const glob = fakeTool("glob", "docs/budget-notes.md") as any;
    const grep = fakeTool("grep", "src/a.ts:3: budget line") as any;
    const tool = createFindTool({ glob, grep })!;
    const result = await tool.execute("t1", { pattern: "budget", path: "docs" });
    expect(firstText(result)).toBe("[file names]\ndocs/budget-notes.md\n\n[file contents]\nsrc/a.ts:3: budget line");
    expect(glob.calls).toEqual([{ pattern: "**/*budget*", path: "docs" }]);
    expect(grep.calls).toEqual([{ pattern: "budget", path: "docs" }]);
  });

  test("两个原工具都缺返回 null；只剩 grep 时只拼内容段", async () => {
    expect(createFindTool({})).toBeNull();
    const grep = fakeTool("grep", "src/a.ts:3: budget line") as any;
    const tool = createFindTool({ grep })!;
    const result = await tool.execute("t1", { pattern: "budget" });
    expect(firstText(result)).toBe("[file contents]\nsrc/a.ts:3: budget line");
    expect(grep.calls).toEqual([{ pattern: "budget", path: undefined }]);
  });
});

describe("load_tools", () => {
  test("执行时调 onLoad 并在结果里列出该组工具名与说明", async () => {
    const loaded: RoutedGroupId[] = [];
    const tool = createLoadToolsTool(["creation", "dev"], (g) => loaded.push(g));
    expect(tool.name).toBe("load_tools");
    // 描述里逐行列出各组 summary。
    expect(tool.description).toContain("creation — generate images");
    expect(tool.description).toContain("dev — run shell commands");
    expect(tool.description).toContain("Call this first when none of your current tools can do what the user asked.");

    const result = await tool.execute("t1", { group: "dev" });
    expect(loaded).toEqual(["dev"]);
    const text = firstText(result);
    expect(text).toContain("dev");
    expect(text).toContain("bash, apply_patch");
  });

  test("未知组报错（说明可用组）；已加载的组由调用方过滤，这里模拟为 available 里没有它", async () => {
    const tool = createLoadToolsTool(["creation"], () => {});
    const result = await tool.execute("t1", { group: "nope" });
    expect(result.details).toMatchObject({ error: "Unknown or already loaded group: nope. Available groups: creation." });
  });

  test("mcp 组（rest）的结果里不列具体工具名（数量是变动的）", async () => {
    const tool = createLoadToolsTool(["mcp"], () => {});
    const result = await tool.execute("t1", { group: "mcp" });
    expect(firstText(result)).toContain("connected apps");
    expect(firstText(result)).not.toContain("(");
  });

  test("全部组都已加载（available 为空）时不暴露 load_tools，直接报错", async () => {
    const tool = createLoadToolsTool([], () => {});
    const result = await tool.execute("t1", { group: "dev" });
    expect((result.details as any).error).toContain("dev");
  });
});

describe("assembleRoutedTools", () => {
  function fixture(): { set: RoutedToolset; loadTools: BuiltTool } {
    const set = partitionRoutedTools(makeAllTools());
    const loadTools = createLoadToolsTool(["mcp"], () => {});
    return { set, loadTools };
  }

  test("只加载 dev：core + dev，无 load_tools（其余组空）", () => {
    const { set } = fixture();
    const out = assembleRoutedTools(set, ["dev"], null);
    const names = out.map((t) => t.name);
    expect(names).toEqual([...ROUTED_CORE_TOOL_NAMES, "bash", "apply_patch"]);
  });

  test("组按 ROUTED_GROUP_ORDER 排序，与调用方传入的顺序无关", () => {
    const { set } = fixture();
    const out = assembleRoutedTools(set, ["mcp", "creation"], null);
    const names = out.map((t) => t.name);
    expect(names).toEqual([
      ...ROUTED_CORE_TOOL_NAMES,
      ...GROUP_CREATION_NAMES,
      "mcp__jira__create_issue",
      "mcp__calendar__list_events",
    ]);
  });

  test("还有未加载的非空组时，load_tools 追加在末尾", () => {
    const { set, loadTools } = fixture();
    // 已加载 dev，剩 creation / mcp 未加载 → load_tools 在场；未加载的组仍不进列表。
    const out = assembleRoutedTools(set, ["dev"], loadTools);
    const names = out.map((t) => t.name);
    expect(names).toEqual([...ROUTED_CORE_TOOL_NAMES, "bash", "apply_patch", "load_tools"]);
  });

  test("未加载组 + load_tools 缺席时，加载列表就是 core + 已加载组（无尾巴）", () => {
    const { set } = fixture();
    const out = assembleRoutedTools(set, ["dev"], null);
    expect(out.map((t) => t.name)).toEqual([...ROUTED_CORE_TOOL_NAMES, "bash", "apply_patch"]);
  });

  test("全部组都加载后不再出现 load_tools", () => {
    const { set } = fixture();
    const out = assembleRoutedTools(set, [...ROUTED_GROUP_ORDER], null);
    expect(out.map((t) => t.name)).not.toContain("load_tools");
  });
});
