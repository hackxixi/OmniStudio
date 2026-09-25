import { describe, expect, test } from "bun:test";

import { AGENT_TOOL_STRATEGIES, parseToolStrategy } from "./agent-tool-strategy";

describe("parseToolStrategy", () => {
  test("两个已知值原样通过", () => {
    expect(parseToolStrategy("classic")).toBe("classic");
    expect(parseToolStrategy("routed")).toBe("routed");
  });

  test("空值 / 未知值 / 异常类型一律 classic（老用户升级后行为不变）", () => {
    expect(parseToolStrategy(undefined)).toBe("classic");
    expect(parseToolStrategy(null)).toBe("classic");
    expect(parseToolStrategy("")).toBe("classic");
    expect(parseToolStrategy("unknown")).toBe("classic");
    expect(parseToolStrategy("ROUTED")).toBe("classic");
    expect(parseToolStrategy(1)).toBe("classic");
    expect(parseToolStrategy({})).toBe("classic");
  });

  test("AGENT_TOOL_STRATEGIES 与类型的取值一致", () => {
    expect([...AGENT_TOOL_STRATEGIES].sort()).toEqual(["classic", "routed"]);
    for (const strategy of AGENT_TOOL_STRATEGIES) {
      expect(parseToolStrategy(strategy)).toBe(strategy);
    }
  });
});
