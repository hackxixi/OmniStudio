import { describe, expect, test } from "bun:test";

import {
  parseNvidiaSmiMemory,
  parseVmStatAvailable,
  usedRatio,
} from "./hardware";

/**
 * 顶栏状态胶囊的两个「拿真机输出钉格式」的解析函数（bun/hardware.ts 只做 spawn 与
 * 缓存）。断言都按各平台的**真实**输出写 —— nvidia-smi 的 CSV、macOS 的 vm_stat。
 */

describe("parseVmStatAvailable", () => {
  // 真机输出（M4，16K 页，数字末尾带句点）
  const REAL_VM_STAT = `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              106677.
Pages active:                            314708.
Pages inactive:                          308287.
Pages speculative:                         5435.
Pages throttled:                              0.
Pages wired down:                        137137.
Pages purgeable:                           3284.
"Translation faults":                1609394499.
`;

  test("真机 vm_stat：free + inactive + speculative + purgeable 都算可用，wired / active 不算", () => {
    const expectedPages = 106677 + 308287 + 5435 + 3284;
    expect(parseVmStatAvailable(REAL_VM_STAT)).toBe(expectedPages * 16384);
  });

  test("没有页大小行返回 null（调用方回退 freemem()）", () => {
    expect(parseVmStatAvailable("Pages free: 106677.\n")).toBeNull();
  });

  test("没有 Pages free 行返回 null", () => {
    expect(parseVmStatAvailable("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages active: 314708.\n")).toBeNull();
  });
});

describe("parseNvidiaSmiMemory", () => {
  test("单卡：两列 MiB → bytes", () => {
    // 真机输出（`--query-gpu=memory.total,memory.free --format=csv,noheader,nounits`）
    const res = parseNvidiaSmiMemory("24564, 20133\n");
    expect(res).toEqual({
      totalBytes: 24564 * 1024 * 1024,
      freeBytes: 20133 * 1024 * 1024,
    });
  });

  test("多卡求和（胶囊要的是机器总量）", () => {
    const res = parseNvidiaSmiMemory("24564, 20000\n16384, 12000\n");
    expect(res).toEqual({
      totalBytes: (24564 + 16384) * 1024 * 1024,
      freeBytes: (20000 + 12000) * 1024 * 1024,
    });
  });

  test("某张卡读不出来（[N/A]）就跳过该行，不把缺失当 0", () => {
    const res = parseNvidiaSmiMemory("24564, 20000\n16384, [N/A]\n");
    expect(res).toEqual({
      totalBytes: 24564 * 1024 * 1024,
      freeBytes: 20000 * 1024 * 1024,
    });
  });

  test("没有任何可读行返回 null（没有 N 卡 / 输出为空）", () => {
    expect(parseNvidiaSmiMemory("")).toBeNull();
    expect(parseNvidiaSmiMemory("NOT_FOUND, [N/A]\n")).toBeNull();
    expect(parseNvidiaSmiMemory("24564, [N/A]\n16384, [Not Supported]\n")).toBeNull();
  });
});

describe("usedRatio", () => {
  test("正常比例（0-1）", () => {
    expect(usedRatio({ totalBytes: 100, freeBytes: 30 })).toBeCloseTo(0.7);
    expect(usedRatio({ totalBytes: 100, freeBytes: 0 })).toBe(1);
    expect(usedRatio({ totalBytes: 100, freeBytes: 100 })).toBe(0);
  });

  test("读数缺失返回 null（条不画），free > total 钳到 0 而不是负数", () => {
    expect(usedRatio({ totalBytes: null, freeBytes: 50 })).toBeNull();
    expect(usedRatio({ totalBytes: 100, freeBytes: null })).toBeNull();
    expect(usedRatio({ totalBytes: 0, freeBytes: 0 })).toBeNull();
    expect(usedRatio({ totalBytes: 100, freeBytes: 150 })).toBe(0);
  });
});
