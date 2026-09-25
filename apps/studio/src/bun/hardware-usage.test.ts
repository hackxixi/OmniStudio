import { describe, expect, test } from "bun:test";

import {
  clearResourceUsageCache,
  getResourceUsage,
  type CommandRunner,
} from "./hardware";

/**
 * getResourceUsage 的三条降级路径各走一遍：vm_stat 挂掉回 freemem()、nvidia-smi
 * 读不出来只把 vram 置 null、2 秒缓存内不重复 spawn。
 *
 * 注：runner 注入只影响 macOS 的 vm_stat 分支；非 darwin 平台的 freemem / 非 Apple
 * 芯片的 nvidia-smi 走的是**真实**命令（与 getHardwareInfo 同一注入面），所以断言
 * 只钉「结构 + 降级方向」，不钉具体字节数。
 */
const GB = 1e9;

function fakeRunner(answers: Record<string, { code?: number; stdout?: string; stderr?: string }>) {
  const calls: string[] = [];
  return {
    calls,
    run: (cmd: string[]) => {
      const key = cmd.join(" ");
      calls.push(key);
      const answer = answers[key];
      return {
        code: answer?.code ?? 127,
        stdout: answer?.stdout ?? "",
        stderr: answer?.stderr ?? `command not found: ${cmd[0]}`,
      };
    },
    runStreaming: async () => -1,
  };
}

const VM_STAT_OK = `Mach Virtual Memory Statistics: (page size of 16384 bytes)

Pages free:                                   24576
Pages inactive:                              524288
Pages speculative:                             1024
Pages purgeable:                              262144
`;

describe("getResourceUsage", () => {
  test("macOS 正常路径：vm_stat 算出的可用内存进 ram，Apple 芯片只有一段内存", () => {
    clearResourceUsageCache();
    const runner = fakeRunner({
      "vm_stat": { code: 0, stdout: VM_STAT_OK },
    });
    const usage = getResourceUsage({
      refresh: true,
      now: 1_000,
      runner,
      readMacAvailable: () => null, // 屏蔽真实 vm_stat，走 runner
      readNvidiaMemory: () => null, // Apple 芯片没有独立显存
    });

    // 24576 + 524288 + 1024 + 262144 = 811 332 页 × 16384 = 13.29 GiB（freemem() 会少得多）
    expect(usage.ram.freeBytes).toBe((24576 + 524288 + 1024 + 262144) * 16384);
    expect(usage.ram.totalBytes).toBeGreaterThan(0);
    expect(usage.vram).toBeNull();
    expect(usage.unifiedMemory).toBe(true);
    // vm_stat 跑了一次，nvidia-smi 不该被跑（Apple 芯片没有独立显存，连命令都不跑）
    expect(runner.calls.some((c) => c.startsWith("vm_stat"))).toBe(true);
    expect(runner.calls.some((c) => c.startsWith("nvidia-smi"))).toBe(false);
  });

  test("vm_stat 解析失败回退 freemem()", () => {
    clearResourceUsageCache();
    const runner = fakeRunner({
      // 输出格式坏了：没有页大小行
      "vm_stat": { code: 0, stdout: "vm_stat: unknown option -p\n" },
    });
    const usage = getResourceUsage({
      refresh: true,
      now: 2_000,
      runner,
      readMacAvailable: () => null, // 走 runner 的坏输出 → parseVmStatAvailable 返回 null
      readNvidiaMemory: () => null,
    });
    // 回退到 freemem()：是真实读数，只断言方向（小于总量、为正）
    expect(usage.ram.freeBytes).toBeGreaterThan(0);
    expect(usage.ram.freeBytes).toBeLessThanOrEqual(usage.ram.totalBytes!);
  });

  test("NVIDIA 机器：两段（内存 + 显存），多卡求和", () => {
    clearResourceUsageCache();
    const runner = fakeRunner({
      "vm_stat": { code: 0, stdout: VM_STAT_OK },
    });
    const usage = getResourceUsage({
      refresh: true,
      now: 3_000,
      runner,
      readMacAvailable: () => 4 * GB,
      // 两张 24 GiB 卡，各剩 20 GiB
      readNvidiaMemory: () => ({
        totalBytes: 2 * 24 * GB,
        freeBytes: 2 * 20 * GB,
      }),
    });
    expect(usage.ram.freeBytes).toBe(4 * GB);
    expect(usage.vram).toEqual({ totalBytes: 48 * GB, freeBytes: 40 * GB });
  });

  test("nvidia-smi 读不出来：vram 置 null，内存段照常", () => {
    clearResourceUsageCache();
    const runner = fakeRunner({
      "vm_stat": { code: 0, stdout: VM_STAT_OK },
    });
    const usage = getResourceUsage({
      refresh: true,
      now: 4_000,
      runner,
      readMacAvailable: () => 4 * GB,
      readNvidiaMemory: () => null, // 没有独立显存
    });
    expect(usage.vram).toBeNull();
    expect(usage.ram.freeBytes).toBe(4 * GB);
  });

  test("2 秒缓存：窗口内重复调用不重复 spawn", () => {
    clearResourceUsageCache();
    const runner = fakeRunner({
      "vm_stat": { code: 0, stdout: VM_STAT_OK },
    });
    const a = getResourceUsage({
      refresh: true,
      now: 10_000,
      runner,
      readMacAvailable: () => null,
      readNvidiaMemory: () => null,
    });
    const b = getResourceUsage({
      refresh: false, // 不强制刷新
      now: 11_000, // 1 秒后（< 2s TTL）
      runner,
      readMacAvailable: () => null,
      readNvidiaMemory: () => null,
    });
    // 第二次命中缓存，没有再 spawn
    expect(b).toBe(a);
    const vmStatCalls = runner.calls.filter((c) => c.startsWith("vm_stat")).length;
    expect(vmStatCalls).toBe(1);

    clearResourceUsageCache();
  });

  test("refresh: true 忽略缓存", () => {
    clearResourceUsageCache();
    const runner = fakeRunner({
      "vm_stat": { code: 0, stdout: VM_STAT_OK },
    });
    getResourceUsage({
      refresh: true,
      now: 20_000,
      runner,
      readMacAvailable: () => null,
      readNvidiaMemory: () => null,
    });
    getResourceUsage({
      refresh: true, // 强制刷新
      now: 20_500,
      runner,
      readMacAvailable: () => null,
      readNvidiaMemory: () => null,
    });
    const vmStatCalls = runner.calls.filter((c) => c.startsWith("vm_stat")).length;
    expect(vmStatCalls).toBe(2);

    clearResourceUsageCache();
  });
});
