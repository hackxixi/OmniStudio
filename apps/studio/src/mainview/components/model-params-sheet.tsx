import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangleIcon,
  CheckIcon,
  ChevronDownIcon,
  CopyIcon,
  Loader2Icon,
  RotateCcwIcon,
  SlidersHorizontalIcon,
} from "lucide-react";

import { rpcClient } from "@lib/rpc";
import { Button } from "@ui/button";
import { Input } from "@ui/input";
import { Spinner } from "@ui/spinner";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@ui/collapsible";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@ui/sheet";
import { useServedStore } from "@stores/served";
import { useT } from "@stores/ui-lang";
import {
  SAMPLING_FIELDS,
  type ModelParams,
  type ResolvedSampling,
  type SamplingField,
  type SamplingParams,
  type SamplingSource,
  type ThinkingMode,
} from "@/shared/model-params";
import { SAMPLING_PRESETS } from "@/shared/sampling-presets";
import { cn } from "@/mainview/lib/utils";

/**
 * 按模型参数抽屉（「模型参数」）：已启动模型行 / 已安装模型行各有一个入口。
 *
 * `model` = 启动这个模型时用的 target 字符串（已安装行的 runtimeTarget、已服务实例的 modelRef），
 * 服务端会再按注册表规则归一一次，两处入口落到同一份参数。
 *
 * 表单约定：输入框留空 = 不覆盖（落回自动规划 / 全局 / 家族推荐），不会把空串发给后端；
 * 保存是整份替换（setModelParams），所以提交的永远是整张表单。
 */

// ---------------------------------------------------------------------------
// 纯函数：表单草稿 <-> ModelParams（导出给测试）
// ---------------------------------------------------------------------------

/**
 * llama.cpp `--cache-type-k/-v` 认的取值。与主进程 bun/db/model-params.ts 的 KV_CACHE_TYPES 同一份
 * （界面不能 import bun 侧模块：那边会连带初始化数据库），改一边记得改另一边。
 */
export const KV_CACHE_TYPE_OPTIONS = [
  "f16",
  "q8_0",
  "q4_0",
  "q4_1",
  "iq4_nl",
  "q5_0",
  "q5_1",
  "bf16",
  "f32",
] as const;

type Range = { min: number; max: number; int?: boolean };

/** 范围与主进程校验（sanitizeModelParams）一致；前端先拦一遍，免得「保存了却被静默丢掉」。 */
export const LAUNCH_RANGES = {
  ctxSize: { min: 256, max: 1_048_576, int: true },
  parallel: { min: 1, max: 64, int: true },
  gpuLayers: { min: -1, max: 999, int: true },
} as const satisfies Record<string, Range>;

export const SAMPLING_RANGES: Record<SamplingField, Range> = {
  temperature: { min: 0, max: 5 },
  topP: { min: 0, max: 1 },
  topK: { min: 0, max: 1000, int: true },
  minP: { min: 0, max: 1 },
  presencePenalty: { min: -2, max: 2 },
  repeatPenalty: { min: 0.5, max: 2 },
};

type LaunchNumberField = keyof typeof LAUNCH_RANGES;
const LAUNCH_NUMBER_FIELDS: LaunchNumberField[] = ["ctxSize", "parallel", "gpuLayers"];

/** 下拉里「不覆盖」的占位值（radix Select 不接受空串）。 */
const INHERIT = "__inherit";

export type ModelParamsDraft = {
  ctxSize: string;
  parallel: string;
  gpuLayers: string;
  cacheTypeK: string;
  cacheTypeV: string;
  flashAttn: string;
  thinking: ThinkingMode;
  extraArgs: string;
  sampling: Record<SamplingField, string>;
};

export type DraftError = { kind: "range"; min: number; max: number } | { kind: "int" } | { kind: "nan" };
export type DraftErrors = Partial<Record<LaunchNumberField | SamplingField, DraftError>>;

export function draftFromParams(params: ModelParams | null | undefined): ModelParamsDraft {
  const p = params ?? {};
  const str = (v: number | undefined) => (v === undefined ? "" : String(v));
  const sampling = {} as Record<SamplingField, string>;
  for (const f of SAMPLING_FIELDS) sampling[f] = str(p.sampling?.[f]);
  return {
    ctxSize: str(p.ctxSize),
    parallel: str(p.parallel),
    gpuLayers: str(p.gpuLayers),
    cacheTypeK: p.cacheTypeK ?? INHERIT,
    cacheTypeV: p.cacheTypeV ?? INHERIT,
    flashAttn: p.flashAttn ?? INHERIT,
    thinking: p.thinking ?? "auto",
    extraArgs: p.extraArgs ?? "",
    sampling,
  };
}

function parseField(raw: string, range: Range): { value?: number; error?: DraftError } {
  const s = raw.trim();
  if (s === "") return {};
  const n = Number(s);
  if (!Number.isFinite(n)) return { error: { kind: "nan" } };
  if (range.int && !Number.isInteger(n)) return { error: { kind: "int" } };
  if (n < range.min || n > range.max) return { error: { kind: "range", min: range.min, max: range.max } };
  return { value: n };
}

/**
 * 草稿 → 要提交的 ModelParams。空输入 / 「跟随」不出现在结果里；有错的字段也不出现
 * （调用方看到 errors 非空就不该提交）。「思考：跟随模板」= 不覆盖，不写 thinking 字段。
 */
export function buildModelParams(draft: ModelParamsDraft): { params: ModelParams; errors: DraftErrors } {
  const params: ModelParams = {};
  const errors: DraftErrors = {};
  for (const f of LAUNCH_NUMBER_FIELDS) {
    const { value, error } = parseField(draft[f], LAUNCH_RANGES[f]);
    if (error) errors[f] = error;
    else if (value !== undefined) params[f] = value;
  }
  if (draft.cacheTypeK !== INHERIT && draft.cacheTypeK) params.cacheTypeK = draft.cacheTypeK;
  if (draft.cacheTypeV !== INHERIT && draft.cacheTypeV) params.cacheTypeV = draft.cacheTypeV;
  if (draft.flashAttn === "auto" || draft.flashAttn === "on" || draft.flashAttn === "off") {
    params.flashAttn = draft.flashAttn;
  }
  if (draft.thinking === "on" || draft.thinking === "off") params.thinking = draft.thinking;
  const sampling: SamplingParams = {};
  for (const f of SAMPLING_FIELDS) {
    const { value, error } = parseField(draft.sampling[f], SAMPLING_RANGES[f]);
    if (error) errors[f] = error;
    else if (value !== undefined) sampling[f] = value;
  }
  if (Object.keys(sampling).length > 0) params.sampling = sampling;
  const extra = draft.extraArgs.trim();
  if (extra) params.extraArgs = extra;
  return { params, errors };
}

// ---------------------------------------------------------------------------
// 查询 key 与「已自定义」标记
// ---------------------------------------------------------------------------

export const modelParamsQueryKey = (model: string) => ["model-params", model] as const;
export const MODEL_PARAMS_LIST_KEY = ["model-params-list"] as const;

/**
 * 存过按模型参数的模型集合（listModelParams），模型行上标「已自定义」用。
 * 服务端存的是归一后的 key，行上的 target 一般就是它；调用方把几种写法都拿来比一下。
 */
export function useCustomizedModels(): (...targets: (string | undefined)[]) => boolean {
  const { data } = useQuery({
    queryKey: MODEL_PARAMS_LIST_KEY,
    queryFn: () => rpcClient.listModelParams(undefined),
    staleTime: 30_000,
  });
  return useMemo(() => {
    const set = new Set((data?.entries ?? []).map((e) => e.model));
    return (...targets) => targets.some((t) => !!t && set.has(t));
  }, [data]);
}

/** 行上的小标记：「已自定义」。 */
export function CustomizedParamsBadge({ className }: { className?: string }) {
  const t = useT();
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center gap-1 rounded-full bg-primary/10 px-1.5 text-[10px] font-medium text-primary",
        className,
      )}
      title={t("modelParams.customizedHint")}
    >
      <span className="size-1.5 rounded-full bg-primary" />
      {t("modelParams.customized")}
    </span>
  );
}

/** 行上的入口按钮 + 抽屉本体（抽屉只在打开时拉数据）。 */
export function ModelParamsButton({
  model,
  label,
  size = "icon-sm",
}: {
  model: string;
  label: string;
  size?: "icon-sm" | "xs";
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  return (
    <>
      {size === "xs" ? (
        <Button variant="outline" size="xs" tooltip={t("modelParams.title")} onClick={() => setOpen(true)}>
          <SlidersHorizontalIcon data-icon="inline-start" />
          {t("modelParams.open")}
        </Button>
      ) : (
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={t("modelParams.title")}
          tooltip={t("modelParams.title")}
          onClick={() => setOpen(true)}
          className="text-muted-foreground"
        >
          <SlidersHorizontalIcon className="size-4" />
        </Button>
      )}
      <ModelParamsSheet model={model} label={label} open={open} onOpenChange={setOpen} />
    </>
  );
}

// ---------------------------------------------------------------------------
// 抽屉
// ---------------------------------------------------------------------------

export function ModelParamsSheet({
  model,
  label,
  open,
  onOpenChange,
}: {
  model: string;
  label: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full gap-0 data-[side=right]:w-full data-[side=right]:sm:max-w-lg"
      >
        <SheetHeader className="border-b pr-12">
          <SheetTitle className="text-sm">{t("modelParams.title")}</SheetTitle>
          <SheetDescription className="truncate text-xs" title={model}>
            {label}
          </SheetDescription>
        </SheetHeader>
        {open && <ModelParamsBody model={model} />}
      </SheetContent>
    </Sheet>
  );
}

/**
 * 抽屉内容：拉数据 + 保存 / 清除 / 重启。
 * 所有 hook 都在任何提前 return 之前（这个仓库吃过 React #300/#310 的亏）。
 */
export function ModelParamsBody({ model }: { model: string }) {
  const t = useT();
  const queryClient = useQueryClient();
  const servedModels = useServedStore((s) => s.models);
  const [actionError, setActionError] = useState<string | null>(null);
  const [restartPending, setRestartPending] = useState(false);

  const query = useQuery({
    queryKey: modelParamsQueryKey(model),
    queryFn: () => rpcClient.getModelParams({ model }),
    // 表单草稿按这份数据初始化：窗口聚焦重拉会把正在填的内容冲掉
    refetchOnWindowFocus: false,
  });
  const data = query.data;

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: modelParamsQueryKey(model) });
    queryClient.invalidateQueries({ queryKey: MODEL_PARAMS_LIST_KEY });
  };

  const saveMutation = useMutation({
    mutationFn: (params: ModelParams) => rpcClient.setModelParams({ model, params }),
    onSuccess: (res) => {
      if (!res.ok) {
        setActionError(res.error ?? t("modelParams.saveFailed"));
        return;
      }
      setActionError(null);
      if (res.needsRestart) setRestartPending(true);
      invalidate();
    },
    onError: (e: unknown) => setActionError(e instanceof Error ? e.message : String(e)),
  });
  const clearMutation = useMutation({
    mutationFn: () => rpcClient.clearModelParams({ model }),
    onSuccess: (res) => {
      if (!res.ok) {
        setActionError(t("modelParams.saveFailed"));
        return;
      }
      setActionError(null);
      if (res.needsRestart) setRestartPending(true);
      invalidate();
    },
    onError: (e: unknown) => setActionError(e instanceof Error ? e.message : String(e)),
  });

  const keys = new Set([model, data?.model].filter(Boolean) as string[]);
  const served = servedModels.find((m) => keys.has(m.modelRef));
  const restartMutation = useMutation({
    mutationFn: async () => {
      if (!served) throw new Error("not running");
      const res = await rpcClient.restartServedModel({ id: served.id });
      if (!res.ok) throw new Error(res.error ?? t("console.restartFailed"));
      return res;
    },
    onSuccess: () => {
      setActionError(null);
      setRestartPending(false);
      queryClient.invalidateQueries({ queryKey: ["served-models"] });
      invalidate();
    },
    onError: (e: unknown) => setActionError(e instanceof Error ? e.message : String(e)),
  });

  if (query.isLoading || !data) {
    return (
      <div className="flex flex-1 items-center justify-center py-10">
        {query.isError ? (
          <p className="px-4 text-xs text-destructive">{String(query.error)}</p>
        ) : (
          <Spinner className="size-5" />
        )}
      </div>
    );
  }

  const needsRestart = data.needsRestart || restartPending;

  return (
    <ModelParamsForm
      // 数据换了（保存后重拉）就用新数据重新初始化草稿
      key={`${data.model}:${query.dataUpdatedAt}`}
      params={data.params}
      sampling={data.sampling}
      launchPreview={data.launchPreview}
      hasSaved={data.params !== null}
      saving={saveMutation.isPending}
      clearing={clearMutation.isPending}
      error={actionError}
      needsRestart={needsRestart}
      canRestart={!!served}
      restarting={restartMutation.isPending}
      onSave={(params) => saveMutation.mutate(params)}
      onClear={() => clearMutation.mutate()}
      onRestart={() => restartMutation.mutate()}
    />
  );
}

function ModelParamsForm({
  params,
  sampling,
  launchPreview,
  hasSaved,
  saving,
  clearing,
  error,
  needsRestart,
  canRestart,
  restarting,
  onSave,
  onClear,
  onRestart,
}: {
  params: ModelParams | null;
  sampling: ResolvedSampling;
  launchPreview?: string;
  hasSaved: boolean;
  saving: boolean;
  clearing: boolean;
  error: string | null;
  needsRestart: boolean;
  canRestart: boolean;
  restarting: boolean;
  onSave: (params: ModelParams) => void;
  onClear: () => void;
  onRestart: () => void;
}) {
  const t = useT();
  const [draft, setDraft] = useState<ModelParamsDraft>(() => draftFromParams(params));
  const { params: built, errors } = useMemo(() => buildModelParams(draft), [draft]);
  const hasErrors = Object.keys(errors).length > 0;

  const set = <K extends keyof ModelParamsDraft>(key: K, value: ModelParamsDraft[K]) =>
    setDraft((d) => ({ ...d, [key]: value }));
  const setSampling = (field: SamplingField, value: string) =>
    setDraft((d) => ({ ...d, sampling: { ...d.sampling, [field]: value } }));

  const errorText = (e: DraftError | undefined) =>
    !e
      ? null
      : e.kind === "range"
        ? t("modelParams.err.range", { min: String(e.min), max: String(e.max) })
        : e.kind === "int"
          ? t("modelParams.err.int")
          : t("modelParams.err.nan");

  const presetLabel = sampling.preset ? (SAMPLING_PRESETS[sampling.preset]?.label ?? sampling.preset) : null;

  return (
    <>
      <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto p-4 text-xs">
        {/* 启动 */}
        <section className="flex flex-col gap-3">
          <SectionTitle title={t("modelParams.launch")} hint={t("modelParams.launchHint")} />
          <div className="grid grid-cols-2 gap-3">
            <NumberField
              className="col-span-2"
              label={t("modelParams.ctxSize")}
              hint={t("modelParams.ctxSizeHint")}
              value={draft.ctxSize}
              placeholder={t("modelParams.inheritAuto")}
              error={errorText(errors.ctxSize)}
              onChange={(v) => set("ctxSize", v)}
            />
            <NumberField
              label={t("modelParams.parallel")}
              value={draft.parallel}
              placeholder={t("modelParams.inheritGlobal")}
              error={errorText(errors.parallel)}
              onChange={(v) => set("parallel", v)}
            />
            <NumberField
              label={t("modelParams.gpuLayers")}
              hint={t("modelParams.gpuLayersHint")}
              value={draft.gpuLayers}
              placeholder={t("modelParams.inheritGlobal")}
              error={errorText(errors.gpuLayers)}
              onChange={(v) => set("gpuLayers", v)}
            />
            <SelectField
              label={t("modelParams.cacheTypeK")}
              value={draft.cacheTypeK}
              onChange={(v) => set("cacheTypeK", v)}
              options={KV_CACHE_TYPE_OPTIONS.map((v) => ({ value: v, label: v }))}
            />
            <SelectField
              label={t("modelParams.cacheTypeV")}
              value={draft.cacheTypeV}
              onChange={(v) => set("cacheTypeV", v)}
              options={KV_CACHE_TYPE_OPTIONS.map((v) => ({ value: v, label: v }))}
            />
            <SelectField
              className="col-span-2"
              label={t("modelParams.flashAttn")}
              value={draft.flashAttn}
              onChange={(v) => set("flashAttn", v)}
              options={[
                { value: "auto", label: t("models.params.flashAttn.auto") },
                { value: "on", label: t("models.params.flashAttn.on") },
                { value: "off", label: t("models.params.flashAttn.off") },
              ]}
            />
            <label className="col-span-2 flex flex-col gap-1">
              <span className="text-[11px] text-muted-foreground">{t("modelParams.extraArgs")}</span>
              <Input
                value={draft.extraArgs}
                onChange={(e) => set("extraArgs", e.target.value)}
                placeholder="--jinja --reasoning-budget 0"
                className="h-8 font-mono text-xs"
                spellCheck={false}
              />
              <span className="text-[10px] leading-relaxed text-muted-foreground/70">
                {t("modelParams.extraArgsHint")}
              </span>
            </label>
          </div>
        </section>

        {/* 思考 */}
        <section className="flex flex-col gap-2">
          <SectionTitle title={t("modelParams.thinking")} hint={t("modelParams.thinkingHint")} />
          <div role="radiogroup" className="inline-flex w-fit rounded-md border p-0.5">
            {(["auto", "on", "off"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                role="radio"
                aria-checked={draft.thinking === mode}
                onClick={() => set("thinking", mode)}
                className={cn(
                  "rounded px-3 py-1 text-xs transition-colors",
                  draft.thinking === mode
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {t(`modelParams.thinking.${mode}`)}
              </button>
            ))}
          </div>
        </section>

        {/* 采样 */}
        <section className="flex flex-col gap-3">
          <div className="flex items-start gap-2">
            <SectionTitle
              title={t("modelParams.sampling")}
              hint={
                presetLabel
                  ? t("modelParams.samplingHintPreset", { preset: presetLabel })
                  : t("modelParams.samplingHint")
              }
            />
            <Button
              variant="ghost"
              size="xs"
              className="ml-auto shrink-0"
              disabled={SAMPLING_FIELDS.every((f) => draft.sampling[f] === "")}
              onClick={() =>
                setDraft((d) => ({ ...d, sampling: draftFromParams(null).sampling }))
              }
            >
              <RotateCcwIcon data-icon="inline-start" />
              {t("modelParams.resetSection")}
            </Button>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {SAMPLING_FIELDS.map((f) => {
              const source = sampling.sources[f];
              const value = draft.sampling[f];
              // 存过覆盖、这次清空了：显示的有效值还是覆盖值，提示「保存后回到推荐」
              const pendingReset = value === "" && source === "model-override";
              return (
                <div key={f} className="flex flex-col gap-1" data-field={f}>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] text-muted-foreground">{t(`modelParams.s.${f}`)}</span>
                    <SourceBadge
                      source={pendingReset ? null : source}
                      presetLabel={presetLabel}
                    />
                    {value !== "" && (
                      <button
                        type="button"
                        onClick={() => setSampling(f, "")}
                        title={t("modelParams.resetField")}
                        aria-label={t("modelParams.resetField")}
                        className="ml-auto text-muted-foreground/70 hover:text-foreground"
                      >
                        <RotateCcwIcon className="size-3" />
                      </button>
                    )}
                  </div>
                  <Input
                    type="number"
                    step="any"
                    name={f}
                    value={value}
                    placeholder={pendingReset ? "" : String(sampling.values[f])}
                    onChange={(e) => setSampling(f, e.target.value)}
                    aria-invalid={errors[f] ? true : undefined}
                    className="h-8 text-xs"
                  />
                  {errors[f] && <span className="text-[10px] text-destructive">{errorText(errors[f])}</span>}
                </div>
              );
            })}
          </div>
        </section>

        {launchPreview && <LaunchPreview command={launchPreview} />}
      </div>

      <SheetFooter className="gap-2 border-t">
        {needsRestart && (
          <div
            className="flex items-center gap-2 rounded-md bg-amber-500/10 px-2.5 py-1.5 text-[11px] text-amber-700 dark:text-amber-400"
            data-testid="model-params-restart"
          >
            <AlertTriangleIcon className="size-3.5 shrink-0" />
            <span className="flex-1">{t("modelParams.needsRestart")}</span>
            {canRestart && (
              <Button variant="outline" size="xs" disabled={restarting} onClick={onRestart}>
                {restarting ? (
                  <Loader2Icon data-icon="inline-start" className="animate-spin" />
                ) : (
                  <RotateCcwIcon data-icon="inline-start" />
                )}
                {t("console.restart")}
              </Button>
            )}
          </div>
        )}
        {error && (
          <p className="flex items-start gap-1 text-[11px] text-destructive">
            <AlertTriangleIcon className="mt-0.5 size-3 shrink-0" />
            <span className="min-w-0 break-words">{error}</span>
          </p>
        )}
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 text-xs text-muted-foreground"
            disabled={!hasSaved || clearing || saving}
            onClick={onClear}
          >
            {clearing && <Loader2Icon data-icon="inline-start" className="animate-spin" />}
            {t("modelParams.clearAll")}
          </Button>
          <Button
            size="sm"
            className="ml-auto h-8 text-xs"
            disabled={hasErrors || saving || clearing}
            onClick={() => onSave(built)}
          >
            {saving && <Loader2Icon data-icon="inline-start" className="animate-spin" />}
            {t("common.save")}
          </Button>
        </div>
      </SheetFooter>
    </>
  );
}

function SectionTitle({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <h3 className="text-xs font-medium">{title}</h3>
      {hint && <p className="text-[10px] leading-relaxed text-muted-foreground/70">{hint}</p>}
    </div>
  );
}

function NumberField({
  label,
  hint,
  value,
  placeholder,
  error,
  onChange,
  className,
}: {
  label: string;
  hint?: string;
  value: string;
  placeholder?: string;
  error: string | null;
  onChange: (v: string) => void;
  className?: string;
}) {
  return (
    <label className={cn("flex flex-col gap-1", className)}>
      <span className="text-[11px] text-muted-foreground">{label}</span>
      <Input
        type="number"
        step="1"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        aria-invalid={error ? true : undefined}
        className="h-8 text-xs"
      />
      {error ? (
        <span className="text-[10px] text-destructive">{error}</span>
      ) : (
        hint && <span className="text-[10px] leading-relaxed text-muted-foreground/70">{hint}</span>
      )}
    </label>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
  className,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  className?: string;
}) {
  const t = useT();
  return (
    <label className={cn("flex flex-col gap-1", className)}>
      <span className="text-[11px] text-muted-foreground">{label}</span>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-8 w-full text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={INHERIT}>{t("modelParams.inheritGlobal")}</SelectItem>
          {options.map((o) => (
            <SelectItem key={o.value} value={o.value}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  );
}

const SOURCE_TONE: Record<SamplingSource, string> = {
  "model-override": "bg-primary/10 text-primary",
  "model-metadata": "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  "family-preset": "bg-sky-500/15 text-sky-700 dark:text-sky-400",
  global: "bg-muted text-muted-foreground",
};

function SourceBadge({ source, presetLabel }: { source: SamplingSource | null; presetLabel: string | null }) {
  const t = useT();
  if (source === null) {
    return (
      <span className="rounded-sm bg-amber-500/15 px-1 text-[9px] leading-4 text-amber-700 dark:text-amber-400">
        {t("modelParams.src.pendingReset")}
      </span>
    );
  }
  const text =
    source === "family-preset"
      ? presetLabel
        ? t("modelParams.src.familyPresetNamed", { preset: presetLabel })
        : t("modelParams.src.familyPreset")
      : t(`modelParams.src.${source}`);
  return (
    <span className={cn("rounded-sm px-1 text-[9px] leading-4", SOURCE_TONE[source])} data-source={source}>
      {text}
    </span>
  );
}

function LaunchPreview({ command }: { command: string }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪贴板不可用：忽略
    }
  };
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="flex flex-col gap-2 border-t pt-3">
      <div className="flex items-center gap-2">
        <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground">
          <ChevronDownIcon className={cn("size-3.5 transition-transform", !open && "-rotate-90")} />
          {t("modelParams.launchPreview")}
        </CollapsibleTrigger>
        <Button variant="ghost" size="icon-xs" className="ml-auto" aria-label={t("common.copy")} onClick={() => void copy()}>
          {copied ? <CheckIcon className="text-emerald-500" /> : <CopyIcon />}
        </Button>
      </div>
      <CollapsibleContent>
        <pre className="max-h-48 overflow-auto rounded-md bg-muted p-2 font-mono text-[10px] leading-relaxed break-all whitespace-pre-wrap">
          {command}
        </pre>
      </CollapsibleContent>
    </Collapsible>
  );
}
