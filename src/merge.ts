function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export type MergeTrace = (path: string) => void

function traceLeaves(value: unknown, path: string, trace: MergeTrace) {
  if (isRecord(value) && Object.keys(value).length > 0) {
    for (const [key, child] of Object.entries(value)) {
      if (child !== undefined) traceLeaves(child, path ? `${path}.${key}` : key, trace)
    }
  } else trace(path)
}

export function mergeDeep<T>(target: T, source: unknown, trace?: MergeTrace, prefix = ""): T {
  if (!isRecord(target) || !isRecord(source)) {
    if (source !== undefined && trace) traceLeaves(source, prefix, trace)
    return source === undefined ? target : (source as T)
  }

  const output: Record<string, unknown> = { ...target }
  for (const [key, value] of Object.entries(source)) {
    if (value === undefined) continue
    const path = prefix ? `${prefix}.${key}` : key
    const existing = output[key]
    output[key] = isRecord(existing) && isRecord(value)
      ? mergeDeep(existing, value, trace, path)
      : value
    if (!(isRecord(existing) && isRecord(value)) && trace) traceLeaves(value, path, trace)
  }
  return output as T
}
