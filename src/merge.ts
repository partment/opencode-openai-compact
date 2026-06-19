function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export function mergeDeep<T>(target: T, source: unknown): T {
  if (!isRecord(target) || !isRecord(source)) return source === undefined ? target : (source as T)

  const output: Record<string, unknown> = { ...target }
  for (const [key, value] of Object.entries(source)) {
    if (value === undefined) continue
    const existing = output[key]
    output[key] = isRecord(existing) && isRecord(value) ? mergeDeep(existing, value) : value
  }
  return output as T
}
