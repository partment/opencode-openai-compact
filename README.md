# opencode-openai-compact

[![npm version](https://img.shields.io/npm/v/opencode-openai-compact?style=flat-square)](https://www.npmjs.com/package/opencode-openai-compact)

OpenCode plugin that lets OpenCode use OpenAI's official Responses API compaction.

## Installation

Add the package to your OpenCode config.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["opencode-openai-compact"]
}
```

For a local checkout during development, use a file URL.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["file:///path/to/opencode-openai-compact"]
}
```

Restart OpenCode after changing plugin config.

## Configuration Files

Create `openai-compact.json` or `openai-compact.jsonc` in any supported location. Later layers override earlier layers. Within the same directory, `openai-compact.jsonc` is read after `openai-compact.json` and can override it.

Read order:

1. Built-in defaults.
2. Global OpenCode config directory: `openai-compact.json`, then `openai-compact.jsonc`.
3. Directory from `OPENCODE_CONFIG_DIR`: `openai-compact.json`, then `openai-compact.jsonc`.
4. Nearest project `.opencode` directory found by walking upward from the current directory: `openai-compact.json`, then `openai-compact.jsonc`.

Global OpenCode config directory:

- `$XDG_CONFIG_HOME/opencode`, when `XDG_CONFIG_HOME` is set.
- `~/.config/opencode`, when `XDG_CONFIG_HOME` is not set.

If neither `openai-compact.json` nor `openai-compact.jsonc` exists in the global OpenCode config directory, the plugin creates an empty `openai-compact.jsonc` file on first load.

## State Database

Runtime checkpoints are stored in SQLite at:

```text
~/.config/opencode/openai-compact/checkpoints.db
```

## Example Configuration

```jsonc
{
  "$schema": "https://raw.githubusercontent.com/partment/opencode-openai-compact/main/configSchema.json",
  "enabled": true,
  "providers": {
    "openai": {
      "compactModel": "gpt-5.4"
    }
  },
  "headers": {
    "compact": "x-opencode-openai-responses-compact",
    "session": "x-opencode-openai-responses-compact-session"
  },
  "responses": {
    "endpointPath": "/responses",
    "compactEndpointPath": "/responses/compact"
  },
  "compactBodyKeys": [
    "input",
    "instructions",
    "previous_response_id",
    "prompt_cache_key",
    "prompt_cache_retention",
    "service_tier"
  ],
  "summary": "Context compacted.\nFollowing conversations will continue from this compacted checkpoint.",
  "state": {
    "retentionDays": 30,
    "deleteOnSessionDeleted": true
  }
}
```

## Configuration Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | `boolean` | `true` | Enables or disables the plugin. |
| `providers` | `object` | `{ "openai": { "compactModel": "gpt-5.4" } }` | Provider ids to wrap, keyed by OpenCode provider id. |
| `headers` | `object` | see below | Internal header names. |
| `responses` | `object` | see below | Responses endpoint path settings. |
| `compactBodyKeys` | `string[]` | see example | Request body keys copied into compact calls. |
| `summary` | `string` | see example | Synthetic assistant text emitted after compaction. |
| `state` | `object` | see below | SQLite retention and delete behavior. |

### `headers`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `compact` | `string` | `"x-opencode-openai-responses-compact"` | Internal header that marks a compaction request. |
| `session` | `string` | `"x-opencode-openai-responses-compact-session"` | Internal header that carries the OpenCode session id. |

### `providers`

| Field | Type | Description |
| --- | --- | --- |
| `providers.<id>.compactModel` | `string` | Model sent to `/responses/compact` for this provider. |

### `responses`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `endpointPath` | `string` | `"/responses"` | Responses API path suffix to intercept. |
| `compactEndpointPath` | `string` | `"/responses/compact"` | Compact endpoint path used for compaction calls. |

### `state`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `retentionDays` | `integer` | `30` | Number of days to keep checkpoints. |
| `deleteOnSessionDeleted` | `boolean` | `true` | Deletes checkpoints when OpenCode emits `session.deleted`. |

## Star us on Github

<p align="center">
  <a href="https://www.star-history.com/#partment/opencode-openai-compact&type=date&legend=bottom-right">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=partment/opencode-openai-compact&type=date&theme=dark&legend=bottom-right" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=partment/opencode-openai-compact&type=date&legend=bottom-right" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=partment/opencode-openai-compact&type=date&legend=bottom-right" />
    </picture>
  </a>
</p>

## Development

```sh
pnpm install
pnpm run typecheck
pnpm run test
pnpm run build
```
