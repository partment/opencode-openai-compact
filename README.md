# OpenAI Native Compaction for OpenCode

[![npm version](https://img.shields.io/npm/v/opencode-openai-compact?style=flat-square)](https://www.npmjs.com/package/opencode-openai-compact)
[![GitHub stars](https://img.shields.io/github/stars/partment/opencode-openai-compact?style=flat-square)](https://github.com/partment/opencode-openai-compact/stargazers)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

Use OpenAI's official Responses API `/responses/compact` in OpenCode.

OpenCode can compact long coding sessions. When you are using OpenAI Responses models, this plugin routes compaction through OpenAI's native compact endpoint instead of asking another model to write a text summary.

## Why Native Compaction

| Default prompt summary | OpenAI native compaction |
| --- | --- |
| Generates a plain text summary | Returns an encrypted `compaction` item |
| Can miss tool or reasoning state | Built for the Responses API state model |
| App-owned summary format | Official `/responses/compact` output |
| You decide what to keep | OpenAI returns the next compacted window |

The important part is simple: `/responses/compact` returns compacted output that should be passed to the next `/responses` request as-is. This plugin makes OpenCode do that for OpenAI providers.

## What It Does

1. Intercepts OpenCode session compaction for configured OpenAI providers.
2. Sends the current Responses input window to `/responses/compact`.
3. Removes OpenCode's internal summary prompt from the compact request body.
4. Stores the compacted output in a local SQLite checkpoint.
5. Injects that checkpoint into the next `/responses` request for the same session.

## When To Use It

Use this if:

- You use OpenAI Responses API models in OpenCode.
- You run long coding sessions that hit compaction.
- You want OpenAI's official compaction item instead of a custom text summary.

Skip it if:

- You do not use OpenAI Responses API providers.
- You prefer OpenCode's default prompt-based summary.
- Your sessions are short enough that compaction does not matter.

## Install

Add the npm package to your OpenCode config.

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

Requirements:

| Runtime | Version |
| --- | --- |
| Node.js | `>=22.12.0` |
| OpenCode | `>=1.3.8` |

## Configuration Files

Most users do not need plugin-specific configuration.

Create `openai-compact.json` or `openai-compact.jsonc` only when you want to override defaults. Later layers override earlier layers. Within the same directory, `openai-compact.jsonc` is read after `openai-compact.json` and can override it.

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

The default retention is 30 days. Checkpoints are deleted when OpenCode emits `session.deleted`.

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

## Star Us On GitHub

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
