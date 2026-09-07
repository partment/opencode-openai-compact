# OpenAI Native Compaction for OpenCode

[![npm version](https://img.shields.io/npm/v/opencode-openai-compact?style=flat-square)](https://www.npmjs.com/package/opencode-openai-compact)
[![GitHub stars](https://img.shields.io/github/stars/partment/opencode-openai-compact?style=flat-square)](https://github.com/partment/opencode-openai-compact/stargazers)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

Use OpenAI's official Responses API compaction v2 in OpenCode.

OpenCode can compact long coding sessions. When you are using OpenAI Responses models, this plugin sends a normal `/responses` request whose final input item is `compaction_trigger`, instead of asking another model to write a text summary.

![Hero image introducing this plugin.](/assets/images/hero.webp)

## Why Native Compaction

| Default prompt summary | OpenAI native compaction |
| --- | --- |
| Generates a plain text summary | Returns an encrypted `compaction` item |
| Can miss tool or reasoning state | Built for the Responses API state model |
| App-owned summary format | Official Responses compaction output |
| You decide what to keep | OpenAI returns the next compacted window |

The important part is simple: compaction v2 returns an encrypted `compaction` item that should be passed to the next `/responses` request. This plugin makes OpenCode do that for OpenAI providers.

## What It Does

1. Intercepts OpenCode session compaction for configured OpenAI providers.
2. Sends the current Responses input window to `/responses` with a final `compaction_trigger` item.
3. Removes OpenCode's internal summary prompt from the compact request body.
4. Stores the compacted output in a local SQLite checkpoint.
5. Injects that checkpoint into the next `/responses` request for the same session.
6. Replaces OpenCode's synthetic post-compaction exchange with the active checkpoint while preserving later real messages.

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
| OpenCode | `>=1.18.7` (Tested with 1.18.23) |

## Configuration Files

Most users do not need plugin-specific configuration.

Create `openai-compact.json` or `openai-compact.jsonc` only when you want to override defaults. Later layers override earlier layers. Within the same directory, `openai-compact.jsonc` is read after `openai-compact.json` and can override it.

Read order:

1. Built-in defaults.
2. Global OpenCode config directory: `openai-compact.json`, then `openai-compact.jsonc`.
3. Directory from `OPENCODE_CONFIG_DIR`: `openai-compact.json`, then `openai-compact.jsonc`.
4. Nearest project `.opencode` directory found by walking upward from the current directory: `openai-compact.json`, then `openai-compact.jsonc`.

Exception: `state.retentionDays` is read only from layer 2 because all projects share one global database. Values in layers 3 and 4 are ignored with a warning.

Global OpenCode config directory:

- `$XDG_CONFIG_HOME/opencode`, when `XDG_CONFIG_HOME` is set.
- `~/.config/opencode`, when `XDG_CONFIG_HOME` is not set.

If neither `openai-compact.json` nor `openai-compact.jsonc` exists in the global OpenCode config directory, the plugin creates an empty `openai-compact.jsonc` file on first load.

## State Database

Runtime checkpoints are stored in SQLite at:

```text
~/.config/opencode/openai-compact/checkpoints.db
```

The database stores checkpoints and the message IDs of OpenCode's internal post-compaction control turns. Tracking IDs keeps those turns out of future requests even if OpenCode changes their text, metadata, or position.

When a forked OpenCode session first runs after compaction, checkpoints and control turns before the fork point are copied to the new session with their regenerated message IDs.

The default retention is 30 days. Retention is a global maintenance policy for this shared database, not a per-project setting. Set `state.retentionDays` only in the fixed global OpenCode config directory. Values from `OPENCODE_CONFIG_DIR` or project `.opencode` files are ignored with a warning so one project cannot prune another project's checkpoints.

## Example Configuration

```jsonc
{
  "$schema": "https://raw.githubusercontent.com/partment/opencode-openai-compact/main/configSchema.json",
  "enabled": true,
  "providers": {
    "openai": {
      "enabled": true,
      "compactModel": null,
      "compactReasoningEffort": null
    }
  },
  "headers": {
    "compact": "x-opencode-openai-responses-compact",
    "session": "x-opencode-openai-responses-compact-session"
  },
  "responses": {
    "endpointPath": "/responses"
  },
  "compactBodyKeys": [
    "input",
    "instructions",
    "tools",
    "parallel_tool_calls",
    "reasoning",
    "service_tier",
    "prompt_cache_key",
    "text"
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
| `providers` | `object` | `{ "openai": { "enabled": true, "compactModel": null, "compactReasoningEffort": null } }` | Provider ids to wrap, keyed by OpenCode provider id. Each provider can be explicitly enabled or disabled. |
| `headers` | `object` | see below | Internal header names. |
| `responses` | `object` | see below | Responses endpoint path settings. |
| `compactBodyKeys` | `string[]` | see example | Request body keys copied into compact calls. |
| `summary` | `string` | see example | Synthetic assistant text emitted after compaction. |
| `state` | `object` | see below | SQLite retention and delete behavior. |

### `headers`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `compact` | `string` | `"x-opencode-openai-responses-compact"` | Internal header that marks a compaction request. Header names are normalized to lowercase and must be valid token names. |
| `session` | `string` | `"x-opencode-openai-responses-compact-session"` | Internal header that carries the OpenCode session id. It must differ from `compact` and cannot use reserved authentication, content, connection, or routing names. |

### `providers`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `providers.<id>.enabled` | `boolean` | `true` | Enables or disables compaction wrapping for this provider. Set `providers.openai.enabled` to `false` to leave the built-in OpenAI provider untouched. |
| `providers.<id>.compactModel` | `string \| null` | `null` | Model sent to the `/responses` compaction v2 request. `null` follows the model used by the latest conversation part for both automatic and manual compaction. |
| `providers.<id>.compactReasoningEffort` | `"none" \| "minimal" \| "low" \| "medium" \| "high" \| "xhigh" \| "max" \| null` | `null` | Reasoning effort sent to compaction. `null` follows the current conversation selection. Supported levels vary by model. |

**Priority: explicit setting > latest conversation part > OpenCode's original compaction request.**

`null` uses the next available value in that order.

Provider settings are deep-merged across configuration layers. Adding a custom provider does not remove the default `openai` entry. To use only a custom provider:

```jsonc
{
  "providers": {
    "openai": { "enabled": false },
    "custom-openai": { "enabled": true }
  }
}
```

### `responses`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `endpointPath` | `string` | `"/responses"` | Responses API path suffix to intercept. Whitespace-only values and `/` are rejected; values are trimmed, prefixed with `/`, and stripped of trailing slashes. |
| `compactEndpointPath` | `string` | `"/responses/compact"` | Deprecated and ignored. Retained only so existing configuration files continue to load. |

### `state`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `retentionDays` | `integer` | `30` | Global number of days to keep checkpoints in the shared database. Only the fixed global config directory may override it; non-global values are ignored with a warning. |
| `deleteOnSessionDeleted` | `boolean` | `true` | Deletes stored rows on `session.deleted`. When `false`, rows remain but are tombstoned and never used again. |

### Configuration diagnostics

Set `OPENCODE_OPENAI_COMPACT_DEBUG=1` to print an effective-configuration summary at startup. The summary includes existing configuration files, fields overridden by each layer, enabled providers, endpoint and header settings, the shared database path, the global retention scope, ignored retention overrides, and deprecated fields. It does not print authentication tokens.

Unknown fields in nested objects are rejected instead of silently ignored. JSONC syntax errors include the source file, line, column, and parser error.

## Star Us On GitHub

<p align="center">
  <a href="https://www.star-history.com/?type=date&legend=bottom-right&repos=partment%2Fopencode-openai-compact">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=partment/opencode-openai-compact&type=date&theme=dark&legend=bottom-right&sealed_token=B1P0xeMCKz6n5QMyiC8UosMAD2wW5_RlxtLF4oF9jDSlwYyHsd1Mk59iIJ8dpCmuECo1AbQkF4H48Lq2fDdJqwqq4TOCVRlmMCiBDct6vXpJ7zcy6qeT1v1KlcNeI7MlZF8h9x-FNoDp4Dl8Ve3bJTQhGn8G9VQT7ANV4ZCaVj7u2XVeiudddp-9LCC9" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=partment/opencode-openai-compact&type=date&legend=bottom-right&sealed_token=B1P0xeMCKz6n5QMyiC8UosMAD2wW5_RlxtLF4oF9jDSlwYyHsd1Mk59iIJ8dpCmuECo1AbQkF4H48Lq2fDdJqwqq4TOCVRlmMCiBDct6vXpJ7zcy6qeT1v1KlcNeI7MlZF8h9x-FNoDp4Dl8Ve3bJTQhGn8G9VQT7ANV4ZCaVj7u2XVeiudddp-9LCC9" />
      <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=partment/opencode-openai-compact&type=date&legend=bottom-right&sealed_token=B1P0xeMCKz6n5QMyiC8UosMAD2wW5_RlxtLF4oF9jDSlwYyHsd1Mk59iIJ8dpCmuECo1AbQkF4H48Lq2fDdJqwqq4TOCVRlmMCiBDct6vXpJ7zcy6qeT1v1KlcNeI7MlZF8h9x-FNoDp4Dl8Ve3bJTQhGn8G9VQT7ANV4ZCaVj7u2XVeiudddp-9LCC9" />
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
