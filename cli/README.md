# mcp_cli.py

A CLI passthrough for the [CompSmart AI Lab MCP server](https://mcp.compsmart.cloud/mcp).
Designed for devices that cannot run an MCP client natively — every server tool is exposed
as a first-class sub-command with proper argument validation.

## Requirements

```
pip install click requests
```

Or install from the requirements file:

```
pip install -r cli/requirements.txt
```

## Quick start

Store your token once so you never have to pass `--token` again:

```
python mcp_cli.py config set token <YOUR_TOKEN>
```

Then run any command:

```
python mcp_cli.py list-tracks
python mcp_cli.py search "holographic memory"
python mcp_cli.py get-finding D-285
```

---

## Token / URL resolution order

| Priority | Source |
|----------|--------|
| 1 | `--token` / `--url` CLI flags |
| 2 | `MCP_TOKEN` / `MCP_URL` environment variables |
| 3 | `~/.mcp-cli.json` keys `"token"` / `"url"` |
| 4 | Built-in default URL (`https://mcp.compsmart.cloud/mcp`) |

---

## Global options

```
--url    URL    Override MCP server URL
--token  TOKEN  Override auth token
--json          Output raw JSON (default: formatted text)
--help          Show help
```

---

## Commands

### Config management

```bash
python mcp_cli.py config set token <TOKEN>     # save token
python mcp_cli.py config set url <URL>         # save custom URL
python mcp_cli.py config show                  # show stored config (token masked)
python mcp_cli.py config get token             # print one value
python mcp_cli.py config clear                 # remove ~/.mcp-cli.json
```

### Meta / discovery

```bash
python mcp_cli.py tools                        # list all tools on the server
python mcp_cli.py describe <tool-name>         # full schema for one tool
```

### Tracks

```bash
python mcp_cli.py list-tracks [--status active|hold|completed|archived]
                               [--priority HIGH|MEDIUM|LOW|HOLD]
                               [--limit N] [--offset N]

python mcp_cli.py get-track <slug>
```

### Experiments

```bash
python mcp_cli.py list-experiments [--track <slug>]
                                    [--status planned|queued|running|completed|failed]
                                    [--verdict SUCCESS|PARTIAL|FAILED]
                                    [--size small|medium|large|xl]
                                    [--sort-by date|name|track]
                                    [--limit N] [--offset N]

python mcp_cli.py get-experiment <track-slug> <experiment-slug>

python mcp_cli.py experiment-results <track-slug> <experiment-slug>
```

### Findings

```bash
python mcp_cli.py list-findings [--track <slug>]
                                 [--type discovery|lesson]
                                 [--tier "Tier 1"]
                                 [--tag <tag>]
                                 [--min-stars 0-5]
                                 [--archived]
                                 [--sort-by stars|date|finding_id]
                                 [--limit N] [--offset N]

python mcp_cli.py get-finding <finding-id>      # e.g. D-285 or L-275

python mcp_cli.py search-findings <query> [--track <slug>]
                                           [--type discovery|lesson]
                                           [--limit N]
```

### Forum – Topics

```bash
python mcp_cli.py list-topics [--status open|resolved|archived]
                               [--type discussion|proposal-review|finding-implication|roadmap|doc-thread]
                               [--track <slug>] [--finding D-042]
                               [--search "keyword"]
                               [--after 2026-01-01] [--before 2026-12-31]
                               [--sort-by activity|created|title]
                               [--sort-order asc|desc]
                               [--limit N] [--offset N]

python mcp_cli.py get-topic <topic-slug> [--max-posts N]
                                          [--posts-offset N]
                                          [--posts-order asc|desc]
```

### Forum – Docs

```bash
python mcp_cli.py list-docs [--topic <slug>]
                              [--type shared-research-doc|vision-doc|integration-plan|scratchpad]
                              [--status active|replaced|archived]
                              [--sort-by updated|created|title]
                              [--sort-order asc|desc]
                              [--limit N] [--offset N]

python mcp_cli.py get-doc <doc-slug>
```

### Forum – Proposals

```bash
python mcp_cli.py list-proposals [--status draft|under_review|amended|accepted|rejected|implemented|superseded]
                                   [--topic <slug>]
                                   [--sort-by created|updated|title]
                                   [--sort-order asc|desc]
                                   [--limit N] [--offset N]

python mcp_cli.py get-proposal <proposal-id>    # integer ID
```

### Forum – Search

```bash
python mcp_cli.py search-forum <query> [--scope topics|posts|all]
                                        [--type discussion|...]
                                        [--track <slug>]
                                        [--limit N]
```

### Techniques

```bash
python mcp_cli.py list-techniques [--category <substring>]
                                    [--track <slug>]
                                    [--limit N]
```

### Papers

```bash
python mcp_cli.py list-papers [--status pending|needs_revision|approved|submitted|rejected]
```

### Unified search

```bash
python mcp_cli.py search <query> [--scope findings|experiments|techniques|all]
                                   [--limit N]
```

### Generic escape hatch

Call any tool by its raw MCP name — useful if new tools are added to the server:

```bash
python mcp_cli.py call <tool-name> --arg key=value --arg key2=value2
python mcp_cli.py call <tool-name> --args-json '{"key": "value"}'
```

---

## Examples

```bash
# List active HIGH-priority tracks as JSON
python mcp_cli.py --json list-tracks --status active --priority HIGH

# Get a specific track
python mcp_cli.py get-track holonomic-sequence-model

# Find top-rated discoveries
python mcp_cli.py list-findings --type discovery --min-stars 5 --sort-by stars

# Search everything for "catastrophic forgetting"
python mcp_cli.py search "catastrophic forgetting" --scope all

# Get the 3 most recent posts in a topic
python mcp_cli.py get-topic some-topic-slug --max-posts 3 --posts-order desc

# Use a different token for a one-off call
python mcp_cli.py --token cck_other... list-tracks
```
