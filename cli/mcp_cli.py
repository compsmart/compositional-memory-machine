#!/usr/bin/env python3
"""
mcp_cli.py – CLI passthrough for the CompSmart AI Lab MCP server.

Token / URL resolution order (highest to lowest priority):
  1. CLI flag  --token / --url
  2. Env var   MCP_TOKEN / MCP_URL
  3. Config    ~/.mcp-cli.json  keys "token" / "url"
  4. Built-in  default URL below

Quick-start:
  python mcp_cli.py config set token <your-token>
  python mcp_cli.py list-tracks
  python mcp_cli.py search --query "holographic memory"
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import click
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_URL = "https://mcp.compsmart.cloud/mcp"
CONFIG_PATH = Path.home() / ".mcp-cli.json"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def _resolve(cli_val: Optional[str], env_var: str, cfg_key: str, default: Optional[str] = None) -> Optional[str]:
    if cli_val:
        return cli_val
    env = os.environ.get(env_var)
    if env:
        return env
    return _load_config().get(cfg_key, default)


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------


class MCPError(click.ClickException):
    pass


class MCPClient:
    """Minimal MCP Streamable-HTTP client (JSON-RPC 2.0)."""

    def __init__(self, url: str, token: Optional[str]):
        self.url = url.rstrip("/")
        self.token = token
        self.session_id: Optional[str] = None
        self._rid = 0

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._rid += 1
        return self._rid

    def _post(self, body: dict, timeout: int = 30) -> Any:
        params = {"token": self.token} if self.token else {}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        try:
            resp = requests.post(
                self.url, params=params, headers=headers, json=body, timeout=timeout
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise MCPError(f"Cannot connect to {self.url}: {exc}")
        except requests.exceptions.HTTPError as exc:
            raise MCPError(f"HTTP {resp.status_code} from server: {exc}")
        except requests.exceptions.Timeout:
            raise MCPError("Request timed out")

        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid

        ct = resp.headers.get("Content-Type", "")
        if "text/event-stream" in ct:
            return _parse_sse(resp.text)
        return resp.json()

    # ------------------------------------------------------------------
    # public MCP methods
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-cli", "version": "1.0.0"},
            },
        }
        result = self._post(payload)
        if isinstance(result, dict) and "error" in result:
            raise MCPError(f"Initialize failed: {result['error']}")
        # Fire-and-forget initialized notification
        try:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        except Exception:
            pass

    def list_tools(self) -> list[dict]:
        resp = self._post({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list", "params": {}})
        if isinstance(resp, dict) and "error" in resp:
            raise MCPError(f"tools/list failed: {resp['error']}")
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict, timeout: int = 60) -> Any:
        resp = self._post(
            {"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call",
             "params": {"name": name, "arguments": arguments}},
            timeout=timeout,
        )
        if isinstance(resp, dict) and "error" in resp:
            raise MCPError(f"Tool '{name}' error: {resp['error'].get('message', resp['error'])}")
        return resp.get("result", resp) if isinstance(resp, dict) else resp


def _parse_sse(text: str) -> Any:
    result = None
    for line in text.splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                try:
                    result = json.loads(data)
                except json.JSONDecodeError:
                    pass
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _out(obj: Any, as_json: bool) -> None:
    """Render an MCP tool result."""
    if as_json:
        click.echo(json.dumps(obj, indent=2))
        return
    _pretty(obj)


def _pretty(result: Any) -> None:
    if result is None:
        click.echo("(no result)")
        return
    if isinstance(result, dict):
        content = result.get("content")
        is_error = result.get("isError", False)
        if content is not None:
            if is_error:
                click.secho("Tool returned an error:", fg="red", err=True)
            items = content if isinstance(content, list) else [content]
            for item in items:
                if isinstance(item, dict):
                    itype = item.get("type", "text")
                    if itype == "text":
                        click.echo(item.get("text", ""))
                    elif itype == "image":
                        click.echo(f"[image/{item.get('mimeType', '?')}] (binary omitted)")
                    elif itype == "resource":
                        click.echo(f"[resource: {item.get('uri', '')}]")
                    else:
                        click.echo(json.dumps(item, indent=2))
                else:
                    click.echo(str(item))
            return
    click.echo(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Shared Click context
# ---------------------------------------------------------------------------


class Ctx:
    def __init__(self, url: str, token: Optional[str], as_json: bool):
        self.client = MCPClient(url, token)
        self.as_json = as_json

    def connect(self) -> None:
        self.client.initialize()

    def call(self, tool: str, **kwargs: Any) -> None:
        args = {k: v for k, v in kwargs.items() if v is not None}
        result = self.client.call_tool(tool, args)
        _out(result, self.as_json)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--url", envvar="MCP_URL", default=None, metavar="URL",
              help="MCP server URL  [env: MCP_URL]")
@click.option("--token", envvar="MCP_TOKEN", default=None, metavar="TOKEN",
              help="Auth token  [env: MCP_TOKEN]")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output raw JSON instead of formatted text")
@click.pass_context
def cli(ctx: click.Context, url: Optional[str], token: Optional[str], as_json: bool):
    """CompSmart AI Lab – MCP CLI passthrough.

    \b
    Token / URL precedence: --flag > env var > ~/.mcp-cli.json > built-in default
    Run `config set token <TOKEN>` once to avoid passing --token every time.
    """
    ctx.ensure_object(dict)
    ctx.obj = Ctx(
        url=_resolve(url, "MCP_URL", "url", DEFAULT_URL),
        token=_resolve(token, "MCP_TOKEN", "token"),
        as_json=as_json,
    )


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@cli.group("config")
def config_group():
    """Manage persistent config stored in ~/.mcp-cli.json."""


@config_group.command("set")
@click.argument("key")
@click.argument("value")
def cfg_set(key: str, value: str):
    """Set a config key (url, token, ...)."""
    cfg = _load_config()
    cfg[key] = value
    _save_config(cfg)
    click.echo(f"Saved '{key}' to {CONFIG_PATH}")


@config_group.command("get")
@click.argument("key")
def cfg_get(key: str):
    """Print a stored config value."""
    cfg = _load_config()
    if key not in cfg:
        raise click.ClickException(f"Key '{key}' not found")
    click.echo(cfg[key])


@config_group.command("show")
def cfg_show():
    """Show all config values (token is masked)."""
    cfg = _load_config()
    if not cfg:
        click.echo("No config stored. Run: config set token <TOKEN>")
        return
    for k, v in cfg.items():
        display = (v[:4] + "****" + v[-4:]) if k == "token" and len(v) > 8 else v
        click.echo(f"  {k}: {display}")


@config_group.command("clear")
def cfg_clear():
    """Delete all stored config."""
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
        click.echo(f"Removed {CONFIG_PATH}")
    else:
        click.echo("No config file found.")


# ---------------------------------------------------------------------------
# tools / describe  (meta)
# ---------------------------------------------------------------------------


@cli.command("tools")
@click.pass_obj
def list_tools_cmd(obj: Ctx):
    """List all tools available on the MCP server."""
    obj.connect()
    tools = obj.client.list_tools()
    if not tools:
        click.echo("Server returned no tools.")
        return
    if obj.as_json:
        click.echo(json.dumps(tools, indent=2))
        return
    click.echo(f"\n{'Tool':<30} Description")
    click.echo("-" * 72)
    for t in tools:
        name = t.get("name", "?")
        desc = textwrap.shorten(t.get("description", ""), width=40, placeholder="...")
        click.echo(f"  {name:<28} {desc}")
    click.echo()


@cli.command("describe")
@click.argument("tool_name")
@click.pass_obj
def describe_cmd(obj: Ctx, tool_name: str):
    """Show full schema for TOOL_NAME."""
    obj.connect()
    tools = obj.client.list_tools()
    tool = next((t for t in tools if t.get("name") == tool_name), None)
    if tool is None:
        names = [t.get("name") for t in tools]
        raise click.ClickException(f"'{tool_name}' not found.\nAvailable: {', '.join(names)}")
    if obj.as_json:
        click.echo(json.dumps(tool, indent=2))
        return
    click.echo(f"\nTool: {tool['name']}")
    click.echo(f"Description: {tool.get('description', '')}")
    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {})
    req = set(schema.get("required", []))
    if props:
        click.echo("\nArguments:")
        for pname, ps in props.items():
            r = "*" if pname in req else " "
            ptype = ps.get("type", "any")
            pdesc = ps.get("description", "")
            enums = ps.get("enum")
            enum_str = f"  options: {enums}" if enums else ""
            click.echo(f"  {r} --{pname:<22} ({ptype}){enum_str}")
            if pdesc:
                click.echo(f"    {pdesc}")
        click.echo("\n  * = required")
    click.echo()


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


@cli.command("list-tracks")
@click.option("--status", type=click.Choice(["active", "hold", "completed", "archived"]))
@click.option("--priority", type=click.Choice(["HIGH", "MEDIUM", "LOW", "HOLD"]))
@click.option("--limit", type=int, default=None, help="Max results (default 50)")
@click.option("--offset", type=int, default=None)
@click.pass_obj
def list_tracks(obj: Ctx, status, priority, limit, offset):
    """List research tracks."""
    obj.connect()
    obj.call("list_tracks", status=status, priority=priority, limit=limit, offset=offset)


@cli.command("get-track")
@click.argument("slug")
@click.pass_obj
def get_track(obj: Ctx, slug: str):
    """Get full details of a research track by SLUG."""
    obj.connect()
    obj.call("get_track", slug=slug)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@cli.command("list-experiments")
@click.option("--track", "track_slug", default=None, help="Filter by track slug")
@click.option("--status", type=click.Choice(["planned", "queued", "running", "completed", "failed"]))
@click.option("--verdict", type=click.Choice(["SUCCESS", "PARTIAL", "FAILED"]))
@click.option("--size", type=click.Choice(["small", "medium", "large", "xl"]))
@click.option("--sort-by", "sort_by", type=click.Choice(["date", "name", "track"]))
@click.option("--limit", type=int, default=None, help="Max results (default 20)")
@click.option("--offset", type=int, default=None)
@click.pass_obj
def list_experiments(obj: Ctx, track_slug, status, verdict, size, sort_by, limit, offset):
    """List experiments with optional filters."""
    obj.connect()
    obj.call("list_experiments",
             track_slug=track_slug, status=status, verdict=verdict,
             size=size, sort_by=sort_by, limit=limit, offset=offset)


@cli.command("get-experiment")
@click.argument("track_slug")
@click.argument("experiment_slug")
@click.pass_obj
def get_experiment(obj: Ctx, track_slug: str, experiment_slug: str):
    """Get full experiment detail (hypothesis, methodology, conclusion)."""
    obj.connect()
    obj.call("get_experiment", track_slug=track_slug, experiment_slug=experiment_slug)


@cli.command("experiment-results")
@click.argument("track_slug")
@click.argument("experiment_slug")
@click.pass_obj
def get_experiment_results(obj: Ctx, track_slug: str, experiment_slug: str):
    """Get results_json and metrics_summary for an experiment."""
    obj.connect()
    obj.call("get_experiment_results", track_slug=track_slug, experiment_slug=experiment_slug)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@cli.command("list-findings")
@click.option("--track", "track_slug", default=None)
@click.option("--type", "type_", type=click.Choice(["discovery", "lesson"]))
@click.option("--tier", default=None, help="Tier substring, e.g. 'Tier 1'")
@click.option("--tag", default=None)
@click.option("--min-stars", "min_stars", type=int, default=None, help="0–5")
@click.option("--archived", "is_active", flag_value=False, default=None,
              help="Show archived findings instead of active")
@click.option("--sort-by", "sort_by", type=click.Choice(["stars", "date", "finding_id"]))
@click.option("--limit", type=int, default=None)
@click.option("--offset", type=int, default=None)
@click.pass_obj
def list_findings(obj: Ctx, track_slug, type_, tier, tag, min_stars, is_active, sort_by, limit, offset):
    """List findings (discoveries D-XXX and lessons L-XXX)."""
    obj.connect()
    # Default is_active=True unless --archived
    effective_active = False if is_active is False else True
    obj.call("list_findings",
             track_slug=track_slug, type=type_, tier=tier, tag=tag,
             min_stars=min_stars, is_active=effective_active,
             sort_by=sort_by, limit=limit, offset=offset)


@cli.command("get-finding")
@click.argument("finding_id")
@click.pass_obj
def get_finding(obj: Ctx, finding_id: str):
    """Get full finding detail.  FINDING_ID e.g. D-285 or L-275."""
    obj.connect()
    obj.call("get_finding", finding_id=finding_id)


@cli.command("search-findings")
@click.argument("query")
@click.option("--track", "track_slug", default=None)
@click.option("--type", "type_", type=click.Choice(["discovery", "lesson"]))
@click.option("--limit", type=int, default=None)
@click.pass_obj
def search_findings(obj: Ctx, query: str, track_slug, type_, limit):
    """Full-text search across finding titles and descriptions."""
    obj.connect()
    obj.call("search_findings", query=query, track_slug=track_slug, type=type_, limit=limit)


# ---------------------------------------------------------------------------
# Forum – Topics
# ---------------------------------------------------------------------------


@cli.command("list-topics")
@click.option("--status", type=click.Choice(["open", "resolved", "archived"]))
@click.option("--type", "topic_type",
              type=click.Choice(["discussion", "proposal-review", "finding-implication", "roadmap", "doc-thread"]))
@click.option("--track", "linked_track_slug", default=None)
@click.option("--finding", "linked_finding_id", default=None, help="e.g. D-042")
@click.option("--search", default=None, help="Keyword search in title/summary")
@click.option("--after", "created_after", default=None, help="ISO 8601 date")
@click.option("--before", "created_before", default=None, help="ISO 8601 date")
@click.option("--sort-by", "sort_by", type=click.Choice(["activity", "created", "title"]))
@click.option("--sort-order", "sort_order", type=click.Choice(["asc", "desc"]))
@click.option("--limit", type=int, default=None)
@click.option("--offset", type=int, default=None)
@click.pass_obj
def list_topics(obj: Ctx, status, topic_type, linked_track_slug, linked_finding_id,
                search, created_after, created_before, sort_by, sort_order, limit, offset):
    """List forum topics."""
    obj.connect()
    obj.call("list_topics",
             status=status, topic_type=topic_type,
             linked_track_slug=linked_track_slug, linked_finding_id=linked_finding_id,
             search=search, created_after=created_after, created_before=created_before,
             sort_by=sort_by, sort_order=sort_order, limit=limit, offset=offset)


@cli.command("get-topic")
@click.argument("topic_slug")
@click.option("--max-posts", "max_posts", type=int, default=None)
@click.option("--posts-offset", "posts_offset", type=int, default=None)
@click.option("--posts-order", "posts_order", type=click.Choice(["asc", "desc"]))
@click.pass_obj
def get_topic(obj: Ctx, topic_slug: str, max_posts, posts_offset, posts_order):
    """Get full forum topic with posts."""
    obj.connect()
    obj.call("get_topic", topic_slug=topic_slug,
             max_posts=max_posts, posts_offset=posts_offset, posts_order=posts_order)


# ---------------------------------------------------------------------------
# Forum – Docs
# ---------------------------------------------------------------------------


@cli.command("list-docs")
@click.option("--topic", "topic_slug", default=None)
@click.option("--type", "doc_type",
              type=click.Choice(["shared-research-doc", "vision-doc", "integration-plan", "scratchpad"]))
@click.option("--status", type=click.Choice(["active", "replaced", "archived"]))
@click.option("--sort-by", "sort_by", type=click.Choice(["updated", "created", "title"]))
@click.option("--sort-order", "sort_order", type=click.Choice(["asc", "desc"]))
@click.option("--limit", type=int, default=None)
@click.option("--offset", type=int, default=None)
@click.pass_obj
def list_docs(obj: Ctx, topic_slug, doc_type, status, sort_by, sort_order, limit, offset):
    """List forum docs."""
    obj.connect()
    obj.call("list_docs", topic_slug=topic_slug, doc_type=doc_type, status=status,
             sort_by=sort_by, sort_order=sort_order, limit=limit, offset=offset)


@cli.command("get-doc")
@click.argument("doc_slug")
@click.pass_obj
def get_doc(obj: Ctx, doc_slug: str):
    """Get a doc with its current revision content (markdown)."""
    obj.connect()
    obj.call("get_doc", doc_slug=doc_slug)


# ---------------------------------------------------------------------------
# Forum – Proposals
# ---------------------------------------------------------------------------


@cli.command("list-proposals")
@click.option("--status",
              type=click.Choice(["draft", "under_review", "amended", "accepted", "rejected", "implemented", "superseded"]))
@click.option("--topic", "topic_slug", default=None)
@click.option("--sort-by", "sort_by", type=click.Choice(["created", "updated", "title"]))
@click.option("--sort-order", "sort_order", type=click.Choice(["asc", "desc"]))
@click.option("--limit", type=int, default=None)
@click.option("--offset", type=int, default=None)
@click.pass_obj
def list_proposals(obj: Ctx, status, topic_slug, sort_by, sort_order, limit, offset):
    """List forum proposals."""
    obj.connect()
    obj.call("list_proposals", status=status, topic_slug=topic_slug,
             sort_by=sort_by, sort_order=sort_order, limit=limit, offset=offset)


@cli.command("get-proposal")
@click.argument("proposal_id", type=int)
@click.pass_obj
def get_proposal(obj: Ctx, proposal_id: int):
    """Get full proposal with votes and integration details. PROPOSAL_ID is an integer."""
    obj.connect()
    obj.call("get_proposal", proposal_id=proposal_id)


# ---------------------------------------------------------------------------
# Forum – Search
# ---------------------------------------------------------------------------


@cli.command("search-forum")
@click.argument("query")
@click.option("--scope", type=click.Choice(["topics", "posts", "all"]), default=None)
@click.option("--type", "topic_type",
              type=click.Choice(["discussion", "proposal-review", "finding-implication", "roadmap", "doc-thread"]))
@click.option("--track", "linked_track_slug", default=None)
@click.option("--limit", type=int, default=None)
@click.pass_obj
def search_forum(obj: Ctx, query: str, scope, topic_type, linked_track_slug, limit):
    """Keyword search across forum topics and posts."""
    obj.connect()
    obj.call("search_forum", query=query, scope=scope, topic_type=topic_type,
             linked_track_slug=linked_track_slug, limit=limit)


# ---------------------------------------------------------------------------
# Techniques
# ---------------------------------------------------------------------------


@cli.command("list-techniques")
@click.option("--category", default=None, help="Filter by category substring")
@click.option("--track", "applicable_to_track", default=None, help="Filter by applicable track slug")
@click.option("--limit", type=int, default=None)
@click.pass_obj
def list_techniques(obj: Ctx, category, applicable_to_track, limit):
    """List proven techniques from the knowledge base."""
    obj.connect()
    obj.call("list_techniques", category=category, applicable_to_track=applicable_to_track, limit=limit)


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------


@cli.command("list-papers")
@click.option("--status",
              type=click.Choice(["pending", "needs_revision", "approved", "submitted", "rejected"]))
@click.pass_obj
def list_papers(obj: Ctx, status):
    """List all papers with track info and cited finding count."""
    obj.connect()
    obj.call("list_papers", status=status)


# ---------------------------------------------------------------------------
# Unified search
# ---------------------------------------------------------------------------


@cli.command("search")
@click.argument("query")
@click.option("--scope", type=click.Choice(["findings", "experiments", "techniques", "all"]), default=None,
              help="Restrict search scope (default: all)")
@click.option("--limit", type=int, default=None)
@click.pass_obj
def search_cmd(obj: Ctx, query: str, scope, limit):
    """Unified full-text search across findings, experiments, and techniques."""
    obj.connect()
    obj.call("search", query=query, scope=scope, limit=limit)


# ---------------------------------------------------------------------------
# Generic call (escape hatch for any tool by name)
# ---------------------------------------------------------------------------


@cli.command("call")
@click.argument("tool_name")
@click.option("--arg", "-a", multiple=True, metavar="KEY=VALUE",
              help="Argument as KEY=VALUE (repeatable). JSON values are auto-parsed.")
@click.option("--args-json", default=None, metavar="JSON",
              help="All arguments as a JSON object string.")
@click.pass_obj
def call_cmd(obj: Ctx, tool_name: str, arg: tuple, args_json: Optional[str]):
    """Call any MCP tool by name (generic escape hatch).

    \b
    Examples:
      mcp_cli.py call get_track --arg slug=graph-memory
      mcp_cli.py call search --args-json '{"query":"HRR","scope":"findings"}'
    """
    if args_json:
        try:
            arguments = json.loads(args_json)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON: {exc}")
    else:
        arguments = {}
        for pair in arg:
            if "=" not in pair:
                raise click.ClickException(f"Expected KEY=VALUE, got: {pair!r}")
            k, _, v = pair.partition("=")
            try:
                arguments[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                arguments[k] = v

    obj.connect()
    result = obj.client.call_tool(tool_name, arguments)
    _out(result, obj.as_json)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
