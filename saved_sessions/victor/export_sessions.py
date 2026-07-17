#!/usr/bin/env python3
"""
Export all Copilot Chat session histories into a readable folder.
Reads the JSONL transcript files and converts them to formatted Markdown.
Also copies the raw JSONL as a backup.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# --- Configuration ---
# Use the EXACT workspace storage from the current session context
WORKSPACE_ID = "40e3249719fc5f686b09b6bea04731d3"
SOURCE_DIR = (
    Path.home()
    / "Library/Application Support/Code/User/workspaceStorage"
    / WORKSPACE_ID
    / "GitHub.copilot-chat"
    / "transcripts"
)

# Output: folder in current workspace (or specify custom)
OUTPUT_DIR = Path.cwd() / "saved_sessions"


def format_timestamp(ts: str) -> str:
    """Convert ISO timestamp to human-readable format."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


# All known event types and how to render them
EVENT_LABELS = {
    "session.start": "🚀 Session Started",
    "user.message": "👤 You",
    "assistant.message": "🤖 Copilot",
    "assistant.turn_start": None,   # skip — just a boundary marker
    "assistant.turn_end": None,
    "tool.execution_start": "🔧 Tool Call",
    "tool.execution_complete": "✅ Tool Result",
}


def extract_conversation(lines: list[dict]) -> list[str]:
    """Extract all meaningful events into readable Markdown."""
    messages = []
    turn_num = 0

    for event in lines:
        etype = event.get("type", "")
        data = event.get("data", {})
        ts = format_timestamp(event.get("timestamp", ""))

        # --- session.start ---
        if etype == "session.start":
            messages.append(f"### 🚀 Session Started — {ts}\n")

        # --- user.message ---
        elif etype == "user.message":
            turn_num += 1
            content = data.get("content", "")
            messages.append(
                f"#### 💬 Turn {turn_num}: You — {ts}\n\n{content}\n"
            )

        # --- assistant.message ---
        elif etype == "assistant.message":
            content = data.get("content", "")
            reasoning = data.get("reasoningText", "")
            tool_requests = data.get("toolRequests", [])

            if content:
                messages.append(
                    f"#### 🤖 Copilot Response — {ts}\n\n{content}\n"
                )

            if reasoning:
                # Truncate very long reasoning for readability
                if len(reasoning) > 3000:
                    reasoning = reasoning[:3000] + "\n\n... (reasoning truncated)"
                messages.append(
                    "<details>\n<summary>🧠 Reasoning</summary>\n\n```\n"
                    f"{reasoning}\n```\n</details>\n"
                )

            if tool_requests:
                parts = []
                for t in tool_requests:
                    name = t.get("name", "?")
                    args = t.get("arguments", "")
                    parts.append(f"- `{name}`")
                    if args:
                        try:
                            args_obj = json.loads(args) if isinstance(args, str) else args
                            # Show filePath or key args briefly
                            short_args = {}
                            for k in ("filePath", "query", "description", "command"):
                                if k in args_obj:
                                    v = args_obj[k]
                                    short_args[k] = v[:80] if isinstance(v, str) else v
                            if short_args:
                                parts.append(f"  {json.dumps(short_args, ensure_ascii=False)}")
                        except (json.JSONDecodeError, TypeError):
                            pass
                messages.append(
                    "<details>\n<summary>🔧 Tool Requests</summary>\n\n"
                    + "\n".join(parts)
                    + "\n</details>\n"
                )

        # --- tool.execution_complete (show results) ---
        elif etype == "tool.execution_complete":
            tool_name = data.get("toolName", data.get("name", "?"))
            result = data.get("result", data.get("output", ""))
            if result:
                result_str = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
                if len(result_str) > 500:
                    result_str = result_str[:500] + "\n... (truncated)"
                messages.append(
                    "<details>\n<summary>📤 Tool Output: `"
                    f"{tool_name}`</summary>\n\n```\n{result_str}\n```\n</details>\n"
                )

    return messages


def export_session(filepath: Path) -> tuple[str, str]:
    """Export a single session JSONL file. Returns (filename, content)."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if not lines:
        return "", ""

    # Get session metadata from first event
    first = lines[0]
    session_id = first.get("data", {}).get("sessionId", filepath.stem)
    start_time = format_timestamp(first.get("timestamp", ""))
    copilot_ver = first.get("data", {}).get("copilotVersion", "?")
    vscode_ver = first.get("data", {}).get("vscodeVersion", "?")

    # Count event types
    from collections import Counter
    ecounts = Counter(e.get("type", "?") for e in lines)

    # Extract conversation
    messages = extract_conversation(lines)

    # Build markdown header
    short_id = session_id[:8]
    user_msgs = ecounts.get("user.message", 0)
    asst_msgs = ecounts.get("assistant.message", 0)
    tool_calls = ecounts.get("tool.execution_start", 0)

    output = [
        f"# Session `{short_id}...`",
        f"**Started:** {start_time}  ",
        f"**VS Code:** {vscode_ver} | **Copilot:** {copilot_ver}  ",
        f"**Session ID:** `{session_id}`  ",
        f"**Summary:** {user_msgs} user messages, {asst_msgs} assistant responses, {tool_calls} tool calls ({len(lines)} total events)  ",
        "",
        "---",
        "",
    ]

    if messages:
        output.extend(messages)
    elif ecounts.get("session.start", 0) == 1 and len(lines) == 1:
        output.append("*This session was started but has no recorded messages.*\n")
    else:
        output.append("*No conversation messages found in this session.*\n")

    # Filename: date + short id
    date_str = start_time.replace(" ", "_").replace(":", "-")[:10]
    filename = f"{date_str}_{short_id}.md"

    return filename, "\n".join(output)


def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: Transcript directory not found: {SOURCE_DIR}", file=sys.stderr)
        print("Try running with a custom path:", file=sys.stderr)
        print("  python export_sessions.py /path/to/transcripts/", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(SOURCE_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No transcript files found.")
        return

    # Clean and recreate output directory
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Also copy raw JSONL files for backup
    raw_dir = OUTPUT_DIR / "_raw_jsonl"
    raw_dir.mkdir(exist_ok=True)

    total_exported = 0
    for fp in jsonl_files:
        # Copy raw file
        import shutil
        shutil.copy2(fp, raw_dir / fp.name)

        # Generate markdown
        filename, content = export_session(fp)
        if not filename:
            print(f"  ⚠️  Skipped empty: {fp.name}")
            continue

        out_path = OUTPUT_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ {filename}")
        total_exported += 1

    print(f"\n✅ Exported {total_exported} sessions (Markdown) + raw JSONL backups")
    print(f"   → {OUTPUT_DIR}")
    print(f"   → Raw backups: {raw_dir}")


if __name__ == "__main__":
    # Allow custom source dir as argument
    if len(sys.argv) > 1:
        SOURCE_DIR = Path(sys.argv[1])
    main()
