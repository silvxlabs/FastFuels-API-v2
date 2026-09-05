#!/usr/bin/env bash
# commit-msg hook: reject AI-tool attribution in commit messages.
#
# CONTRIBUTING.md / CLAUDE.md: GitHub artifacts are human-authored; the author
# signal is the GitHub account, not the tool. Usage: check_commit_message.sh <file>
set -euo pipefail

msg_file="${1:?usage: $0 <commit-msg-file>}"

# One regex per forbidden line shape (case-insensitive, extended regex).
patterns=(
  'Co-Authored-By:.*(claude|anthropic|codex|openai|copilot|gemini|cursor|aider|devin)'
  'Generated with \[?(Claude Code|Codex|Copilot|Gemini|Cursor|Aider)'
  'claude\.ai/code/session'
  '^Claude-Session:'
)

# Ignore git's comment lines and anything after a scissors line.
body=$(sed -e '/^# ------------------------ >8 ------------------------$/,$d' -e '/^#/d' "$msg_file")

hits=""
for p in "${patterns[@]}"; do
  found=$(printf '%s\n' "$body" | grep -inE -- "$p" || true)
  [ -n "$found" ] && hits+="$found"$'\n'
done

if [ -n "$hits" ]; then
  {
    echo "commit rejected: AI attribution found in the commit message:"
    printf '%s' "$hits" | sort -t: -k1,1n -u | sed 's/^/  line /'
    echo
    echo "Remove the trailer(s) and retry. Claude Code: set attribution.commit=\"\" in settings.json."
  } >&2
  exit 1
fi
