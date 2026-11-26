#!/usr/bin/env bash
# Macht .agents/ für Claude Code sichtbar: .claude/skills und .claude/agents als Symlinks.
# .agents/ bleibt die einzige Quelle; .claude/ ist lokal (gitignored) und jederzeit neu erzeugbar.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .claude
for d in skills agents; do
  if [ -e ".claude/$d" ] && [ ! -L ".claude/$d" ]; then
    echo "Abbruch: .claude/$d existiert und ist kein Symlink – bitte manuell prüfen." >&2
    exit 1
  fi
  ln -sfn "../.agents/$d" ".claude/$d"
done
ls -l .claude
