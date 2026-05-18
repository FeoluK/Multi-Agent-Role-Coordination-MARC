#!/bin/bash
# Versioned backup BEFORE changing a file, so we can always roll back.
# Usage: scripts/snapshot.sh <file> "<short reason>"
# Copy is byte-identical (safe restore); metadata in the filename + INDEX.md.
set -e
F="$1"; REASON="${2:-change}"
[ -f "$F" ] || { echo "no such file: $F"; exit 1; }
ROOT="/Users/frozone/Documents/MARC"
SNAP="$ROOT/_snapshots"
mkdir -p "$SNAP"
TS=$(date +%Y%m%d-%H%M%S)
SLUG=$(echo "$REASON" | tr ' /' '__' | tr -cd 'A-Za-z0-9_-' | cut -c1-40)
BASE=$(basename "$F")
REL=${F#"$ROOT"/}
DEST="$SNAP/${BASE}.${TS}.${SLUG}"
cp "$F" "$DEST"
echo "[$TS] $REL  ->  _snapshots/$(basename "$DEST")  (reason: $REASON)" \
  >> "$SNAP/INDEX.md"
echo "snapshot: $DEST"
