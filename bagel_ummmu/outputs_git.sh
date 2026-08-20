#!/usr/bin/env bash
# Persist Uni-MMMU sampling outputs to a GitHub branch, so that molab session
# restarts (which wipe .git dirs, images and other binaries) lose nothing.
#
# The outputs directory ($WORK_DIR/Uni-MMMU/outputs) is made a standalone git
# repo whose 'outputs' branch lives in your GitHub repo, separate from main.
#
# Usage:
#   bash outputs_git.sh restore   # session start: pull the outputs branch back
#   bash outputs_git.sh push      # commit current outputs and push (periodic)
#
# Required environment:
#   GITHUB_TOKEN   PAT with write access to the repo (never commit this!)
#   OUTPUTS_REPO   owner/repo, e.g. MrTractorWheel/zebra_Cot-BAGEL-UniMMMU-test
# Optional:
#   OUTPUTS_BRANCH (default: outputs)
#   WORK_DIR       (default: $HOME/bagel_ummmu)
set -euo pipefail

CMD="${1:?usage: outputs_git.sh restore|push}"
WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/Uni-MMMU/outputs}"
BRANCH="${OUTPUTS_BRANCH:-outputs}"

: "${GITHUB_TOKEN:?set GITHUB_TOKEN (a PAT with write access to the repo)}"
: "${OUTPUTS_REPO:?set OUTPUTS_REPO, e.g. youruser/zebra_Cot-BAGEL-UniMMMU-test}"
URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${OUTPUTS_REPO}.git"

g() { git -C "$OUT_DIR" "$@"; }

ensure_repo() {
  mkdir -p "$OUT_DIR"
  if [ ! -d "$OUT_DIR/.git" ]; then
    g init --initial-branch "$BRANCH" >/dev/null 2>&1 \
      || { g init >/dev/null; g checkout -b "$BRANCH" >/dev/null 2>&1 || true; }
  fi
  g remote set-url origin "$URL" 2>/dev/null || g remote add origin "$URL"
  g config user.email "molab-runner@localhost"
  g config user.name "molab runner"
}

restore() {
  ensure_repo
  if g fetch origin "$BRANCH" 2>/dev/null; then
    # Remote is the source of truth at session start (local state after a
    # restart is a partial skeleton). -f overwrites conflicting local files.
    g checkout -f -B "$BRANCH" FETCH_HEAD
    echo "[outputs_git] restored outputs from origin/$BRANCH ($(g rev-parse --short HEAD))"
  else
    echo "[outputs_git] no '$BRANCH' branch on the remote yet - starting fresh"
  fi
}

push() {
  ensure_repo
  g add -A
  # 'diff --cached --quiet' exits non-zero both when there are staged changes
  # and on an unborn HEAD - in either case we want to commit.
  if g diff --cached --quiet 2>/dev/null; then
    echo "[outputs_git] nothing new to push"
    return 0
  fi
  g commit -q -m "outputs sync $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if g push -q -u origin "$BRANCH" 2>/dev/null; then
    echo "[outputs_git] pushed $(g rev-parse --short HEAD) to origin/$BRANCH"
  else
    # The branch holds generated artifacts only; local (freshly restored +
    # extended) content is authoritative, so force-push on divergence.
    echo "[outputs_git] normal push rejected - force-pushing"
    g push -q -u --force origin "$BRANCH"
    echo "[outputs_git] force-pushed $(g rev-parse --short HEAD) to origin/$BRANCH"
  fi
}

case "$CMD" in
  restore) restore ;;
  push)    push ;;
  *) echo "usage: outputs_git.sh restore|push" >&2; exit 2 ;;
esac
