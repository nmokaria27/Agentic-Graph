#!/usr/bin/env bash
# Run a command on gpu02 through the jump host's shared SSH connection.
# The user opens that connection once (password typed by them, never stored):
#   ssh -o ControlPersist=8h -fN mind-jump
# Usage: scripts/jev/remote.sh '<command run in bash on gpu02>'
set -euo pipefail
ssh -O check mind-jump >/dev/null 2>&1 || { echo "no shared connection: run 'ssh -o ControlPersist=8h -fN mind-jump'" >&2; exit 2; }
printf '%s' "$1" | ssh -o BatchMode=yes mind-jump "ssh -o BatchMode=yes gpu02.mind.cs.umd.edu bash -s" 2> >(grep -v "post-quantum\|store now\|upgraded" >&2)
