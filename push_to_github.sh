#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: ./push_to_github.sh <git-remote-url>"
  echo "Example: ./push_to_github.sh git@github.com:USERNAME/eda_park_pd_demo.git"
  exit 1
fi
REMOTE="$1"
git init -b main
git add -A
git commit -m "Initial commit: EDA Parkinson demo"
git remote add origin "$REMOTE" || git remote set-url origin "$REMOTE"
git push -u origin main
