#!/usr/bin/env bash
set -e
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Installing ffmpeg..."
  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg
  else
    echo "Please install ffmpeg manually: https://ffmpeg.org/download.html"; exit 1
  fi
else
  echo "ffmpeg already installed"
fi
