#!/usr/bin/env bash
# stop.sh - Cleanly stop CiteBase server on Linux / macOS / EC2
PID=$(lsof -ti:8000 2>/dev/null)
if [ -n "$PID" ]; then
  echo "Stopping CiteBase process (PID: $PID)..."
  kill -TERM "$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
  echo "CiteBase server stopped. Port 8000 is clean."
else
  echo "No process found running on port 8000."
fi
