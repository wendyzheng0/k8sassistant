#!/bin/bash
echo "🛑 Stopping K8s Assistant services..."

# Graceful stop function: TERM first, wait, then KILL if necessary
graceful_stop() {
  local pid="$1"
  local name="$2"
  local timeout=10

  if [ -z "$pid" ]; then
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $name (PID: $pid)..."
    kill "$pid" 2>/dev/null || true
    for i in $(seq 1 $timeout); do
      if kill -0 "$pid" 2>/dev/null; then
        sleep 1
      else
        break
      fi
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name did not exit within timeout, executing force termination..."
      kill -9 "$pid" 2>/dev/null || true
    fi
  else
    echo "$name is no longer running"
  fi
}

# Try to find and stop processes by port (as fallback)
stop_by_port() {
  local port="$1"
  local name="$2"
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -t -i:"$port" 2>/dev/null | sort -u)
    if [ -n "$pids" ]; then
      echo "Stopping $name via port $port: $pids"
      for p in $pids; do
        graceful_stop "$p" "$name"
      done
    fi
  elif command -v fuser >/dev/null 2>&1; then
    if fuser "$port"/tcp >/dev/null 2>&1; then
      local pids
      pids=$(fuser -v "$port"/tcp 2>/dev/null | awk 'NR>1{for(i=2;i<=NF;i++)print $i}' | sort -u)
      for p in $pids; do
        graceful_stop "$p" "$name"
      done
    fi
  fi
}

# Stop backend (prefer PID file)
if [ -f .backend.pid ]; then
  BACKEND_PID=$(cat .backend.pid 2>/dev/null)
  if [[ "$BACKEND_PID" =~ ^[0-9]+$ ]]; then
    graceful_stop "$BACKEND_PID" "Backend service"
  else
    echo "Backend PID file invalid, trying to stop via port..."
  fi
  rm -f .backend.pid
else
  echo "Backend PID file not found, trying to stop via port (8000)..."
fi
stop_by_port 8000 "Backend service"

# Stop frontend (prefer PID file)
if [ -f .frontend.pid ]; then
  FRONTEND_PID=$(cat .frontend.pid 2>/dev/null)
  if [[ "$FRONTEND_PID" =~ ^[0-9]+$ ]]; then
    graceful_stop "$FRONTEND_PID" "Frontend service"
  else
    echo "Frontend PID file invalid, trying to stop via port..."
  fi
  rm -f .frontend.pid
else
  echo "Frontend PID file not found, trying to stop via port (3000)..."
fi
stop_by_port 3000 "Frontend service"

# Stop Nginx (if using local temp config or process still running)
need_stop_nginx=false
if [ -f /tmp/nginx_native.conf ]; then
  need_stop_nginx=true
fi
if pgrep -x nginx >/dev/null 2>&1; then
  need_stop_nginx=true
fi

if [ "$need_stop_nginx" = true ]; then
  echo "Stopping Nginx..."
  nginx -s quit 2>/dev/null || true
  # Wait up to 5 seconds
  for i in $(seq 1 5); do
    if pgrep -x nginx >/dev/null 2>&1; then
      sleep 1
    else
      break
    fi
  done
  if pgrep -x nginx >/dev/null 2>&1; then
    echo "Nginx still running, executing force termination..."
    pkill -9 -x nginx 2>/dev/null || true
  fi
  rm -f /tmp/nginx_native.conf 2>/dev/null || true
fi

echo "✅ All services stopped"