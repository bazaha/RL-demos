#!/usr/bin/env bash
# GPU monitor: samples nvidia-smi every second into a CSV.
# Usage: gpu_monitor.sh start|stop [gpu_id] [log_path]
set -u
GPU_ID="${2:-0}"
LOG="${3:-results/gpu_log.csv}"
PIDFILE="results/gpu_monitor.$(basename "$LOG" .csv).pid"

case "${1:-start}" in
  start)
    mkdir -p results
    echo "timestamp,util_pct,mem_used_mib,power_w,temp_c" > "$LOG"
    (
      while true; do
        nvidia-smi --id="$GPU_ID" --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu \
          --format=csv,noheader,nounits | \
          awk -v t="$(date +%s)" -F', *' '{printf "%s,%s,%s,%s,%s\n", t, $1, $2, $3, $4}' >> "$LOG"
        sleep 1
      done
    ) &
    echo $! > "$PIDFILE"
    echo "gpu monitor started on GPU $GPU_ID -> $LOG (pid $(cat $PIDFILE))"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      kill "$(cat $PIDFILE)" 2>/dev/null
      rm -f "$PIDFILE"
      echo "gpu monitor stopped; $(wc -l < $LOG) samples in $LOG"
    fi
    ;;
esac
