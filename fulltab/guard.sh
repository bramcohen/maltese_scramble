#!/bin/zsh
# guard.sh PID [FLOOR_GB]: kill PID (and only PID) if the memory not held by anonymous, wired or compressed pages
# (i.e. free plus reclaimable file cache) drops below FLOOR_GB (default 12).  Logs the lowest level seen to fulltab/data/guard.log.
pid=$1; floor=${2:-12}; low=1000000; total=$(sysctl -n hw.memsize)
while kill -0 $pid 2>/dev/null; do
  avail=$(vm_stat | awk -v t=$total '/Pages wired down/ {w=$4} /Anonymous pages/ {a=$3} /Pages occupied by compressor/ {c=$5} END {printf "%d", (t - (w+a+c)*16384)/1073741824}')
  (( avail < low )) && low=$avail
  if (( avail < floor )); then
    echo "$(date): available ${avail} GB < ${floor} GB, killing $pid" >> /Users/bram/maltese_scramble/fulltab/data/guard.log
    kill -9 $pid; break
  fi
  sleep 2
done
echo "$(date): pid $pid done; lowest available ${low} GB" >> /Users/bram/maltese_scramble/fulltab/data/guard.log
