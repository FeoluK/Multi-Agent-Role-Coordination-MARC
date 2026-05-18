#!/bin/bash
# Throttled auto-submitter: drips the remaining baseline sweeps into the
# gpu QOS as headroom frees (MaxSubmitJobsPerUser=32). Idempotent via a
# state file (restartable). Only ever submits the 3 fixed chunks below;
# never touches the thesis array (it's just counted in `used`, so this
# also never starves it). Self-terminates when all are submitted.
set -u
S=/scratch/users/flukol/marc
CAP=32
STATE=$S/logs/queue_filler.state
LOG(){ echo "$(date '+%F %T') $*"; }

CHUNKS=(
  "cds_mpe|9|sbatch --array=0-8%4 $S/scripts/cds.sbatch"
  "cds_oc|15|sbatch --array=9-23%4 $S/scripts/cds.sbatch"
  "mappo_oc|15|sbatch --array=9-23%4 $S/scripts/mappo.sbatch"
)

i=$(cat "$STATE" 2>/dev/null || echo 0)
LOG "queue_filler start at chunk index $i / ${#CHUNKS[@]}"
for ((it=0; it<240; it++)); do
  if [ "$i" -ge "${#CHUNKS[@]}" ]; then LOG "ALL SUBMITTED"; break; fi
  used=$(squeue -u flukol -h -r -t PENDING,RUNNING 2>/dev/null | wc -l)
  IFS='|' read -r name sz cmd <<< "${CHUNKS[$i]}"
  head=$((CAP - used))
  if [ "$head" -ge "$sz" ]; then
    LOG "submitting $name (size $sz, used $used, headroom $head)"
    if eval "$cmd"; then
      i=$((i + 1)); echo "$i" > "$STATE"
      LOG "submitted $name; next index $i"
    else
      LOG "submit FAILED for $name (will retry)"
    fi
  else
    LOG "waiting: $name needs $sz, headroom $head (used $used)"
  fi
  sleep 300
done
LOG "queue_filler exiting (index $i)"
