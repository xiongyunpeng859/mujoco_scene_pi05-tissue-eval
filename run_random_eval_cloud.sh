#!/usr/bin/env bash
set -euo pipefail
task_repo=$(cd "$(dirname "$0")" && pwd)
export PI05_PROJECT_ROOT="$task_repo"
source "$task_repo/cloud_env.sh"
cd "$task_repo"
count="${1:-10}"
out_root="${2:-$HOME/eval_random}"
[[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Episode count must be positive" >&2; exit 2; }
if [ -e "$out_root/summary.tsv" ]; then echo "Use a new output directory" >&2; exit 2; fi
mkdir -p "$out_root"
summary="$out_root/summary.tsv"
printf 'episode\tseed\tstatus\tresult\n' > "$summary"
for ((i=0; i<count; i++)); do
  seed=$((2026092700+i))
  out="$out_root/episode-$(printf '%04d' "$i")"
  mkdir -p "$out"
  echo "[$i/$count] random object placement seed=$seed"
  if python -u pi05_scripts/test_jax_policy_rollout.py \
      --checkpoint "$CHECKPOINT" --output-dir "$out" \
      --execute-chunk 50 --ticks 375 --seed "$seed" \
      >"$out/run.log" 2>&1; then
    result=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("final_success"))' "$out/result.json" 2>/dev/null || echo unknown)
    printf '%s\t%s\tOK\t%s\n' "$i" "$seed" "$result" >> "$summary"
  else
    printf '%s\t%s\tFAILED\t%s\n' "$i" "$seed" "$out/run.log" >> "$summary"
    echo "failed: episode $i (see $out/run.log)"
  fi
done
python - "$summary" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ok=[r for r in rows if r['status']=='OK']
wins=[r for r in ok if r['result']=='True']
import json
from pathlib import Path
report=dict(episodes=len(rows),completed=len(ok),errors=len(rows)-len(ok),successes=len(wins),success_rate_completed=len(wins)/len(ok) if ok else None,scope='one selected target per episode; fixed scene; random object xy/yaw')
Path(sys.argv[1]).with_suffix('.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
