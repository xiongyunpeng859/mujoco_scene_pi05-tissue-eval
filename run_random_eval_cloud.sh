#!/usr/bin/env bash
set -u
source "$(dirname "$0")/cloud_env.sh"
count="${1:-10}"
out_root="${2:-$HOME/eval_random}"
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
print(f'episodes={len(rows)} completed={len(ok)} successes={len(wins)}')
PY
