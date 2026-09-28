#!/usr/bin/env bash
set -u
source "$(dirname "$0")/cloud_env.sh"
rounds_root="${1:-$HOME/artifacts/rounds}"
out_root="${2:-$HOME/eval_many}"
limit="${3:-0}"
mkdir -p "$out_root"
summary="$out_root/summary.tsv"
printf 'round\tstatus\tresult\n' > "$summary"
i=0
while IFS= read -r round; do
  [ "$limit" -gt 0 ] && [ "$i" -ge "$limit" ] && break
  name="$(basename "$round")"
  out="$out_root/$name"
  mkdir -p "$out"
  echo "[${i}] evaluating $name"
  if python -u pi05_scripts/test_jax_policy_rollout.py \
      --checkpoint "$CHECKPOINT" --known-round "$round" \
      --output-dir "$out" --execute-chunk 50 --ticks 375 \
      >"$out/run.log" 2>&1; then
    result=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("final_success"))' "$out/result.json" 2>/dev/null || echo unknown)
    printf '%s\tOK\t%s\n' "$name" "$result" >> "$summary"
  else
    printf '%s\tFAILED\t%s\n' "$name" "$out/run.log" >> "$summary"
    echo "failed: $name (see $out/run.log)"
  fi
  i=$((i+1))
done < <(find "$rounds_root" -mindepth 1 -maxdepth 1 -type d -name 'round-*' | sort -V)
python - "$summary" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ok=[r for r in rows if r['status']=='OK']
wins=[r for r in ok if r['result']=='True']
print(f'evaluated={len(rows)} completed={len(ok)} successes={len(wins)}')
PY
