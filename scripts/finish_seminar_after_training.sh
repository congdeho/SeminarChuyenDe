#!/bin/bash
# Final review artifacts are built after each of the eight queued runs completes.
set -euo pipefail

export MPLCONFIGDIR=.cache/matplotlib
export XDG_CACHE_HOME=.cache
output=outputs/phase6/ban_ket_qua
training_dir=runs/phase5
python=.venv/bin/python

previous=""
last_progress=$SECONDS
while true; do
  progress=""
  complete=0
  for seed in 123 2026; do
    for experiment in e1 e2 e3 e4; do
      run="$training_dir/${experiment}_seed${seed}"
      if [ -s "$run/history.csv" ]; then
        progress+="$(wc -l < "$run/history.csv"),"
      fi
      if [ -f "$run/run_summary.json" ] && rg -q '"complete": true' "$run/run_summary.json"; then
        complete=$((complete + 1))
      fi
    done
  done
  if [ "$complete" -eq 8 ]; then
    break
  fi
  if [ "$progress" != "$previous" ]; then
    previous="$progress"
    last_progress=$SECONDS
  elif [ "$((SECONDS - last_progress))" -ge 1500 ]; then
    printf 'Training has produced no epoch for 25 minutes. See the run histories under %s.\n' "$training_dir" >&2
    exit 1
  fi
  sleep 30
done

"$python" -m src.finalize_trashnet
if [ ! -f "$output/Demo_Seminar_TrashNet.zip" ]; then
  "$python" -m src.package_seminar --output "$output"
fi
unzip -tq "$output/Demo_Seminar_TrashNet.zip"
if [ ! -f "$output/Seminar_TrashNet.pptx" ]; then
  "${SEMINAR_RUNTIME_NODE:?}" scripts/build_seminar_deck.mjs "$output"
fi
printf 'Artifacts ready for final visual review in %s.\n' "$output"
