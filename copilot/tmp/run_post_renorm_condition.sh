#!/usr/bin/env bash
set -euo pipefail

connectivity=$1
normalization=$2
connection_rate=$3
output_dir=$4

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"

tsv_file="$output_dir/${connectivity}_${normalization}.tsv"
printf 'connectivity\tnormalization\trun\texit_code\taccuracy_pct\tzero_classes\tavg_firing\tassigned\n' > "$tsv_file"

for run in 1 2 3 4 5; do
    log_file="$output_dir/${connectivity}_${normalization}_${run}.log"
    if ./target/release/stdp-experiments \
        --train-length 2000 \
        --mark-length 500 \
        --test-length 1000 \
        --per-sample-ticks 100 \
        --output-num 200 \
        --connection-rate "$connection_rate" \
        --normalization "$normalization" \
        > "$log_file" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi

    accuracy=$(grep '^Accuracy:' "$log_file" | awk '{gsub("%", "", $2); print $2}' || true)
    assigned=$(grep '^Assigned class neurons:' "$log_file" | sed 's/^Assigned class neurons: //' || true)
    if [[ -n "$assigned" ]]; then
        zero_classes=$(printf '%s\n' "$assigned" | tr -d '[],' | awk '{z=0; for(i=1;i<=NF;i++) if($i==0) z++; print z}')
    else
        zero_classes=NA
    fi
    avg_firing=$(grep '^Firing rate statistics' "$log_file" | tr -d ',' | awk '{print $11}' || true)

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$connectivity" "$normalization" "$run" "$exit_code" "${accuracy:-NA}" "$zero_classes" "${avg_firing:-NA}" "${assigned:-NA}" \
        >> "$tsv_file"
done

touch "$output_dir/${connectivity}_${normalization}.done"