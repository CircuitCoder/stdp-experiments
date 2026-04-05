#!/usr/bin/env bash
set -euo pipefail

label=$1
connection_rate=$2
output_dir=$3
shift 3

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"

tsv_file="$output_dir/${label}.tsv"
printf 'label\trun\texit_code\taccuracy_pct\tzero_classes\tavg_firing\tmin_gain\tavg_gain\tmax_gain\tmin_rate_ema\tavg_rate_ema\tmax_rate_ema\tassigned\n' > "$tsv_file"

for run in 1 2 3; do
    log_file="$output_dir/${label}_${run}.log"
    if ./target/release/stdp-experiments \
        --train-length 2000 \
        --mark-length 500 \
        --test-length 1000 \
        --per-sample-ticks 100 \
        --output-num 200 \
        --connection-rate "$connection_rate" \
        "$@" \
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

    gain_stats=$(grep '^Feedforward gain statistics after training:' "$log_file" | tr -d ',' | awk '{print $7 "\t" $9 "\t" $11}' || true)
    if [[ -z "$gain_stats" ]]; then
        gain_stats=$'NA\tNA\tNA'
    fi

    rate_stats=$(grep '^Slow scaling EMA rate statistics after training:' "$log_file" | tr -d ',' | awk '{print $8 "\t" $10 "\t" $12}' || true)
    if [[ -z "$rate_stats" ]]; then
        rate_stats=$'NA\tNA\tNA'
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$run" "$exit_code" "${accuracy:-NA}" "$zero_classes" "${avg_firing:-NA}" \
        ${gain_stats} ${rate_stats} "${assigned:-NA}" >> "$tsv_file"
done

touch "$output_dir/${label}.done"