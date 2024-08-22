for i in {0..39}; do
  device_num=$((i % 8))
  device_name="cuda:$device_num"

  RUST_BACKTRACE=1 hap run -- python incremental_parsing/generation/stack_completion_right_context.py \
      --min-seed 0 --max-seed 0 --min-data-idx $((i * 250)) --max-data-idx $((i * 250 + 249)) \
      --min-cut-idx 0 --max-cut-idx 9 --beam-size 1 \
      --model-name "bigcode/santacoder" --dataset-name "bigcode/the-stack-smol-xl" --dataset-path "data/python" \
      --results-path "/path/to/results" \
      --device $device_name --max-tokens 500 --num-token-finding-attempts 50
done