export DATA_NAME_OR_PATH="${DATA_NAME_OR_PATH:?set DATA_NAME_OR_PATH to input dataset (hf path or local file)}"
export OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to save shards}"
export DATA_FORMAT="${DATA_FORMAT:-sharegpt}"
export DATA_SHARD_SIZE="${DATA_SHARD_SIZE:-50000}"
export BASE_PORT="${BASE_PORT:-6000}"
export NUM_THREADS="${NUM_THREADS:-256}"


# Generate data
python3 ./tools/generate_data_for_target_model.py \
    --data_name_or_path "$DATA_NAME_OR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --data_format "$DATA_FORMAT" \
    --data_shard_size "$DATA_SHARD_SIZE" \
    --base_port "$BASE_PORT" \
    --num_threads "$NUM_THREADS"
