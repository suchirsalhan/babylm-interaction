#!/bin/bash

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CONFIG_PATH" ]; then
    echo "Error: --config is required"
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "Error: --mode is required"
    exit 1
fi

# Set default output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="outputs"
fi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the appropriate script based on mode
case $MODE in
    "train")
        python -m core.trainer --config $CONFIG_PATH --output_dir $OUTPUT_DIR
        ;;
    "train_reward")
        python -m reward_model.train --config $CONFIG_PATH --output_dir $OUTPUT_DIR
        ;;
    "eval")
        python -m core.evaluator --config $CONFIG_PATH --output_dir $OUTPUT_DIR
        ;;
    *)
        echo "Error: Unknown mode $MODE"
        exit 1
        ;;
esac 