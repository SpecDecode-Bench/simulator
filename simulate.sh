#!/bin/bash

# Full simulation script
# Runs all proposers, datasets, methods, and batch sizes

echo "=========================================="
echo "Starting Full Simulation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Proposers: eagle3, ngram, combined"
echo "  - Datasets: gsm8k, instructcoder, sharegpt, cnn"
echo "  - Predict Methods: FIXED_1, FIXED_3, FIXED_5, ORACLE"
echo "  - Batch Sizes: 1, 8, 16, 32, 64, 128"
echo ""
echo "This will generate CSV files in the output directory"
echo ""

# All parameters
DATASETS="gsm8k instructcoder sharegpt cnn"
PREDICT_METHODS="FIXED_1 FIXED_3 FIXED_5 ORACLE"
BATCH_SIZES="1 8 16 32 64 128"
OUTPUT_DIR="results"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run EAGLE3 proposer with all predict methods
echo "================================================"
echo "Running EAGLE3 proposer simulations..."
echo "================================================"
python main.py \
    --proposer eagle3 \
    --datasets ${DATASETS} \
    --predict-methods ${PREDICT_METHODS} \
    --batch-sizes ${BATCH_SIZES} \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "EAGLE3 simulations completed!"
echo ""

# Run NGRAM proposer with all predict methods
echo "================================================"
echo "Running NGRAM proposer simulations..."
echo "================================================"
python main.py \
    --proposer ngram \
    --datasets ${DATASETS} \
    --predict-methods ${PREDICT_METHODS} \
    --batch-sizes ${BATCH_SIZES} \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "NGRAM simulations completed!"
echo ""

# Run COMBINED proposer (automatically uses ORACLE method only)
echo "================================================"
echo "Running COMBINED (NGRAM+EAGLE) proposer simulations..."
echo "================================================"
python main.py \
    --proposer combined \
    --datasets ${DATASETS} \
    --batch-sizes ${BATCH_SIZES} \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "COMBINED simulations completed!"
echo ""

echo "=========================================="
echo "All Simulations Complete!"
echo "=========================================="
echo ""
echo "Generated CSV files:"
ls -lh ${OUTPUT_DIR}/*.csv
echo ""
echo "Summary:"
echo "  - Total CSV files: $(ls ${OUTPUT_DIR}/*.csv 2>/dev/null | wc -l)"
echo "  - Location: ${OUTPUT_DIR}/"