#!/bin/bash
# Quick end-to-end pipeline test: 2 epochs, fold 1 only
set -e

echo "=== Step 1: Training (2 epochs, all 12 folds) ==="
python -u train_EPInformer_abc.py \
    --h5_path ./training_data/k562_run_v2/samples.h5 \
    --epochs 2 \
    --output_dir ./EPInformer_models_test/ \
    --batch_size 50

echo ""
echo "=== Step 2: Prediction (attention extraction) ==="
python -u predict_E2G_abc_feat.py \
    --h5_path ./training_data/k562_run_v2/samples.h5 \
    --model_dir ./EPInformer_models_test/ \
    --output_dir ./enhancer_attn_scores_test/

echo ""
echo "=== Step 3: Report ==="
python -u report_EPInformer.py \
    --results_dir ./EPInformer_models_test/ \
    --output_dir ./EPInformer_report_test/

echo ""
echo "=== Pipeline test complete! ==="
