#!/usr/bin/env bash
set -euo pipefail

# V3 main.py is YAML/config driven. Legacy flags such as --mode,
# --llm-model, --iters, and --edge-from are now encoded in config.yaml.

CONFIG=${CONFIG:-config.yaml}

# COPDGene / CVD Prediction rerun after the clean prediction label fix.
python main.py --config "$CONFIG" --only copdgene_cvd_pred5y_baseline
python main.py --config "$CONFIG" --only copdgene_cvd_pred5y_qwen

# Useful checks / alternatives:
# python main.py --config "$CONFIG" --only copdgene_cvd_pred5y_baseline --dry-run
# python main.py --config "$CONFIG" --only copdgene_cvd_pred5y_qwen --dry-run
# python main.py --config "$CONFIG" --pattern copdgene_cvd_pred5y --dry-run
#
# Status experiments are configured in config.yaml as well, but the current
# rerun target is Prediction:
# python main.py --config "$CONFIG" --only copdgene_cvd_status_baseline
# python main.py --config "$CONFIG" --only copdgene_cvd_status_qwen
