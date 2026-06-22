# LLM-Steered Clinical Knowledge Discovery for CVD Outcome Modeling

This repository contains the official code release for our accepted MICCAI 2026 paper:

**LLM-Steered Clinical Knowledge Discovery for CVD Outcome Modeling**
Yuxuan Liang, Xuangang Xu, Ge Wang, and Pingkun Yan
Department of Biomedical Engineering and Center for Biotechnology and Interdisciplinary Studies, Rensselaer Polytechnic Institute

The code implements **Evidence-Grounded Learning (EGL)** and **EGL-LLM** for clinical knowledge discovery from repeated fixed-budget model evaluations. EGL accumulates auditable node and edge evidence in an evidence graph. EGL-LLM adds an LLM-based semantic control plane that steers node/edge probing and subset sampling while keeping the fixed evaluator and evidence definitions unchanged.

## Paper Scope and Configured Tasks

The MICCAI 2026 paper reports the **COPDGene CVD status** experiment. After the paper submission, the experiment configuration was extended for additional internal analyses. Therefore, `config.yaml` currently includes:

- implemented COPDGene CVD status experiments: `copdgene_cvd_status_baseline`, `copdgene_cvd_status_qwen`;
- implemented COPDGene CVD 5-year prediction experiments: `copdgene_cvd_pred5y_baseline`, `copdgene_cvd_pred5y_qwen`;
- template task bases for NLST CVD status/prediction and COPDGene COPD status/prediction.

Only the COPDGene CVD status results are reported in the accepted MICCAI 2026 paper. The additional configured tasks are included to document the post-submission development state of the codebase and may require task-specific data and label checks before reuse.

## Data Availability

The experiments in the paper use COPDGene-derived clinical and CT biomarker tables. These data cannot be redistributed in this public repository. Users who wish to reproduce the full experiments must obtain the relevant COPDGene/NLST data through the appropriate data-use agreements and then place processed tables in the expected local structure.

The code expects project-local data files such as:

```text
data/
  raw_data/
    COPDGene_P1P2P3_SM_NS_Long_SEP24.xlsx or .feather
    full_label_copd_nlst.csv or .feather
  splitted_data/
    COPD_ct_optimal_series_only.feather
    NLST_ct_optimal_series_only.feather
    ct_variable_summary.csv
```

The main experiment config is kept as used in development (`config.yaml`). To reproduce the paper experiments, replace the data paths in `config.yaml` with your local processed data paths while preserving the expected columns.

Expected data fields include:

- subject identifier: `sid`
- visit/phase identifier: `Phase_study`
- CVD event/status field: `diag_cvd`
- CT-derived and clinical candidate feature columns
- feature metadata in `ct_variable_summary.csv`, including at least:
  - `var_name`
  - variable type/domain fields used by the preprocessing code
  - CVD inclusion flags
  - missingness information

The `data_processing/` directory provides preprocessing scripts and templates, but no raw or processed data are included.

## Installation

We recommend using [`uv`](https://docs.astral.sh/uv/) for a reproducible Python environment.

```bash
git clone https://github.com/liangyuxuan1/EGL.git
cd EGL
uv sync
```

The project requires Python 3.12. The locked dependency set is provided in `uv.lock`.

## LLM Endpoint Setup

EGL baseline experiments do not require an LLM endpoint. EGL-LLM experiments use an OpenAI-compatible Qwen endpoint. Configure it through environment variables:

```bash
export EGL_QWEN_API_BASE=http://localhost:8001/v1
export EGL_QWEN_API_KEY=your_api_key_or_placeholder
export EGL_QWEN_MODEL=Qwen3-14B-Instruct
```

## Repository Structure

```text
EGL/
  main.py                         # YAML-driven experiment entry point
  main_loop.py                    # parallel interleaving EGL/EGL-LLM search loop
  config.yaml                     # experiment definitions and run parameters
  config_cvd.py                   # CVD event-column definitions
  config_graph.py                 # graph/evidence configuration helpers
  config_registry.py              # event-column registry
  utils_dataset.py                # DatasetCtx construction, labels, preprocessing
  utils_models.py                 # fixed evaluator and Qwen client wrapper
  utils_control.py                # online thresholds, rewards, LLM control utilities
  utils_graph.py                  # evidence graph / knowledge graph utilities
  utils_cache.py                  # SQLite evaluation cache
  utils.py                        # general helpers
  utils_plot.py                   # plotting helpers
  utils_prompt.md                 # prompt template text

  plot_all_from_config.py         # plot workflow driver
  plot_config.yaml                # plot workflow config
  plot_config_utils.py            # plot config helpers
  plot_01_iterations.py           # iteration/reward/metric trace plots
  plot_02_stable_kg.py            # stable evidence graph node/edge plots
  plot_03_path_agenda.py          # path and agenda analysis plots
  plot_04_edge_interaction.py     # edge interaction effect analysis
  plot_05_compare_figures.py      # figure comparison helpers

  data_processing/                # preprocessing templates for private local data
  run_batch.sh                    # example batch command script
  pyproject.toml                  # uv project metadata
  uv.lock                         # locked dependency set
```

Generated files are intentionally ignored by git, including local data, cached CV evaluations, experiment outputs, and generated figures.

## Running Experiments

List configured experiments:

```bash
uv run python main.py --config config.yaml --list
```

Inspect a resolved config without running:

```bash
uv run python main.py --config config.yaml --only copdgene_cvd_status_baseline --dry-run
```

Run EGL baseline on COPDGene CVD status:

```bash
uv run python main.py --config config.yaml --only copdgene_cvd_status_baseline
```

Run EGL-LLM on COPDGene CVD status:

```bash
uv run python main.py --config config.yaml --only copdgene_cvd_status_qwen
```

Run CVD prediction variants (post-submission extension; not reported in the MICCAI 2026 paper):

```bash
uv run python main.py --config config.yaml --only copdgene_cvd_pred5y_baseline
uv run python main.py --config config.yaml --only copdgene_cvd_pred5y_qwen
```

The helper script `run_batch.sh` shows an example batch workflow.

## Plotting

After experiment outputs are available locally, run:

```bash
uv run python plot_all_from_config.py
```

Common partial plotting commands:

```bash
uv run python plot_all_from_config.py --only status --skip-comparisons
uv run python plot_all_from_config.py --only prediction --skip-comparisons
uv run python plot_all_from_config.py --sections comparisons
```

The plotting scripts expect local experiment outputs under `outputs/`, which are not included in the repository.

## Expected Outputs

A completed EGL/EGL-LLM run writes local artifacts such as:

```text
outputs/<experiment_name>_iters500_BaseS[_useP_useDB]/
  df_all.csv
  df_all.feather
  pool_meta.csv
  iteration_log.csv
  kg_snapshot.json
  control_log.csv            # EGL-LLM only
  warmup_params.json
  per_fold_metrics.csv
  per_fold_metrics_std.csv
```

Plotting creates additional local figure folders under `outputs/`. These generated outputs are ignored by git.

## Data Processing Templates

The scripts in `data_processing/` document the private-data preprocessing workflow used for the paper:

```text
data_processing/
  S00_convert_raw_to_feather.py
  S01_split_COPD_NLST.py
  S02_keep_optimal_CT_series_only.py
  S02_check_diag_cvd.py
  S03_merge_clinical_data.py
  S03_sanity_check_cvd_events.py
  COPD_CT_Dict.py
  utils_data_processing.py
  README_preprocessing.md
```

They are provided as templates for users who have obtained the required datasets. They do not include or download COPDGene/NLST data.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{liang2026egl,
  title     = {LLM-Steered Clinical Knowledge Discovery for CVD Outcome Modeling},
  author    = {Liang, Yuxuan and Xu, Xuangang and Wang, Ge and Yan, Pingkun},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026}
}
```

## License

This repository is released under the license included in `LICENSE`.
