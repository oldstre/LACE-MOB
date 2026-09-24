# LACE-Mob: A Lag-Aware Context-Enhanced Framework for Human Mobility Forecasting During Disasters

This repository provides the official implementation of **LACE-Mob**, a
lag-aware, context-enhanced framework for human mobility forecasting during
disasters. It contains the proposed model and the code required to train and
evaluate it.

## Structure

```text
LACE-Mob/
|-- model/
|   `-- model.py          Proposed lag-aware context-enhanced model
|-- data/                 Mobility tensors and adjacency matrices
|-- train.py              Training entry point
|-- test.py               Test-set evaluation entry point
|-- engine.py             Optimization wrapper
|-- util.py               Data loading and metrics
|-- requirements.txt
`-- LICENSE
```

The release includes processed mobility tensors and adjacency matrices for
`Japan_Typoon_in`, `Japan_Typoon_out`, `Florida`, and `NYC_Storm_in`.

The CA-Fire dataset and the external-condition data used in this study,
including disaster-related tweet counts, precipitation, wind, and PM2.5, were
collected and processed by the authors from multiple sources. These data are
not redistributed because they may be subject to privacy considerations,
source-platform terms of use, and third-party licensing or redistribution
restrictions. To protect the original data subjects and comply with the
applicable data-use conditions, this repository therefore releases only the
redistributable mobility tensors and graph structures. Researchers who have
obtained the required permissions may prepare compatible external signals and
provide them locally through `--phys_path` and `--social_path`.

## Installation

```bash
pip install -r requirements.txt
```

## Mobility-only training

This mode works directly with the distributed files:

```bash
python train.py \
  --data data/Japan_Typoon_in \
  --device cuda:0 \
  --gcn_bool --adjtype doubletransition --addaptadj --randomadj \
  --epochs 300 \
  --save_dir log \
  --exp_name mobility_only
```

## Full-model training

The external CSV files must contain an hourly timestamp column followed by one
column per graph node. Physical and social files must cover the complete time
range and use matching node columns.

```bash
python train.py \
  --data data/Japan_Typoon_in \
  --device cuda:0 \
  --gcn_bool --adjtype doubletransition --addaptadj --randomadj \
  --use_dynamic_graph --use_phys --use_social --use_lag_align \
  --lag_window 12 --lag_temperature 1.0 --lag_div_gamma 0.1 \
  --phys_path /path/to/physical_signal.csv \
  --social_path /path/to/social_signal.csv \
  --time_start "2019-07-01 00:00:00" \
  --time_end "2019-10-30 23:00:00" \
  --epochs 300 \
  --save_dir log \
  --exp_name full
```

## Testing

`test.py` automatically reads the training configuration stored beside the
checkpoint:

```bash
python test.py \
  --checkpoint log/full_YYYYMMDD_HHMMSS/best.pth \
  --device cuda:0
```

When testing a Full checkpoint on another machine, use `--phys_path` and
`--social_path` to override the private paths recorded during training.
