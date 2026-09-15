import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import torch

import util
from engine import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train the proposed mobility forecasting model.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--adjdata', default=None)
    parser.add_argument('--adjtype', default='doubletransition')
    parser.add_argument('--gcn_bool', action='store_true')
    parser.add_argument('--addaptadj', action='store_true')
    parser.add_argument('--randomadj', action='store_true')
    parser.add_argument('--use_dynamic_graph', action='store_true')
    parser.add_argument('--use_phys', action='store_true')
    parser.add_argument('--use_social', action='store_true')
    parser.add_argument('--use_lag_align', action='store_true')
    parser.add_argument('--phys_path', default=None)
    parser.add_argument('--social_path', default=None)
    parser.add_argument('--time_start', default=None)
    parser.add_argument('--time_end', default=None)
    parser.add_argument('--drop_node_indices', default='')
    parser.add_argument('--exog_fill', default='error', choices=['error', 'zero', 'ffill_bfill'])
    parser.add_argument('--node_align', default='name', choices=['name', 'normalized_name', 'position'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--d_int', type=int, default=16)
    parser.add_argument('--dynamic_rank', type=int, default=10)
    parser.add_argument('--lag_window', type=int, default=12)
    parser.add_argument('--lag_temperature', type=float, default=1.0)
    parser.add_argument('--lag_div_gamma', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--save_dir', default='log')
    parser.add_argument('--exp_name', default='full')
    return parser.parse_args()


def prepare(args):
    if args.use_dynamic_graph and not (args.use_phys or args.use_social):
        raise ValueError('--use_dynamic_graph requires --use_phys and/or --use_social.')
    if args.use_lag_align and not (args.use_phys and args.use_social):
        raise ValueError('--use_lag_align requires both --use_phys and --use_social.')
    if (args.use_phys or args.use_social) and (args.time_start is None or args.time_end is None):
        raise ValueError('External inputs require --time_start and --time_end.')
    util.seed_everything(args.seed)
    device = util.resolve_device(args.device)
    metadata = util.infer_dataset_metadata(args.data)
    args.adjdata = util.resolve_adj_path(args.data, args.adjdata)
    args.phys_in_dim = 1
    data = util.load_dataset(
        args.data,
        args.batch_size,
        args.batch_size,
        args.batch_size,
        phys_path=args.phys_path,
        social_path=args.social_path,
        time_start=args.time_start,
        time_end=args.time_end,
        drop_node_indices=args.drop_node_indices,
        exog_fill=args.exog_fill,
        node_align=args.node_align,
        load_exogenous=(args.use_phys or args.use_social),
    )
    args.phys_in_dim = data['phys_dim']
    _, _, matrices = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(matrix, dtype=torch.float32, device=device) for matrix in matrices]
    adjinit = None if args.randomadj else supports[0]
    return device, metadata, data, supports, adjinit


def batch_tensors(batch, device):
    x, y, phys, social = batch
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device).transpose(1, 3),
        torch.as_tensor(y, dtype=torch.float32, device=device).transpose(1, 3)[:, 0],
        torch.as_tensor(phys, dtype=torch.float32, device=device),
        torch.as_tensor(social, dtype=torch.float32, device=device),
    )


def evaluate(trainer, loader, device, size):
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader.get_iterator():
            x, y, phys, social = batch_tensors(batch, device)
            predictions.append(trainer.predict(x, phys, social).cpu())
            targets.append(y.transpose(1, 2).cpu())
    prediction = torch.cat(predictions, dim=0)[:size].squeeze(-1)
    target = torch.cat(targets, dim=0)[:size]
    return np.asarray(util.metric(prediction, target), dtype=np.float64)


def main():
    args = parse_args()
    device, metadata, data, supports, adjinit = prepare(args)
    trainer = Trainer(data['scaler'], metadata, args, device, supports, adjinit)
    run_name = '{}_{}'.format(args.exp_name, datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'run_config.json')
    with open(config_path, 'w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, indent=2)

    best_validation = float('inf')
    best_epoch = 0
    checkpoint_path = os.path.join(output_dir, 'best.pth')
    for epoch in range(1, args.epochs + 1):
        data['train_loader'].shuffle()
        train_metrics = []
        for batch in data['train_loader'].get_iterator():
            train_metrics.append(trainer.step(*batch_tensors(batch, device), training=True))
        train_mean = np.mean(train_metrics, axis=0)
        validation = evaluate(trainer, data['val_loader'], device, len(data['y_val']))
        if validation[0] < best_validation:
            best_validation = float(validation[0])
            best_epoch = epoch
            torch.save(trainer.model.state_dict(), checkpoint_path)
        print(
            'Epoch {:03d} | train MAE {:.4f} | val MAE {:.4f} | val MAPE {:.4f} | val RMSE {:.4f}'.format(
                epoch, train_mean[0], validation[0], validation[1], validation[2]
            ),
            flush=True,
        )

    trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(trainer, data['test_loader'], device, len(data['y_test']))
    results_path = os.path.join(output_dir, 'results.csv')
    with open(results_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['best_epoch', 'validation_mae', 'test_mae', 'test_mape', 'test_rmse'])
        writer.writerow([best_epoch, best_validation, *test_metrics.tolist()])
    print('Best checkpoint:', checkpoint_path)
    print('Test MAE {:.4f} | MAPE {:.4f} | RMSE {:.4f}'.format(*test_metrics))


if __name__ == '__main__':
    main()
