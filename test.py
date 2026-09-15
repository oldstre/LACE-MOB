import argparse
import json
import os

import torch

import util
from model.model import gwnet


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint on the test split.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--data', default=None)
    parser.add_argument('--phys_path', default=None)
    parser.add_argument('--social_path', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config or os.path.join(os.path.dirname(args.checkpoint), 'run_config.json')
    with open(config_path, 'r', encoding='utf-8') as handle:
        config = json.load(handle)
    data_dir = args.data or config['data']
    device = util.resolve_device(args.device or config['device'])
    metadata = util.infer_dataset_metadata(data_dir)
    adj_path = util.resolve_adj_path(data_dir, config.get('adjdata'))
    _, _, matrices = util.load_adj(adj_path, config.get('adjtype', 'doubletransition'))
    supports = [torch.tensor(matrix, dtype=torch.float32, device=device) for matrix in matrices]
    adjinit = None if config.get('randomadj', False) else supports[0]
    use_exogenous = config.get('use_phys', False) or config.get('use_social', False)
    data = util.load_dataset(
        data_dir,
        config.get('batch_size', 32),
        config.get('batch_size', 32),
        config.get('batch_size', 32),
        phys_path=args.phys_path or config.get('phys_path'),
        social_path=args.social_path or config.get('social_path'),
        time_start=config.get('time_start'),
        time_end=config.get('time_end'),
        drop_node_indices=config.get('drop_node_indices', ''),
        exog_fill=config.get('exog_fill', 'error'),
        node_align=config.get('node_align', 'name'),
        load_exogenous=use_exogenous,
    )
    model = gwnet(
        device,
        metadata['num_nodes'],
        config.get('dropout', 0.3),
        supports=supports,
        gcn_bool=config.get('gcn_bool', False),
        addaptadj=config.get('addaptadj', False),
        aptinit=adjinit,
        in_dim=metadata['in_dim'],
        out_dim=metadata['seq_length'],
        residual_channels=config.get('nhid', 32),
        dilation_channels=config.get('nhid', 32),
        skip_channels=config.get('nhid', 32) * 8,
        end_channels=config.get('nhid', 32) * 16,
        use_dynamic_graph=config.get('use_dynamic_graph', False),
        use_phys=config.get('use_phys', False),
        use_social=config.get('use_social', False),
        phys_in_dim=data['phys_dim'],
        social_in_dim=1,
        d_int=config.get('d_int', 16),
        dynamic_rank=config.get('dynamic_rank', 10),
        use_lag_align=config.get('use_lag_align', False),
        lag_window=config.get('lag_window', 12),
        lag_temperature=config.get('lag_temperature', 1.0),
        lag_div_gamma=config.get('lag_div_gamma', 0.1),
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    prediction_batches = []
    with torch.no_grad():
        for x, _, phys, social in data['test_loader'].get_iterator():
            model_input = torch.as_tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
            model_input = torch.nn.functional.pad(model_input, (1, 0, 0, 0))
            output = model(
                model_input,
                phys=torch.as_tensor(phys, dtype=torch.float32, device=device),
                social=torch.as_tensor(social, dtype=torch.float32, device=device),
            )
            prediction_batches.append(data['scaler'].inverse_transform(output).cpu())
    prediction = torch.cat(prediction_batches, dim=0)[:len(data['y_test'])]
    real = torch.as_tensor(data['y_test'][..., 0], dtype=torch.float32)
    mae, mape, rmse = util.metric(prediction.squeeze(-1), real)
    print('Test MAE {:.4f} | MAPE {:.4f} | RMSE {:.4f}'.format(mae, mape, rmse))


if __name__ == '__main__':
    main()
