import torch
import torch.nn as nn
import torch.optim as optim

import util
from model.model import gwnet


class Trainer:
    def __init__(self, scaler, metadata, args, device, supports, adjinit):
        self.model = gwnet(
            device,
            metadata['num_nodes'],
            args.dropout,
            supports=supports,
            gcn_bool=args.gcn_bool,
            addaptadj=args.addaptadj,
            aptinit=adjinit,
            in_dim=metadata['in_dim'],
            out_dim=metadata['seq_length'],
            residual_channels=args.nhid,
            dilation_channels=args.nhid,
            skip_channels=args.nhid * 8,
            end_channels=args.nhid * 16,
            use_dynamic_graph=args.use_dynamic_graph,
            use_phys=args.use_phys,
            use_social=args.use_social,
            phys_in_dim=args.phys_in_dim,
            social_in_dim=1,
            d_int=args.d_int,
            dynamic_rank=args.dynamic_rank,
            use_lag_align=args.use_lag_align,
            lag_window=args.lag_window,
            lag_temperature=args.lag_temperature,
            lag_div_gamma=args.lag_div_gamma,
        ).to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        self.scaler = scaler

    def step(self, x, y, phys, social, training):
        self.model.train(training)
        if training:
            self.optimizer.zero_grad()
        x = nn.functional.pad(x, (1, 0, 0, 0))
        prediction = self.model(x, phys=phys, social=social).transpose(1, 3)
        real = y.unsqueeze(1)
        prediction = self.scaler.inverse_transform(prediction)
        loss = util.masked_mae(prediction, real, 0.0)
        if training:
            lag_loss = self.model.get_lag_reg_loss()
            if torch.is_tensor(lag_loss):
                loss = loss + lag_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
        return (
            float(loss.item()),
            float(util.masked_mape(prediction, real, 0.0).item()),
            float(util.masked_rmse(prediction, real, 0.0).item()),
        )

    def predict(self, x, phys, social):
        self.model.eval()
        x = nn.functional.pad(x, (1, 0, 0, 0))
        prediction = self.model(x, phys=phys, social=social)
        return self.scaler.inverse_transform(prediction)
