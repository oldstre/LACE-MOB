import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        if A.dim() == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        elif A.dim() == 4:
            x = torch.einsum('bcvl,bvwl->bcwl', (x, A))
        else:
            raise ValueError('Adjacency tensor must be 2D or 4D, got {}D'.format(A.dim()))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class IntensityEncoder(nn.Module):
    def __init__(self, phys_in_dim, social_in_dim, d_int):
        super(IntensityEncoder, self).__init__()
        self.phys_proj = nn.Conv2d(phys_in_dim, d_int, kernel_size=(1, 1), bias=True) if phys_in_dim > 0 else None
        self.social_proj = nn.Conv2d(social_in_dim, d_int, kernel_size=(1, 1), bias=True) if social_in_dim > 0 else None

    def forward(self, phys=None, social=None):
        h_phys = self.phys_proj(phys) if phys is not None and self.phys_proj is not None else None
        h_soc = self.social_proj(social) if social is not None and self.social_proj is not None else None
        return h_phys, h_soc

class GatedFusion(nn.Module):
    def __init__(self, d_int, use_gate=True):
        super(GatedFusion, self).__init__()
        self.use_gate = use_gate
        self.gate_conv = nn.Conv2d(d_int * 2, d_int, kernel_size=(1, 1), bias=True)

    def forward(self, h_phys=None, h_soc=None):
        if h_phys is None and h_soc is None:
            return None
        if h_phys is None:
            return h_soc
        if h_soc is None:
            return h_phys
        if not self.use_gate:
            return h_phys + h_soc

        gate = torch.sigmoid(self.gate_conv(torch.cat([h_phys, h_soc], dim=1)))
        return gate * h_phys + (1.0 - gate) * h_soc

class BiLACM(nn.Module):
    def __init__(self, d_int, K=6, adaptive=True, temperature=1.0, div_gamma=0.0, meanpool=False):
        super(BiLACM, self).__init__()
        if K < 1:
            raise ValueError('lag window K must be >= 1, got {}'.format(K))
        self.K = K
        self.adaptive = adaptive
        self.meanpool = meanpool
        # 反塌缩: temperature 调 softmax 尖锐度; div_gamma 惩罚 expected_lag 处处相同
        self.temperature = temperature
        self.div_gamma = div_gamma
        self.reg_loss = None

        if meanpool:
            pass
        elif adaptive:
            self.phys_logits = nn.Conv2d(d_int * 2, K, kernel_size=(1, 1), bias=True)
            self.soc_logits = nn.Conv2d(d_int * 2, K, kernel_size=(1, 1), bias=True)
            nn.init.zeros_(self.phys_logits.weight)
            nn.init.zeros_(self.phys_logits.bias)
            nn.init.zeros_(self.soc_logits.weight)
            nn.init.zeros_(self.soc_logits.bias)
            self.phys_logits.bias.data[K - 1] = 2.0
            self.soc_logits.bias.data[K - 1] = 2.0
        else:
            phys_kernel = torch.zeros(K, dtype=torch.float32)
            soc_kernel = torch.zeros(K, dtype=torch.float32)
            phys_kernel[K - 1] = 2.0
            soc_kernel[K - 1] = 2.0
            self.phys_kernel = nn.Parameter(phys_kernel)
            self.soc_kernel = nn.Parameter(soc_kernel)

        lag_values = torch.arange(K - 1, -1, -1, dtype=torch.float32)
        self.register_buffer('lag_values', lag_values.view(1, K, 1, 1))
        self.expected_lag_phys = None
        self.expected_lag_soc = None
        self._shape_logged = False

    def _causal_align_one(self, h, cond, logits_layer=None, kernel_param=None):
        hp = F.pad(h, (self.K - 1, 0))
        windows = hp.unfold(-1, self.K, 1)

        if self.meanpool:
            weights = torch.full(
                (h.size(0), self.K, h.size(2), h.size(3)),
                1.0 / self.K,
                dtype=h.dtype,
                device=h.device,
            )
        elif self.adaptive:
            logits = logits_layer(cond)
            weights = F.softmax(logits / self.temperature, dim=1)
        else:
            weights = F.softmax(kernel_param / self.temperature, dim=0).view(1, self.K, 1, 1)
            weights = weights.expand(h.size(0), self.K, h.size(2), h.size(3))

        expected_lag = torch.sum(self.lag_values.to(weights.device) * weights, dim=1)
        h_aligned = torch.einsum('bcntk,bknt->bcnt', windows, weights)
        return h_aligned, expected_lag, windows, weights

    def forward(self, h_phys=None, h_soc=None):
        if h_phys is None or h_soc is None or self.K == 1:
            self.expected_lag_phys = None
            self.expected_lag_soc = None
            return h_phys, h_soc

        cond = torch.cat([h_phys, h_soc], dim=1)
        h_phys_lag, expected_phys, phys_windows, phys_weights = self._causal_align_one(
            h_phys,
            cond,
            logits_layer=self.phys_logits if self.adaptive and not self.meanpool else None,
            kernel_param=self.phys_kernel if not self.adaptive and not self.meanpool else None,
        )
        h_soc_lag, expected_soc, soc_windows, soc_weights = self._causal_align_one(
            h_soc,
            cond,
            logits_layer=self.soc_logits if self.adaptive and not self.meanpool else None,
            kernel_param=self.soc_kernel if not self.adaptive and not self.meanpool else None,
        )

        self.expected_lag_phys = expected_phys
        self.expected_lag_soc = expected_soc
        # 多样性正则: 惩罚 expected_lag 跨样本/节点/时刻塌成常数 (std→0)
        if self.div_gamma > 0:
            self.reg_loss = -self.div_gamma * (expected_phys.std() + expected_soc.std())
        else:
            self.reg_loss = None
        if not self._shape_logged:
            print('BiLACM shapes:', {
                'cond': tuple(cond.shape),
                'phys_windows': tuple(phys_windows.shape),
                'phys_weights': tuple(phys_weights.shape),
                'soc_windows': tuple(soc_windows.shape),
                'soc_weights': tuple(soc_weights.shape),
                'h_phys_lag': tuple(h_phys_lag.shape),
                'h_soc_lag': tuple(h_soc_lag.shape),
            })
            self._shape_logged = True
        return h_phys_lag, h_soc_lag

class DeltaAGenerator(nn.Module):
    def __init__(self, d_int, rank):
        super(DeltaAGenerator, self).__init__()
        self.src_proj = nn.Conv2d(d_int, rank, kernel_size=(1, 1), bias=True)
        self.dst_proj = nn.Conv2d(d_int, rank, kernel_size=(1, 1), bias=True)
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, h_dis):
        e1 = self.src_proj(h_dis)
        e2 = self.dst_proj(h_dis)
        deltaA = torch.einsum('brnl,brml->bnml', e1, e2)
        deltaA = F.relu(deltaA)
        deltaA = F.softmax(deltaA, dim=2)
        return self.alpha * deltaA

def resize_dynamic_graph(deltaA, target_length):
    if deltaA is None:
        return None
    if deltaA.size(-1) == target_length:
        return deltaA
    b, n, m, l = deltaA.size()
    pooled = F.adaptive_avg_pool1d(deltaA.view(b * n * m, 1, l), target_length)
    return pooled.view(b, n, m, target_length)

def build_dynamic_supports(static_supports, adaptive_support, deltaA):
    supports = list(static_supports) if static_supports is not None else []
    if adaptive_support is None:
        return supports
    if deltaA is None:
        supports.append(adaptive_support)
    else:
        adaptive_expanded = adaptive_support.unsqueeze(0).unsqueeze(-1)
        supports.append(adaptive_expanded + deltaA)
    return supports

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, use_dynamic_graph=False, use_phys=True, use_social=True, use_gate=True, phys_in_dim=1, social_in_dim=1, d_int=16, dynamic_rank=10, use_lag_align=False, lag_window=6, lag_adaptive=True, lag_temperature=1.0, lag_div_gamma=0.0, lag_meanpool=False, use_feat_fusion=False):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.use_dynamic_graph = use_dynamic_graph
        self.use_phys = use_phys
        self.use_social = use_social
        self.use_gate = use_gate
        self.use_lag_align = use_lag_align
        self.lag_meanpool = lag_meanpool
        self.use_feat_fusion = use_feat_fusion
        self.lag_window = lag_window
        self.lag_adaptive = lag_adaptive
        if use_dynamic_graph and use_feat_fusion:
            raise ValueError('use_dynamic_graph and use_feat_fusion are mutually exclusive conditioning modes')
        if lag_meanpool and not (use_phys and use_social):
            raise ValueError('lag_meanpool requires both physical and social inputs')
        if use_feat_fusion and not (use_phys and use_social):
            raise ValueError('use_feat_fusion requires both physical and social inputs')

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.static_supports = list(supports) if supports is not None else None

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        self.intensity_encoder = IntensityEncoder(
            phys_in_dim if use_phys else 0,
            social_in_dim if use_social else 0,
            d_int,
        )
        self.gated_fusion = GatedFusion(d_int, use_gate=use_gate)
        self.lag_align = BiLACM(
            d_int,
            K=lag_window,
            adaptive=lag_adaptive,
            temperature=lag_temperature,
            div_gamma=0.0 if lag_meanpool else lag_div_gamma,
            meanpool=lag_meanpool,
        ) if (use_lag_align or lag_meanpool) else None
        self.delta_a_generator = DeltaAGenerator(d_int, dynamic_rank)
        self.feature_fusion = nn.Conv2d(
            residual_channels + d_int, residual_channels, kernel_size=(1, 1), bias=True
        ) if use_feat_fusion else None



        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def _prepare_exogenous(self, phys=None, social=None):
        phys_tensor = None
        social_tensor = None
        if self.use_phys and phys is not None:
            phys_tensor = phys.permute(0, 3, 2, 1).contiguous()
        if self.use_social and social is not None:
            social_tensor = social.permute(0, 3, 2, 1).contiguous()
        return phys_tensor, social_tensor

    def _compute_adaptive_support(self):
        if not (self.gcn_bool and self.addaptadj):
            return None
        return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

    def _compute_disaster_context(self, phys=None, social=None):
        if not (self.use_dynamic_graph or self.use_feat_fusion) or (phys is None and social is None):
            return None
        h_phys, h_soc = self.intensity_encoder(phys=phys, social=social)
        if self.lag_align is not None and h_phys is not None and h_soc is not None:
            h_phys, h_soc = self.lag_align(h_phys, h_soc)
        return self.gated_fusion(h_phys=h_phys, h_soc=h_soc)

    def get_lag_reg_loss(self):
        """训练循环加进总损失: loss = mae + model.get_lag_reg_loss()。
        未启用多样性正则或无 lag 时返回 0。"""
        if self.lag_align is not None and getattr(self.lag_align, 'reg_loss', None) is not None:
            return self.lag_align.reg_loss
        return 0.0

    def get_lag_stats(self):
        """诊断: expected_lag 的 std/mean/min/max。
        std≈0 = 又塌成常数; std 显著>0 = 真自适应 (比 MAE 更关键的判据)。"""
        stats = {}
        if self.lag_align is not None:
            for name, t in (('phys', getattr(self.lag_align, 'expected_lag_phys', None)),
                            ('social', getattr(self.lag_align, 'expected_lag_soc', None))):
                if t is not None:
                    stats[name] = {
                        'std': float(t.std().item()),
                        'mean': float(t.mean().item()),
                        'min': float(t.min().item()),
                        'max': float(t.max().item()),
                    }
        return stats

    def forward(self, input, phys=None, social=None):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input

        phys_tensor, social_tensor = self._prepare_exogenous(phys=phys, social=social)
        if phys_tensor is not None and phys_tensor.size(-1) != x.size(-1):
            phys_tensor = F.adaptive_avg_pool2d(phys_tensor, (phys_tensor.size(2), x.size(-1)))
        if social_tensor is not None and social_tensor.size(-1) != x.size(-1):
            social_tensor = F.adaptive_avg_pool2d(social_tensor, (social_tensor.size(2), x.size(-1)))

        x = self.start_conv(x)
        disaster_context = self._compute_disaster_context(
            phys=phys_tensor, social=social_tensor
        )
        if self.use_feat_fusion and disaster_context is not None:
            x = self.feature_fusion(torch.cat([x, disaster_context], dim=1))
        skip = 0

        adaptive_support = self._compute_adaptive_support()
        dynamic_delta = (
            self.delta_a_generator(disaster_context)
            if self.use_dynamic_graph and disaster_context is not None
            else None
        )
        base_supports = self.static_supports if self.static_supports is not None else []
        default_supports = build_dynamic_supports(base_supports, adaptive_support, None)

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                current_delta = resize_dynamic_graph(dynamic_delta, x.size(3))
                if self.addaptadj:
                    current_supports = build_dynamic_supports(base_supports, adaptive_support, current_delta)
                    x = self.gconv[i](x, current_supports)
                else:
                    x = self.gconv[i](x, default_supports if default_supports else self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
