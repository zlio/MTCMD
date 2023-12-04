import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from layers.RevIN import RevIN


def FFT_for_Period(x, k=2):
    # [B, T, C]
    B, T, N = x.size()
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, (k // 3)*2)
    frequency_list[0] = 1000
    _, low_list =torch.topk(frequency_list, k // 3, largest = False)
    list = torch.cat([low_list,top_list],0)
    list = list.detach().cpu().numpy()
    list.sort()
    period = x.shape[1] // list
    return period, abs(xf).mean(-1)[:, list]




class MSGRU(nn.Module):
    def __init__(self, input_size: int, dg_hidden_size: int,configs):
        super(MSGRU, self).__init__()
        self.rz_gate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size * 2)
        self.dg_hidden_size = dg_hidden_size
        self.h_candidate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size)
        # self.conv = normal_conv(dg_hidden_size)
        # self.conv2 = normal_conv(dg_hidden_size)
        self.configs=configs
    def init_hidden_state(self, batch_size,length):
        return torch.zeros(batch_size, length, self.configs.d_model)

    def forward(self, inputs, states):

        b, n, c = states.shape
        states = states.to(inputs.device)
        r_z = torch.sigmoid(self.rz_gate(torch.cat([inputs, states], -1)))
        r, z = r_z.split(self.dg_hidden_size, -1)
        h_ = torch.tanh(self.h_candidate(torch.cat([inputs, r * states], -1)))
        new_state = z * states + (1 - z) * h_


        return new_state,new_state

class TemporalCorrelation(nn.Module):
    def __init__(self, configs):
        super(TemporalCorrelation, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.Multi_Scale = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        self.MSGRU = MSGRU(configs.d_model, configs.d_model, configs)
        decomp_kernel=13
        self.decomp = series_decomp(decomp_kernel)
    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []

        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            state = self.MSGRU.init_hidden_state(x.shape[0], out.shape[1])
            out, state = self.MSGRU(out, state)
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out, trend1 = self.decomp(out)
            out = self.Multi_Scale(out)

            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class TemporalCorrelation1(nn.Module):
    def __init__(self, configs):
        super(TemporalCorrelation1, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=1),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=1)
        )
        self.MSGRU = MSGRU(configs.d_model, configs.d_model, configs)
        decomp_kernel=13
        self.decomp = series_decomp(decomp_kernel)

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []

        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            state = self.MSGRU.init_hidden_state(x.shape[0], out.shape[1])
            out, state = self.MSGRU(out, state)
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out, trend1 = self.decomp(out)
            out = self.conv(out)

            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class moving_avg_2d(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg_2d, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool2d(kernel_size=(1, kernel_size), stride=(1, stride), padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1, :].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, :, -1:, :].repeat(1, 1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)
        return x

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg_2d(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

def print_model(model):
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters.')
    return

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TemporalCorrelation(configs)
                                    for _ in range(configs.e_layers)])
        self.model1 = nn.ModuleList([TemporalCorrelation1(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
        decomp_kernel = [13, 17]
        self.decomp_mulit = series_decomp_multi(decomp_kernel)
        self.conv = nn.Conv1d(self.seq_len, self.pred_len + self.seq_len, kernel_size=3, stride=1, padding=1,
                              padding_mode='circular', bias=False)
        self.conv1 = nn.Conv1d(configs.d_model, configs.c_out, kernel_size=3, stride=1, padding=1,
                              padding_mode='circular', bias=False)
        self.revin_layer = RevIN(configs.enc_in)
        self.revin_layer1 = RevIN(configs.dec_in)
        self.dropout = nn.Dropout(0.1)
        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        x_enc = self.revin_layer(x_enc, 'norm')
        seasonal_init, trend_init = self.decomp_mulit(x_enc)
        enc_out = self.enc_embedding(seasonal_init, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)
        enc_out1=enc_out
        enc_out2=enc_out
        for i in range(self.layer):
            enc_out1 = self.dropout(self.layer_norm(self.model[i](enc_out1)))
        for i in range(self.layer):
            enc_out2 = self.dropout(self.layer_norm(self.model1[i](enc_out2)))

        dec_out1 = self.projection(enc_out1)
        trend = self.conv(trend_init)
        dec_out1 = dec_out1 + trend
        dec_out1 = self.revin_layer1(dec_out1, 'denorm')

        dec_out1 = dec_out1 * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out1 = dec_out1 + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        dec_out2 = self.projection(enc_out2)
        dec_out2 = dec_out2 + trend
        dec_out2 = self.revin_layer1(dec_out2, 'denorm')
        dec_out2 = dec_out2 * \
                   (stdev[:, 0, :].unsqueeze(1).repeat(
                       1, self.pred_len + self.seq_len, 1))
        dec_out2 = dec_out2 + \
                   (means[:, 0, :].unsqueeze(1).repeat(
                       1, self.pred_len + self.seq_len, 1))
        return dec_out1,dec_out2



    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        x_enc = self.revin_layer(x_enc, 'norm')
        seasonal_init, trend_init = self.decomp_mulit(x_enc)
        enc_out = self.enc_embedding(seasonal_init, x_mark_enc)  # [B,T,C]
        enc_out1 = enc_out
        enc_out2 = enc_out
        for i in range(self.layer):
            enc_out1 = self.dropout(self.layer_norm(self.model[i](enc_out1)))
        for i in range(self.layer):
            enc_out2 = self.dropout(self.layer_norm(self.model1[i](enc_out2)))
        dec_out1 = self.projection(enc_out1)
        trend = self.conv(trend_init)
        dec_out1 = dec_out1 + trend
        dec_out1 = self.revin_layer1(dec_out1, 'denorm')

        dec_out1 = dec_out1 * \
                   (stdev[:, 0, :].unsqueeze(1).repeat(
                       1, self.pred_len + self.seq_len, 1))
        dec_out1 = dec_out1 + \
                   (means[:, 0, :].unsqueeze(1).repeat(
                       1, self.pred_len + self.seq_len, 1))

        dec_out2 = self.projection(enc_out2)
        dec_out2 = dec_out2 + trend
        dec_out2 = self.revin_layer1(dec_out2, 'denorm')
        dec_out2 = dec_out2 * \
                   (stdev[:, 0, :].unsqueeze(1).repeat(
                       1, self.pred_len + self.seq_len, 1))
        dec_out2 = dec_out2 + \
                   (means[:, 0, :].unsqueeze(1).repeat(
                       1, self.pred_len + self.seq_len, 1))
        return dec_out1,dec_out2


    def classification(self, x_enc, x_mark_enc):
        x_enc = self.enc_embedding(x_enc, None)  # [B,T,C]
        enc_out1 = x_enc
        enc_out2 = x_enc
        for i in range(self.layer):
            enc_out1 = self.layer_norm(self.model[i](enc_out1))
        for i in range(self.layer):
            enc_out2 = self.layer_norm(self.model[i](enc_out2))
        output1 = self.act(enc_out1)
        output1 = self.dropout(output1)
        output2 = self.act(enc_out2)
        output2= self.dropout(output2)
        output1 = output1 * x_mark_enc.unsqueeze(-1)
        output1 = output1.reshape(output1.shape[0], -1)
        output1 = self.projection(output1)
        output2 = output2 * x_mark_enc.unsqueeze(-1)
        output2 = output2.reshape(output2.shape[0], -1)
        output2 = self.projection(output2)
        return output1,output2

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out1,dec_out2 = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out1[:, -self.pred_len:, :],dec_out2[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
