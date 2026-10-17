"""Original CCC and RTF modules used by RACFNet.

This file contains only RACFNet-original components. It does not contain the
QSFNet backbone or any other upstream source code.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CCCScale(nn.Module):
    """Cross-Modal Consistency Calibration at one decoder scale."""

    def __init__(self, channels: int = 128, projection_channels: int = 32):
        super().__init__()
        self.project_v = nn.Conv2d(channels, projection_channels, 1)
        self.project_d = nn.Conv2d(channels, projection_channels, 1)
        self.project_t = nn.Conv2d(channels, projection_channels, 1)
        self.quality_v = nn.Conv2d(channels, 1, 1)
        self.consistency_v = nn.Conv2d(2, 1, 1)
        self.consistency_d = nn.Conv2d(2, 1, 1)
        self.consistency_t = nn.Conv2d(2, 1, 1)
        self.self_v = nn.Conv2d(channels + 1, 1, 1)
        self.self_d = nn.Conv2d(channels + 1, 1, 1)
        self.self_t = nn.Conv2d(channels + 1, 1, 1)
        self.reliability_v = nn.Conv2d(3, 1, 1)
        self.reliability_d = nn.Conv2d(3, 1, 1)
        self.reliability_t = nn.Conv2d(3, 1, 1)
        self.residual_v = nn.Conv2d(channels, channels, 1)
        self.residual_d = nn.Conv2d(channels, channels, 1)
        self.residual_t = nn.Conv2d(channels, channels, 1)
        self.gamma_v = nn.Parameter(torch.tensor(0.1))
        self.gamma_d = nn.Parameter(torch.tensor(0.1))
        self.gamma_t = nn.Parameter(torch.tensor(0.1))
        for layer in (self.residual_v, self.residual_d, self.residual_t):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _resize_logit(logit: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if logit.shape[-2:] != size:
            logit = F.interpolate(logit, size=size, mode="bilinear", align_corners=True)
        return logit

    def forward(self, fv, fd, ft, logit_d, logit_t):
        size = fv.shape[-2:]
        qv = torch.sigmoid(self.quality_v(fv))
        qd = torch.sigmoid(self._resize_logit(logit_d, size))
        qt = torch.sigmoid(self._resize_logit(logit_t, size))

        uv = F.normalize(self.project_v(fv), p=2, dim=1, eps=1e-6)
        ud = F.normalize(self.project_d(fd), p=2, dim=1, eps=1e-6)
        ut = F.normalize(self.project_t(ft), p=2, dim=1, eps=1e-6)
        avd = ((uv * ud).sum(dim=1, keepdim=True) + 1.0) / 2.0
        avt = ((uv * ut).sum(dim=1, keepdim=True) + 1.0) / 2.0
        adt = ((ud * ut).sum(dim=1, keepdim=True) + 1.0) / 2.0

        cv = torch.sigmoid(self.consistency_v(torch.cat((avd, avt), dim=1)))
        cd = torch.sigmoid(self.consistency_d(torch.cat((avd, adt), dim=1)))
        ct = torch.sigmoid(self.consistency_t(torch.cat((avt, adt), dim=1)))
        sv = torch.sigmoid(self.self_v(torch.cat((fv, qv), dim=1)))
        sd = torch.sigmoid(self.self_d(torch.cat((fd, qd), dim=1)))
        st = torch.sigmoid(self.self_t(torch.cat((ft, qt), dim=1)))

        rv = torch.sigmoid(self.reliability_v(torch.cat((qv, cv, sv), dim=1)))
        rd = torch.sigmoid(self.reliability_d(torch.cat((qd, cd, sd), dim=1)))
        rt = torch.sigmoid(self.reliability_t(torch.cat((qt, ct, st), dim=1)))
        fv_cal = fv + self.gamma_v * self.residual_v(rv * fv)
        fd_cal = fd + self.gamma_d * self.residual_d(rd * fd)
        ft_cal = ft + self.gamma_t * self.residual_t(rt * ft)
        aux = {
            "Qv": qv, "Qd": qd, "Qt": qt,
            "Avd": avd, "Avt": avt, "Adt": adt,
            "Cv": cv, "Cd": cd, "Ct": ct,
            "Sv": sv, "Sd": sd, "St": st,
            "Rv": rv, "Rd": rd, "Rt": rt,
        }
        return fv_cal, fd_cal, ft_cal, aux


class RTFScale(nn.Module):
    """Reliability-Aware Tri-Modal Fusion at one decoder scale."""

    def __init__(self, channels: int = 128, hidden_channels: int = 32):
        super().__init__()
        self.project_v = nn.Conv2d(channels, hidden_channels, 1)
        self.project_d = nn.Conv2d(channels, hidden_channels, 1)
        self.project_t = nn.Conv2d(channels, hidden_channels, 1)
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels * 3 + 3, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        self.delta = nn.Conv2d(channels, channels, 1)
        self.gamma_d = nn.Parameter(torch.tensor(0.1))
        self.gamma_t = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.delta.weight, mean=0.0, std=1e-3)
        if self.delta.bias is not None:
            nn.init.zeros_(self.delta.bias)

    def forward(self, fv, fd, ft, rv, rd, rt, x_d_base, x_t_base):
        hv = F.relu(self.project_v(fv), inplace=False)
        hd = F.relu(self.project_d(fd), inplace=False)
        ht = F.relu(self.project_t(ft), inplace=False)
        logits = self.weight_predictor(torch.cat((hv, hd, ht, rv, rd, rt), dim=1))
        weights = torch.softmax(logits, dim=1)
        wv, wd, wt = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        weighted = wv * fv + wd * fd + wt * ft
        delta = self.delta(weighted)
        x_d_new = x_d_base + self.gamma_d * delta
        x_t_new = x_t_base + self.gamma_t * delta
        return x_d_new, x_t_new, {
            "Z": logits,
            "W": weights,
            "Wv": wv,
            "Wd": wd,
            "Wt": wt,
            "F_weighted": weighted,
            "Delta": delta,
        }
