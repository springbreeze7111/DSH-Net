import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=None, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        # 确保 target 是 int64 (torch.long) 类型
        target = target.long()  # 关键修复：将 target 转换为 long 类型
        log_pt = F.log_softmax(pred, dim=1)
        log_pt = log_pt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)
        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            focal_loss = focal_loss[mask]
        return focal_loss.mean()