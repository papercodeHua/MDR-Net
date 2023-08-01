import torch


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = (torch.diag(h).sum() / h.sum()).item()
        se = ((torch.diag(h) / h.sum(1))[1]).item()
        sp = ((torch.diag(h) / h.sum(1))[0]).item()
        pr = ((torch.diag(h) / h.sum(0))[1]).item()
        F1 = 2 * (pr * se) / (pr + se)
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        mIou = iou.mean().item()
        return acc_global, se, sp, F1, mIou


