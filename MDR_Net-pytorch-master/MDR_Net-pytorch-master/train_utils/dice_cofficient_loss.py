import torch


def dice_loss(x: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1).float()
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter
        d += (2 * inter + epsilon) / (sets_sum + epsilon)
    return 1 - d / batch_size
