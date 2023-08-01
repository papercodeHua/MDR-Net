import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
from .pad import InputPadder
from .disturtd_utils import ConfusionMatrix
from .dice_cofficient_loss import dice_loss


def criterion(inputs, target, dice: bool = True, bce: bool = True):
    loss1 = 0
    if dice:
        loss1 = dice_loss(inputs, target)
    loss2 = 0
    if bce:
        target = target.unsqueeze(1).float()
        loss2 = nn.BCELoss()(inputs, target)
    return loss1 + loss2


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes + 1)
    data_loader = tqdm(data_loader)
    mask = None
    predict = None
    with torch.no_grad():
        for image, target in data_loader:
            padder = InputPadder(image.shape)
            image, target = padder.pad(image, target)
            image, target = image.to(device), target.to(device)
            # (B,1,H,W)
            output = model(image)
            truth = output.clone()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            confmat.update(target.flatten(), output.long().flatten())
            # dice.update(output, target)
            mask = target.flatten() if mask is None else torch.cat((mask, target.flatten()))
            predict = truth.flatten() if predict is None else torch.cat((predict, truth.flatten()))

    mask = mask.cpu().numpy()
    predict = predict.cpu().numpy()
    assert mask.shape == predict.shape, f"Dimension mismatch"
    AUC_ROC = roc_auc_score(mask, predict)

    return confmat.compute()[0], confmat.compute()[1], confmat.compute()[2], confmat.compute()[3], confmat.compute()[
        4], AUC_ROC


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler,
                    scaler=None):
    model.train()
    total_loss = 0

    data_loader = tqdm(data_loader)
    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, True, True)
        total_loss += loss.item()

        data_loader.set_description(f"Epoch[{epoch}/200]-train,train_loss:{loss.item()}")
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # chasedb1
        scheduler.step()
    return total_loss / len(data_loader)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
         Returns a learning rate multiplier factor based on the number of steps,
         Note that pytorch calls the lr_scheduler.step() method once before the training begins
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
