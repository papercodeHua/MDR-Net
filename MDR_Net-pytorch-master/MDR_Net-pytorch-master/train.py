import os
import datetime
import torch
import torch.utils.data
import compute_mean_std
import transforms as T
from datasets import DriveDataset
from model.MDR_Net import MDR_Net
from train_utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from torch import nn


class SegmentationPresetTrain:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    crop_size = 256  # DRIVE:256 0.18 0.13  CHASEDB1:416 0.13 STARE:256 0.9  HRF:1008 0.9

    if train:
        return SegmentationPresetTrain(crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = MDR_Net()
    return model


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes

    model = create_model(num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0])
    model.to(device)

    # using compute_mean_std.py
    # mean, std = compute_mean_std.compute()
    # train
    mean = (0.212, 0.212, 0.212)
    std = (0.157, 0.157, 0.157)
    # test
    mean1 = (0.215, 0.215, 0.215)
    std1 = (0.151, 0.151, 0.151)
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # DRIVE
    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean1, std=std1))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=4,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    # (Initialize logging)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_metric = {"AUC_ROC": 0.5}
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scheduler,
                                    scaler=scaler)
        # drive
        # scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        acc, se, sp, F1, mIou, AUC_ROC = evaluate(model, val_loader, device=device, num_classes=num_classes)
        # print('FLOPs = ' + str(flops/1000**3) + 'G')
        # print('Params = ' + str(params/1000**2) + 'M')
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"AUC: {AUC_ROC:.6f}\n" \
                         f"acc: {acc:.6f}\n" \
                         f"se: {se:.6f}\n" \
                         f"sp: {sp:.6f}\n" \
                         f"F1: {F1:.6f}\n" \
                         f"mIou: {mIou:.6f}\n"
            f.write(train_info + "\n\n")
        print(f"AUC: {AUC_ROC:.6f}")
        print(f"acc: {acc:.6f}")
        print(f"se: {se:.6f}")
        print(f"sp: {sp:.6f}")
        print(f"mIou: {mIou:.6f}")
        print(f"F1: {F1:.6f}")

        if args.save_best is True:
            if best_metric["AUC_ROC"] < AUC_ROC:
                best_metric["AUC_ROC"] = AUC_ROC
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch SA-UNET training")
    parser.add_argument("--data-path", default="DRIVE", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", help="training device")  #
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    # parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)',
    #                     dest='weight_decay')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--early_stop', default=35, type=int)
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use pytorch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
