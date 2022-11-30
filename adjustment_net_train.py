# python imports
import os
import argparse
import time
import sys

# torch imports
import torch
from torch.utils.data import DataLoader

# for visualization
from torch.utils.tensorboard import SummaryWriter

from custom_dataloader import LOLLoader
from utils import AverageMeter, count_parameters, save_tensor_to_image, pad_to, unpad

from decomposition_model import DecompositionNet
from adjustment_model import AdjustmentNet
from adjustment_loss import AdjustmentLoss
from torch.optim import lr_scheduler

import custom_transforms as transforms

import cv2
import numpy as np


parser = argparse.ArgumentParser(description="Low-light Image Enhancement")
parser.add_argument("data_folder", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--epochs", default=180, type=int, metavar="E", help="number of total epochs to run"
)
parser.add_argument(
    "--warmup", default=20, type=int, metavar="W", help="number of warmup epochs"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="E0",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0001,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p", "--print-freq", default=10, type=int, help="print frequency (default: 10)"
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--decomp",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "-b", "--batch-size", default=10, type=int, help="batch size (default: 10)"
)
parser.add_argument(
    "--patch-size", default=48, type=int, help="patch size (default: 48)"
)

def main(args):
    # using mps or gpu or cpu
    device = "mps" if getattr(torch,'has_mps',False) \
            else "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"=> device: {device}")

    # set up model and loss
    model_arch = "AdjustmentNet"
    model_decomposition = DecompositionNet()
    model_adjustment = AdjustmentNet()
    model_decomposition = model_decomposition.to(device)
    model_adjustment = model_adjustment.to(device)
    criterion = AdjustmentLoss(device)
    criterion = criterion.to(device)

    # setup optimizer and scheduler
    optimizer = torch.optim.Adam(model_adjustment.parameters(), lr=0.0001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

    # load decomposition net parameters
    if not os.path.isfile(args.decomp):
        print(f"=> no parameters for decomposition net, exit!")
        sys.exit(-1)
    checkpoint = torch.load(args.decomp)
    model_decomposition.load_state_dict(checkpoint["state_dict"])
    print(f"=> loading parameters for decomposition net '{args.decomp}'")

    # resume from a checkpoint for adjustment net
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model_adjustment.load_state_dict(checkpoint["state_dict"])
            # only load the optimizer if necessary
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    print(f"# parameters: {count_parameters(model_adjustment)}")

    # set up transforms for data augmentation
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    # set up dataset and dataloader
    train_dataset = LOLLoader(args.data_folder, split="train", is_train=True, transforms=train_transforms, patch_size=args.patch_size)
    test_dataset = LOLLoader(args.data_folder, split="test", is_train=False, transforms=test_transforms, patch_size=args.patch_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False)

    # evaluation
    if args.resume and args.evaluate:
        print("Testing the model ...")
        # cudnn.deterministic = True
        validate(test_loader, model_decomposition, model_adjustment, -1, args, device, path="./adjustment_testing_images")
        return

    # start training
    print("Training the model ...")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model_decomposition, model_adjustment, criterion, optimizer, scheduler, epoch, args, device)

        # save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_arch": model_arch,
                "state_dict": model_adjustment.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            file_folder="./models_adjustment/"
        )

        # validation during training
        if epoch % args.print_freq == 0:
            validate(test_loader, model_decomposition, model_adjustment, epoch, args, device, "./adjustment_intermediate_testing_images")

def train(train_loader, model_decomposition, model_adjustment, criterion, optimizer, scheduler, epoch, args, device):
    losses = AverageMeter()
    batch_time = AverageMeter()

    model_decomposition.eval()
    model_adjustment.train()
    for i, (img_high, img_low) in enumerate(train_loader):
        end = time.time()
        if i == 0:
            print(f"[Train] img_high.shape: {img_high.shape}, img_low.shape: {img_low.shape}")

        # zero gradient
        optimizer.zero_grad()

        # put data to device
        img_low = img_low.to(device)
        img_high = img_high.to(device)

        # compute output
        illu_low, illu_high = img_low, img_high
        with torch.no_grad():
            reflect_high, illu_high = model_decomposition(img_high)
            reflect_low, illu_low = model_decomposition(img_low)
            bright_low = torch.mean(illu_low)
            bright_high = torch.mean(illu_high)
            ratio_high_to_low = torch.div(bright_low, bright_high)
            ratio_low_to_high = torch.div(bright_high, bright_low)
        
        adjustment_output_low_to_high = model_adjustment(illu_low, ratio_low_to_high)
        adjustment_output_high_to_low = model_adjustment(illu_high, ratio_high_to_low)
        loss = criterion(adjustment_output_low_to_high, illu_high) + criterion(adjustment_output_high_to_low, illu_low)

        # record loss
        losses.update(loss.item(), img_low.shape[0])

        # compute gradient and do one ADAM step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.2f} ({loss.avg:.2f})".format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                )
            )

    scheduler.step()

    # print the learning rate
    print(
        "[Training]: Epoch {:d} finished with lr={:f}".format(
            epoch + 1, scheduler.get_last_lr()[0]
        )
    )

def validate(test_loader, model_decomposition, model_adjustment, epoch, args, device, path="./testing_image"):
    """Test the model on the validation set"""

    epoch_info = ""
    if epoch > -1:
        epoch_info = f"_epoch_{epoch}"
    if not os.path.exists(path):
        os.mkdir(path)

    # switch to evaluate mode (autograd will still track the graph!)
    model_decomposition.eval()
    model_adjustment.eval()
    with torch.no_grad():
        # loop over validation set
        for i, (img_high, img_low) in enumerate(test_loader):
            if i == 0:
                print(f"[Testing] img_low.shape: {img_low.shape}")

            img_low = img_low.to(device)
            img_high = img_high.to(device)

            # compute output
            reflect_low, illu_low = model_decomposition(img_low)
            reflect_high, illu_high = model_decomposition(img_high)
            bright_low = torch.mean(illu_low)
            bright_high = torch.mean(illu_high)
            ratio_high_to_low = torch.div(bright_low, bright_high)
            ratio_low_to_high = torch.div(bright_high, bright_low)
            
            adjustment_output_low_to_high = model_adjustment(illu_low, ratio_low_to_high)
            adjustment_output_high_to_low = model_adjustment(illu_high, ratio_high_to_low)
            adjustment_output = torch.cat([illu_low, illu_high, adjustment_output_high_to_low, adjustment_output_low_to_high], dim=1)
            if i == 0:
                print(f"[Testing] adjustment_output.shape: {adjustment_output.shape}")

            # store output
            for j in range(adjustment_output.shape[0]):
                if epoch == -1:
                    save_tensor_to_image(f"{path}/input_{j}{epoch_info}.jpg", img_low[j])
                save_tensor_to_image(f"{path}/adjustment_{j}{epoch_info}.jpg", adjustment_output[j])


def save_checkpoint(
    state, file_folder="./models_adjustment/", filename="checkpoint.pth.tar"
):
    """save checkpoint"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, filename))
    # if is_best:
    #     # skip the optimization state
    #     state.pop("optimizer", None)
    #     torch.save(state, os.path.join(file_folder, "model_best.pth.tar"))

def get_train_transforms():
    train_transforms = []
    train_transforms.append(transforms.ToTensor())
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms

def get_test_transforms():
    val_transforms = []
    val_transforms.append(transforms.ToTensor())
    val_transforms = transforms.Compose(val_transforms)
    return val_transforms

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)