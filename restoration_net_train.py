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
from restoration_model import RestorationNet
from restoration_loss import RestorationLoss
from torch.optim import lr_scheduler

import custom_transforms as transforms

import cv2
import numpy as np

from datetime import datetime

parser = argparse.ArgumentParser(description="Low-light Image Enhancement")
parser.add_argument("data_folder", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--epochs", default=90, type=int, metavar="E", help="number of total epochs to run"
)
parser.add_argument(
    "--warmup", default=5, type=int, metavar="W", help="number of warmup epochs"
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
    default=0.1,
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
    "--patch-size", default=128, type=int, help="patch size (default: 128)"
)

def main(args):
    # using mps or gpu or cpu
    device = "mps" if getattr(torch,'has_mps',False) \
            else "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"=> device: {device}")

    # set up model and loss
    model_arch = "RestorationNet"
    model_decomposition = DecompositionNet()
    model_restoration = RestorationNet()
    model_decomposition = model_decomposition.to(device)
    model_restoration = model_restoration.to(device)
    criterion = RestorationLoss(device)
    criterion = criterion.to(device)

    # setup optimizer and scheduler
    optimizer = torch.optim.Adam(model_restoration.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

    # load decomposition net parameters
    if not os.path.isfile(args.decomp):
        print(f"=> no parameters for decomposition net, exit!")
        sys.exit(-1)
    checkpoint = torch.load(args.decomp)
    model_decomposition.load_state_dict(checkpoint["state_dict"])
    print(f"=> loading parameters for decomposition net '{args.decomp}'")

    # resume from a checkpoint for restoration net
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model_restoration.load_state_dict(checkpoint["state_dict"])
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
    print(f"# parameters: {count_parameters(model_restoration)}")

    # TODO: enable cudnn benchmark?

    # set up transforms for data augmentation
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    # set up dataset and dataloader
    train_dataset = LOLLoader(args.data_folder, split="train", is_train=True, transforms=train_transforms, patch_size=args.patch_size)
    test_dataset = LOLLoader(args.data_folder, split="test", is_train=False, transforms=test_transforms, patch_size=args.patch_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=15, pin_memory=True, shuffle=False)

    # evaluation
    if args.resume and args.evaluate:
        print("Testing the model ...")
        # cudnn.deterministic = True
        validate(test_loader, model_decomposition, model_restoration, -1, args, device, path="./restoration_testing_images")
        return

    # start training
    print("Training the model ...")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model_decomposition, model_restoration, criterion, optimizer, scheduler, epoch, args, device)

        # save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_arch": model_arch,
                "state_dict": model_restoration.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            file_folder="./models_restoration/"
        )

        # validation during training
        if epoch % args.print_freq == 0:
            validate(test_loader, model_decomposition, model_restoration, epoch, args, device, "./restoration_intermediate_testing_images")

def train(train_loader, model_decomposition, model_resotration, criterion, optimizer, scheduler, epoch, args, device):
    losses = AverageMeter()
    batch_time = AverageMeter()

    model_decomposition.eval()
    model_resotration.train()
    print(f"[Training] current time: {datetime.now()}")
    for i, (img_high, img_low) in enumerate(train_loader):
        end = time.time()
        if i == 0:
            print(f"[Training] img_high.shape: {img_high.shape}, img_low.shape: {img_low.shape}")

        # zero gradient
        optimizer.zero_grad()

        # put data to device
        img_high = img_high.to(device)
        img_low = img_low.to(device)

        # compute output
        with torch.no_grad():
            reflect_high, _ = model_decomposition(img_high)
            reflect_low, illu_low = model_decomposition(img_low)
        restoration_output = model_resotration(reflect_low, illu_low)
        loss = criterion(restoration_output, reflect_high)

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

def validate(test_loader, model_decomposition, model_restoration, epoch, args, device, path="./testing_image"):
    """Test the model on the validation set"""

    epoch_info = ""
    if epoch > -1:
        epoch_info = f"_epoch_{epoch}"
    if not os.path.exists(path):
        os.mkdir(path)

    # switch to evaluate mode (autograd will still track the graph!)
    model_decomposition.eval()
    model_restoration.eval()
    with torch.no_grad():
        # loop over validation set
        for i, (_, img_low) in enumerate(test_loader):
            # padding for input to avoid odd resolution issue
            img_low, pads = pad_to(img_low, 16)

            if i == 0:
                print(f"[Testing] img_low.shape: {img_low.shape}")

            img_low = img_low.to(device)

            # compute output
            reflect_low, illu_low = model_decomposition(img_low)
            restoration_output = model_restoration(reflect_low, illu_low)
            restoration_output = unpad(restoration_output, pads)
            if i == 0:
                print(f"[Testing] restoration_output.shape: {restoration_output.shape}")

            # store output
            for j in range(restoration_output.shape[0]):
                if epoch == -1:
                    save_tensor_to_image(f"{path}/input_{j}{epoch_info}.jpg", img_low[j])
                save_tensor_to_image(f"{path}/restoration_{j}{epoch_info}.jpg", restoration_output[j])


def save_checkpoint(
    state, file_folder="./models_restoration/", filename="checkpoint.pth.tar"
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