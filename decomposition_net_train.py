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
from utils import AverageMeter, count_parameters, save_tensor_to_image

# from model import DecompositionNet
from decomposition_model import DecompositionNet
from decomposition_loss import DecompositionLoss
from torch.optim import lr_scheduler

import custom_transforms as transforms


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
    default=0.0004,
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
parser.add_argument(
    "--test-gradient",
    dest="test_gradient",
    action="store_true",
    help="test gradient operator",
)


def main(args):
    # using mps or gpu or cpu
    device = "mps" if getattr(torch,'has_mps',False) \
            else "gpu" if torch.cuda.is_available() else "cpu"
    device = "cpu" # 48 x 48 patch runs faster on cpu
    device = torch.device(device)
    print(f"=> device: {device}")

    # set up model and loss
    model_arch = "DecompositionNet"
    model = DecompositionNet()
    model = model.to(device)
    criterion = DecompositionLoss(device)
    criterion = criterion.to(device)

    # setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
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
    print(f"# parameters: {count_parameters(model)}")

    # TODO: enable cudnn benchmark?

    # set up transforms for data augmentation
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    # set up dataset and dataloader
    train_dataset = LOLLoader(args.data_folder, split="train", is_train=True, transforms=train_transforms, patch_size=args.patch_size)
    test_dataset = LOLLoader(args.data_folder, split="test", is_train=False, transforms=test_transforms, patch_size=args.patch_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=15, pin_memory=True, shuffle=False)

    # test gradient
    if args.test_gradient:
        print("Testing gradient ...")
        test_gradient(test_loader, device)
        return

    # evaluation
    if args.resume and args.evaluate:
        print("Testing the model ...")
        # cudnn.deterministic = True
        validate(test_loader, model, -1, args, device, path="decomposition_testing_image")
        return

    # start training
    print("Training the model ...")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, None, epoch, args, device)

        # save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_arch": model_arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            file_folder="./models_decomp/",
        )

        # validation during training
        if epoch % args.print_freq == 0:
            validate(test_loader, model, epoch, args, device, "./decomposition_intermediate_testing_image")

def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, device):
    losses = AverageMeter()
    batch_time = AverageMeter()

    model.train()
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
        reflect_high, illu_high = model(img_high)
        reflect_low, illu_low = model(img_low)
        loss = criterion(reflect_low, reflect_high, illu_low, illu_high, img_low, img_high)

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

    # scheduler.step()

    # print the learning rate
    # print(
    #     "[Training]: Epoch {:d} finished with lr={:f}".format(
    #         epoch + 1, scheduler.get_last_lr()[0]
    #     )
    # )

    print(
        "[Training]: Epoch {:d}".format(
            epoch + 1
        )
    )

def validate(test_loader, model, epoch, args, device, path="./testing_image"):
    """Test the model on the validation set"""

    epoch_info = ""
    if epoch > -1:
        epoch_info = f"_epoch_{epoch}"
    if not os.path.exists(path):
        os.mkdir(path)

    model.eval()
    with torch.no_grad():
        # loop over validation set
        for i, (img_high, img_low) in enumerate(test_loader):
            if i == 0:
                print(f"[Testing] img_high.shape: {img_high.shape}, img_low.shape: {img_low.shape}")

            img_high = img_high.to(device)
            img_low = img_low.to(device)

            # compute output
            reflect_high, illu_high = model(img_high)
            reflect_low, illu_low = model(img_low)
            if i == 0:
                print(f"[Testing] reflect_low.shape: {reflect_low.shape}, illu_low.shape: {illu_low.shape}")

            # store output
            for j in range(reflect_low.shape[0]):
                save_tensor_to_image(f"{path}/img_low_{j}{epoch_info}.jpg", img_low[j])
                save_tensor_to_image(f"{path}/reflect_low_{j}{epoch_info}.jpg", reflect_low[j])
                save_tensor_to_image(f"{path}/illu_low_{j}{epoch_info}.jpg", illu_low[j])
                save_tensor_to_image(f"{path}/img_high_{j}{epoch_info}.jpg", img_high[j])
                save_tensor_to_image(f"{path}/reflect_high_{j}{epoch_info}.jpg", reflect_high[j])
                save_tensor_to_image(f"{path}/illu_high_{j}{epoch_info}.jpg", illu_high[j])


def test_gradient(test_loader, device, path="./test_gradient"):
    from torchvision.transforms.functional import rgb_to_grayscale
    from utils import gradient

    if not os.path.exists(path):
        os.mkdir(path)

    for i, (img_high, img_low) in enumerate(test_loader):
        if i == 0:
            print(f"[Testing] img_high.shape: {img_high.shape}, img_low.shape: {img_low.shape}")

        img_high = img_high.to(device)
        img_low = img_low.to(device)

        img_high_gray = rgb_to_grayscale(img_high)
        img_high_gradient_x = gradient(img_high_gray, "x", device)
        img_high_gradient_y = gradient(img_high_gray, "y", device)

        img_low_gray = rgb_to_grayscale(img_low)
        img_low_gradient_x = gradient(img_low_gray, "x", device)
        img_low_gradient_y = gradient(img_low_gray, "y", device)

        print(f"[Testing] img_low_gradient_x.shape: {img_low_gradient_x.shape}, img_low_gradient_y.shape: {img_low_gradient_y.shape}")

        for j in range(img_high_gray.shape[0]):
            save_tensor_to_image(f"{path}/img_low_gray_{j}.jpg", img_low_gray[j])
            save_tensor_to_image(f"{path}/img_low_gradient_x_{j}.jpg", img_low_gradient_x[j])
            save_tensor_to_image(f"{path}/img_low_gradient_y_{j}.jpg", img_low_gradient_y[j])
            save_tensor_to_image(f"{path}/img_high_gray_{j}.jpg", img_high_gray[j])
            save_tensor_to_image(f"{path}/img_high_gradient_x_{j}.jpg", img_high_gradient_x[j])
            save_tensor_to_image(f"{path}/img_high_gradient_y_{j}.jpg", img_high_gradient_y[j])


def save_checkpoint(
    state, file_folder="./models/", filename="checkpoint.pth.tar"
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