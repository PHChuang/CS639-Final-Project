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
from adjustment_model import AdjustmentNet

import custom_transforms as transforms

from torchmetrics import StructuralSimilarityIndexMeasure

from ignite.engine import *
from ignite.metrics import *


parser = argparse.ArgumentParser(description="Low-light Image Enhancement")
parser.add_argument("data_folder", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--decomp",
    default="",
    type=str,
    metavar="PATH",
    help="path to decomposition net (default: none)",
)
parser.add_argument(
    "--restore",
    default="",
    type=str,
    metavar="PATH",
    help="path to restoration net (default: none)",
)
parser.add_argument(
    "--adjust",
    default="",
    type=str,
    metavar="PATH",
    help="path to adjustment net (default: none)",
)
parser.add_argument(
    "--ratio",
    default=5.0,
    type=float,
    help="ratio for illumination",
)


def main(args):
    # using mps or gpu or cpu
    device = "mps" if getattr(torch,'has_mps',False) \
            else "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"=> device: {device}")

    # set up model and loss
    model_decomposition = DecompositionNet()
    model_restoration = RestorationNet()
    model_adjustment = AdjustmentNet()
    model_decomposition = model_decomposition.to(device)
    model_restoration = model_restoration.to(device)
    model_adjustment = model_adjustment.to(device)

    # load models
    if not os.path.isfile(args.decomp):
        print(f"=> no parameters for decomposition net, exit!")
        sys.exit(-1)
    checkpoint = torch.load(args.decomp)
    model_decomposition.load_state_dict(checkpoint["state_dict"])
    print(f"=> loading parameters for decomposition net '{args.decomp}'")

    if not os.path.isfile(args.restore):
        print(f"=> no parameters for restoration net, exit!")
        sys.exit(-1)
    checkpoint = torch.load(args.restore)
    model_restoration.load_state_dict(checkpoint["state_dict"])
    print(f"=> loading parameters for restoreation net '{args.restore}'")

    if not os.path.isfile(args.adjust):
        print(f"=> no parameters for adjustment net, exit!")
        sys.exit(-1)
    checkpoint = torch.load(args.adjust)
    model_adjustment.load_state_dict(checkpoint["state_dict"])
    print(f"=> loading parameters for adjustment net '{args.adjust}'")

    # TODO: enable cudnn benchmark?

    # set up transforms for data augmentation
    test_transforms = get_test_transforms()

    # set up dataset and dataloader
    test_dataset = LOLLoader(args.data_folder, split="test", is_train=False, transforms=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=15, pin_memory=True, shuffle=False)

    print(f"Testing the model with ratio = {args.ratio} ...")
    # cudnn.deterministic = True
    validate(test_loader, model_decomposition, model_restoration, model_adjustment, args, device, path="./evaluation_images")
    return


def validate(test_loader, model_decomposition, model_restoration, model_adjustment, args, device, path="./testing_image"):
    """Test the model on the validation set"""

    if not os.path.exists(path):
        os.mkdir(path)

    # SSIM evaluator
    ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)

    # PSNR evaluator
    default_evaluator = Engine(eval_step)
    psnr = PSNR(data_range=1.0)
    psnr.attach(default_evaluator, 'psnr')

    # switch to evaluate mode (autograd will still track the graph!)
    model_decomposition.eval()
    model_restoration.eval()
    model_adjustment.eval()
    with torch.no_grad():
        # loop over validation set
        for i, (img_high, img_low) in enumerate(test_loader):
            if i == 0:
                print(f"[Testing] img_low.shape: {img_low.shape}")

            # padding for input to avoid odd resolution issue
            img_low, pads = pad_to(img_low, 16)
            img_low = img_low.to(device)

            img_high = img_high.to(device)

            # compute output
            reflect_low, illu_low = model_decomposition(img_low)
            restoration_output = model_restoration(reflect_low, illu_low)

            img_low = unpad(img_low, pads)
            restoration_output = unpad(restoration_output, pads)
            illu_low = unpad(illu_low, pads)

            adjustment_output = model_adjustment(illu_low, args.ratio)
            adjustment_output = torch.cat([adjustment_output, adjustment_output, adjustment_output], dim=1)
            final_output = adjustment_output * restoration_output

            # store output
            for j in range(final_output.shape[0]):
                save_tensor_to_image(f"{path}/input_{j}_ratio{args.ratio}.jpg", img_low[j])
                save_tensor_to_image(f"{path}/decomp_r_{j}_ratio{args.ratio}.jpg", reflect_low[j])
                save_tensor_to_image(f"{path}/decomp_i_{j}_ratio{args.ratio}.jpg", illu_low[j])
                save_tensor_to_image(f"{path}/restoration_{j}_ratio{args.ratio}.jpg", restoration_output[j])
                save_tensor_to_image(f"{path}/adjustment_{j}_ratio{args.ratio}.jpg", adjustment_output[j])
                save_tensor_to_image(f"{path}/final_{j}_ratio{args.ratio}.jpg", final_output[j])

            # output SSIM
            print(f"SSIM (before): {ssim(img_low, img_high)}")
            print(f"SSIM (after): {ssim(final_output, img_high)}")

            # output PSNR
            state = default_evaluator.run([[img_low.detach().cpu(), img_high.detach().cpu()]])
            print(f"PSNR (before): {state.metrics['psnr']}")
            state = default_evaluator.run([[final_output.detach().cpu(), img_high.detach().cpu()]])
            print(f"PSNR (after): {state.metrics['psnr']}")


def get_test_transforms():
    val_transforms = []
    val_transforms.append(transforms.ToTensor())
    val_transforms = transforms.Compose(val_transforms)
    return val_transforms


def eval_step(engine, batch):
    return batch


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)