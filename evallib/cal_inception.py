from __future__ import print_function

import argparse
import os
import random
from collections import OrderedDict
# import metrics

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.nn.utils import spectral_norm
from tqdm import tqdm

from utils import add_channel, compute_fid, compute_kid, compute_metrics

from download import download_celeb_a

def main(opt):

    # if opt.dataset in ["imagenet", "folder", "lfw"]:
        # folder dataset
    real_dataset = dset.ImageFolder(
        root=opt.real_dataroot,
        # change image size and value(-1,1)
        transform=transforms.Compose(
            [
                #transforms.Resize(opt.imageSize),
                #transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    fake_dataset = dset.ImageFolder(
        root=opt.fake_dataroot,
        # change image size and value(-1,1)
        transform=transforms.Compose(
            [
                #transforms.Resize(opt.imageSize),
                #transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    assert real_dataset
    real_dataloader = torch.utils.data.DataLoader(
        real_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers)
    )

    assert fake_dataset
    fake_dataloader = torch.utils.data.DataLoader(
        fake_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers)
    )

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    # ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    
    augmentation_level = opt.al
    

    global_step = 0
    last_augmentation_step = 0
    kid_score_history = []

    
    r_samples = random.sample(range(len(real_dataset)), opt.kid_batch)
    real_samples = [real_dataset[r][0] for r in r_samples]

    real_samples = torch.stack(real_samples, dim=0).to(device)
    f_samples = random.sample(range(len(fake_dataset)), opt.fid_batch)
    fake_samples = [fake_dataset[f][0] for f in f_samples]
    fake_samples = torch.stack(fake_samples, dim=0).to(device)
    
    print("Computing KID and FID...")                
    kid, fid = compute_metrics(real_samples, fake_samples)
    print("FID: {:.4f}".format(fid))
    # writer.add_scalar("metrics/fid", fid, global_step)
    print("KID: {:.4f}".format(kid))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     dest="dataset", help="cifar10 | lsun | imagenet | folder | lfw | mnist | fake"
    # )
    parser.add_argument("--real_dataroot", default="data/real_datasets", help="path to real dataset")
    parser.add_argument("--fake_dataroot", default="data/fake_datasets", help="path to fake dataset")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch-size", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--imageSize",
        type=int,
        default=256,
        help="the height / width of the input image to network",
    )
    parser.add_argument(
        "--nz", type=int, default=128, help="size of the latent z vector"
    )
    parser.add_argument(
        "--tr", type=int, default=5000, help="steps for p to reach 0.5"
    )
    parser.add_argument(
        "--al", type=int, default=0, help="starting augmentation level"
    )
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument(
        "--kid_batch", default=266, type=int, help="how many images to use to compute kid"
    )
    parser.add_argument(
        "--fid_batch", default=266, type=int, help="how many images to use to compute fid"
    )

    opt = parser.parse_args()
    print(opt)

    cudnn.benchmark = True

    if not torch.cuda.is_available() or opt.cpu:
        opt.cuda = False
    else:
        opt.cuda = True

    main(opt)
