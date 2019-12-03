import numpy as np
import torch
from torchvision import transforms
from torch import nn, optim
from datetime import datetime
import os

import tensorboardX
from data_loader import *
from tensorboardX import SummaryWriter
from torchvision import utils
from vae_cat import *

def test(data_loader, model, out_dir):
    model.eval()
    with torch.no_grad():
        for idx, img in enumerate(data_loader):
            out, _ = model(img)
            out, mu, logvar = model(img)
            z = [torch.randn((512, 32, 32))]
            z = torch.stack(z)
            z = z.to('cuda')
            generated = model.module.decode(z)
            utils.save_image(
                torch.cat([out,img],0),
                os.path.join(args.out_dir, 'out', str(idx)+'.png' ),
                normalize=True,
                range=(-1, 1)
            )
            utils.save_image(
                generated,
                os.path.join(args.out_dir, 'out', str(idx)+'_random.png' ),
                normalize=True,
                range=(-1, 1)
            )


def main(args):
    model = VAE(zsize=512).to(args.device)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    test_loader = get_test_loader(args.batch_size, args.dataset_path, args.device, transform=transform)

    with open(args.model_path, 'rb') as f:
        state_dict = torch.load(f, map_location='cpu')
        model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    test(test_loader, model, args.out_dir)


if __name__ == '__main__':

    import argparse
    import os

    parser = argparse.ArgumentParser(description='parser for vae test')

    parser.add_argument('--dataset_path', type=str, help='path to the dataset')
    parser.add_argument('--model_path', type=str, help='path to the model checkpoint', default=None)


    # miscellaneous
    parser.add_argument('--out_dir', type=str, help='output dir', default='./')
    parser.add_argument('--device', type=str, help='set the device', default='cpu')
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.out_dir, 'out')):
        os.makedirs(os.path.join(args.out_dir, 'out'))
    args.device = torch.device(args.device)

    main(args)
