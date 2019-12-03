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
from vae import *

def test(data_loader, model, out_dir, is_semi):
    model.eval()
    with torch.no_grad():
        if data_loader is not None:
            for idx, img in enumerate(data_loader):
                out, mu, logvar = model(img)
                z = [torch.randn((512, 32, 32))]
                z = torch.stack(z)
                z = z.to('cuda')
                utils.save_image(
                    torch.cat([out,img],0),
                    os.path.join(args.out_dir, 'out', str(idx)+'.png' ),
                    normalize=True,
                    range=(-1, 1)
                )
        else: #10 samples for each digit.
            samples = []
            digits = []
            for i in range(10):
                latent_features = torch.randn(10, 20).cuda()
                if is_semi:
                    label = np.zeros((10,10)).astype(float)
                    label[:,i] = 1
                    label = torch.FloatTensor(label).cuda()
                    out = model.module.decode(torch.cat([latent_features, label], dim=1))
                else:
                    out = model.module.decode(latent_features)
                samples.append(out.view(-1, 28, 28))
            
            for idx, sample in enumerate(samples):
                sample = list(torch.split(sample, 1, dim=0))
                digits.append(torch.cat(sample, dim=1))
                utils.save_image(
                        torch.cat(sample, dim=1),
                        os.path.join(args.out_dir, str(idx) + 'out.png' ),
                        normalize=True,
                        range=(-1, 1)
                    )


def main(args):
    model = VAE(is_semi=args.is_semi).to(args.device)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    if args.dataset_path is not None:
        test_loader = get_test_loader(args.batch_size, args.dataset_path, args.device, transform=transform)

    with open(args.model_path, 'rb') as f:
        if args.device == torch.device('cpu'):
            state_dict = torch.load(f, map_location='cpu')
        else:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()

    if args.dataset_path is not None:#reconstruction
        test_loader = get_test_loader(args.batch_size, args.dataset_path, args.device, transform=transform)
        test(test_loader, model, args.out_dir)
    else:
        test(None, model, args.out_dir, args.is_semi)#generation


if __name__ == '__main__':

    import argparse
    import os
    parser = argparse.ArgumentParser(description='parset for vae')


    parser.add_argument('--dataset_path', type=str, help='path to the dataset', default=None)
    parser.add_argument('--model_path', type=str, help='path to the model checkpoint', default=None)
    parser.add_argument('--is_semi', type=int, help='enable semi-supervised or not', default=1)


    # miscellaneous
    parser.add_argument('--out_dir', type=str, help='output dir', default='./')
    parser.add_argument('--device', type=str, help='set the device', default='cuda')
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.out_dir, 'out')):
        os.makedirs(os.path.join(args.out_dir, 'out'))
    args.out_dir = os.path.join(args.out_dir, 'out')
    args.device = torch.device(args.device)

    main(args)
