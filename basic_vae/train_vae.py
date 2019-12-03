import numpy as np
import torch
from torchvision import transforms
from torch import nn, optim
from datetime import datetime
from vae_cat import *
from data_loader import *
from tensorboardX import SummaryWriter
from torchvision import utils


def train(data_loader, model, optimizer, is_semi=False):
    model.train()

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25

    mse_sum = 0.
    mse_n = 0.
    mean_latent_loss = 0.
    for _, img in enumerate(data_loader):
        out, mu, logvar = model(img)
        recon, kl = loss_function(out, img, mu, logvar)
        loss = recon + kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse_sum += (recon.item()+kl.item()) * img.shape[0]
        mse_n += img.shape[0]

    return mse_sum/mse_n


def val(data_loader, model, is_semi=False):
    model.eval()
    criterion = nn.MSELoss()

    mse_sum = 0.
    mse_n = 0.
    mean_latent_loss = 0.

    with torch.no_grad():
        for _, img in enumerate(data_loader):
            out, mu, logvar = model(img)
            recon, kl = loss_function(out, img, mu, logvar)
            mse_sum += (recon.item()+kl.item()) * img.shape[0]
            mse_n += img.shape[0]
    return mse_sum/mse_n


def main(args):
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    writer = SummaryWriter(os.path.join(args.out_dir, 'logs'))
    model = VAE(zsize=512).to(args.device)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    train_loader, val_loader = get_train_loader(args.batch_size, args.dataset_path,
                                                args.device, transform=transform, is_semi=args.is_semi)

    if args.model_path is not None:
        with open(args.model_path, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        loss_train = train(train_loader, model, optimizer, args.is_semi)
        print('epoch {}, loss: {}'.format(epoch, loss_train))
        writer.add_scalar('loss_train', loss_train, epoch)
        if epoch % 50 == 0:
            loss_val = val(val_loader, model, args.is_semi)
            print('val epoch: {}, loss: {}'.format(epoch, loss_val))
            writer.add_scalar('loss_val', loss_val, epoch)

        if epoch % args.save_per_epochs == 0 and epoch != 0:
            torch.save(model.module.state_dict(),
                       os.path.join(args.out_dir, 'models' , current_time +
                                    '_epoch'+str(epoch) + '.pt'))
    torch.save(model.module.state_dict(),
               os.path.join(args.out_dir, 'models', current_time + '_epoch'+str(args.num_epochs)+'.pt'))


if __name__ == '__main__':

    import argparse
    import os

    parser = argparse.ArgumentParser(description='vae')

    parser.add_argument('--dataset_path', type=str, help='path to the dataset', default=None)
    parser.add_argument('--model_path', type=str, help='path to the model checkpoint', default=None)
    parser.add_argument('--num_samples', type=int, help='number of samples')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)

    # embeddings
    parser.add_argument('--n_embedding', type=int, help='number of latent vectors')
    parser.add_argument('--size_prior', type=int, help='size of prior')
    parser.add_argument('--dim_embedding', type=int, help='dimension of embeddings')
    parser.add_argument('--n_prior_layers', type=int, help='number of layers for prior')

    # optimization
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_per_epochs', type=int, default=500)
    parser.add_argument('--is_semi', type=int, help='enable semi-supervised or not', default=False)

    # miscellaneous
    parser.add_argument('--out_dir', type=str, help='output dir', default='./')
    parser.add_argument('--device', type=str, help='set the device', default='cuda')

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.out_dir, 'logs')):
        os.makedirs(os.path.join(args.out_dir, 'logs'))
    if not os.path.exists(os.path.join(args.out_dir, 'models')):
        os.makedirs(os.path.join(args.out_dir, 'models'))

    args.device = torch.device(args.device)

    main(args)
