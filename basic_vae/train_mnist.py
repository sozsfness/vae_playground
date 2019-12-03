import numpy as np
import torch
from torchvision import transforms, datasets
from torch import nn, optim
from datetime import datetime
from torch.utils import data

from vae import *
from data_loader import *
from tensorboardX import SummaryWriter
from torchvision import utils


def train(data_loader, model, optimizer, islabel=False):
    model.train()
    mse_sum = 0.
    mse_n = 0.
    for _, (img, label) in enumerate(data_loader):
        out, mu, logvar, out_label = model(img.cuda())
        if islabel:
            recon, kl, c_err = loss_function(out, img, mu, logvar, out_label, label)
        else:
            recon, kl, c_err = loss_function(out, img, mu, logvar, None, None)
        loss = recon + kl + c_err

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse_sum += (recon.item()+kl.item()+c_err.item())
        mse_n += 1

    return mse_sum/mse_n


def val(data_loader, model, is_semi=False):
    model.eval()
    mse_sum = 0.
    mse_n = 0.

    with torch.no_grad():
        for _, (img, label) in enumerate(data_loader):
            out, mu, logvar, out_label = model(img)
            if is_semi:
                recon, kl, c_err = loss_function(out, img, mu, logvar, out_label, label)
            else:
                recon, kl, c_err = loss_function(out, img, mu, logvar, out_label, None)                
            if is_semi:
                mse_sum += (recon.item() + kl.item() + c_err.item())
            else:
                mse_sum += (recon.item() + kl.item())
            mse_n += 1
            # print('loss: {}'.format(mse_sum))

    return mse_sum/mse_n


def main(args):
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    writer = SummaryWriter(os.path.join(args.out_dir, 'logs', args.model_name+'_'+current_time))
    model = VAE(zsize=256, channels=1, is_semi=args.is_semi).to(args.device)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform)

    test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    )

    if args.is_semi:
        train_supervisedIDX = list(range(len(train_dataset)))[:50000]
        train_unsupervisedIDX = list(range(len(train_dataset)))[50000:]

        train_labelled = data.Subset(train_dataset, train_supervisedIDX)
        train_unlabelled = data.Subset(train_dataset, train_unsupervisedIDX)

        train_loader_labelled = DataLoader(train_labelled, batch_size=args.batch_size, shuffle=True)
        train_loader_unlabelled = DataLoader(train_unlabelled, batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    val_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    if args.model_path is not None:
        with open(args.model_path, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        if args.is_semi:
            loss_train_labelled = train(train_loader_labelled, model, optimizer, True)
            loss_train_unlabelled = train(train_loader_unlabelled, model, optimizer, False)

            print('epoch {}, loss: {}'.format(epoch, loss_train_labelled))
            writer.add_scalar('loss_train', loss_train_labelled, epoch)
            print('epoch {}, loss_unlabelled: {}'.format(epoch, loss_train_unlabelled))
            writer.add_scalar('loss_train_unlabelled', loss_train_unlabelled, epoch)
        else:
            loss_train = train(train_loader, model, optimizer, False)
            print('epoch {}, loss: {}'.format(epoch, loss_train))
            writer.add_scalar('loss_train', loss_train, epoch)

        if epoch % 5 == 0:
            loss_val = val(val_loader, model, args.is_semi)
            print('val epoch: {}, loss: {}'.format(epoch, loss_val))
            writer.add_scalar('loss_val', loss_val, epoch)

        if epoch % args.save_per_epochs == 0 and epoch != 0:
            os.makedirs(os.path.join(args.out_dir, 'models', args.model_name+'_'+current_time, str(epoch)))
            torch.save(model.module.state_dict(),
                       os.path.join(args.out_dir, 'models', args.model_name+'_'+current_time, str(epoch), 'vae.pt'))


if __name__ == '__main__':

    import argparse
    import os

    parser = argparse.ArgumentParser(description='vae')

    parser.add_argument('--model_path', type=str, help='path to the model checkpoint', default=None)

    # embeddings
    parser.add_argument('--n_embedding', type=int, help='number of latent vectors')
    parser.add_argument('--size_prior', type=int, help='size of prior')
    parser.add_argument('--dim_embedding', type=int, help='dimension of embeddings')
    parser.add_argument('--n_prior_layers', type=int, help='number of layers for prior')

    # optimization
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_per_epochs', type=int, default=50)
    parser.add_argument('--is_semi', type=int, help='enable semi-supervised or not', default=0)

    # miscellaneous
    parser.add_argument('--out_dir', type=str, help='output dir', default='./')
    parser.add_argument('--device', type=str, help='set the device', default='cuda')
    parser.add_argument('--model_name', type=str, help='name of model', default='vanillaVAE')

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.out_dir, 'logs')):
        os.makedirs(os.path.join(args.out_dir, 'logs'))
    if not os.path.exists(os.path.join(args.out_dir, 'models')):
        os.makedirs(os.path.join(args.out_dir, 'models'))

    args.device = torch.device(args.device)

    main(args)
