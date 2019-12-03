import torch
from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, dims):
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x


class VAE(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3, is_semi=False):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        if is_semi:
            self.fc3 = nn.Linear(30, 400)
        else:
            self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.is_semi = is_semi
        if is_semi:
            self.classifier = Classifier([784, 256, 10])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        out_label = None
        if self.is_semi:
            out_label = self.classifier(x.view(-1,784))
        z = self.reparameterize(mu, logvar)
        if self.is_semi:
            z = torch.cat([z, out_label], dim=1)
        return self.decode(z), mu, logvar, out_label

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def loss_function(recon_x, x, mu, logvar, out_label, label):
    recon_criterion = nn.MSELoss()
    recon_loss = recon_criterion(recon_x, x.view(-1,784).cuda())

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    c_err = torch.tensor(0)
    criterion = nn.CrossEntropyLoss()

    if label is not None:
        c_err = torch.mean(criterion(out_label, label.cuda()))

    return recon_loss, KLD, c_err


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
