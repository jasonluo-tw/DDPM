import torch
import torch.nn as nn
from .UNet_torch import UNet, CropNConcat

def sinusoidal_embedding(n, d):
    """
    n: iteration steps,
    d: time embedding dimension
    """
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding

def _make_te(dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

class UNet_DDPM(UNet):
    def __init__(self, input_shape, root_filter=64, layer_depth=5, drop_rate=0.2, n_steps=1000, time_emb_dim=100, training=True):
        super().__init__(input_shape, root_filter, layer_depth, drop_rate, training)

        ## sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)


        self.time_list = nn.ModuleList()
        for i in range(layer_depth-1):
            self.time_list.append(_make_te(time_emb_dim, self.filter_list[i]))

        self.time_list2 = nn.ModuleList()
        for i in range(layer_depth, 1, -1):
            self.time_list2.append(_make_te(time_emb_dim, self.filter_list[i]))

    def forward(self, x, t):
        skip_connects = []
        t = self.time_embed(t)
        nn = len(x)

        for dd, te in zip(self.downlist, self.time_list):
            ## add time
            x = x + te(t).reshape(nn, -1, 1, 1)
            x = dd(x)
            skip_connects.append(x)
            x = self.pooling2d(x)

        for index in range(len(self.uplist)):
            x = self.uplist[index](x)
            x = CropNConcat(x, skip_connects[len(skip_connects)-1-index])
            x = x + self.time_list2[index](t).reshape(nn, -1, 1, 1)
            x = self.downlist2[index](x)

        out = self.out_conv(x)

        return out

class DDPM(nn.Module):
    def __init__(self, backbone, input_shape, n_steps=1000, min_beta=10**(-4), max_beta=0.02):
        super(DDPM, self).__init__()

        self.backbone = backbone
        self.n_steps = n_steps
        self.input_shape = input_shape
        # Number of steps is typically in the order of thousands
        self.betas = torch.linspace(min_beta, max_beta, n_steps)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])

    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w)

        a_bar = a_bar.to(x0.device)
        
        noisy_x = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta

        return noisy_x

    def backward(self, x, t):

        x = x.to('cuda')
        t = t.to(x.device)

        return self.backbone(x, t)


if __name__ == '__main__':
    from torchinfo import summary
    from torch.utils.tensorboard import SummaryWriter
    import torch

    input_shape = (1, 28, 28)
    n_steps = 1000
    backbone = UNet_DDPM(input_shape=input_shape, n_steps=n_steps, layer_depth=4, drop_rate=0.2)
    ddpm_model = DDPM(backbone, input_shape, n_steps=n_steps)

    percent = 1
    t = [int(percent * (200-1)) for _ in range(16)]

    ##summary(model, input_size=(16, 4, 256, 256))
    #t = torch.randint(0, 1000, (16,))
    fake_input = torch.rand((16, 1, 28, 28))
    values = ddpm_model(fake_input, t)
    #values = backbone(fake_input, t=t)
    print('input:', fake_input.shape)
    print('output:', values.shape)
    #writer = SummaryWriter()
    #writer.add_graph(model, fake_input)
    #writer.close()
