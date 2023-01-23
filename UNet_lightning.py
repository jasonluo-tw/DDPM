import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from model_architect.DDPM import UNet_DDPM, DDPM
import pytorch_lightning as pl
## data
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

## others
import numpy as np
import matplotlib.pyplot as plt
import imageio

## add test moving mnist data
#sys.path.append('/home/luo-j/subs')
#from data_load import RadarDataSet
#from metrics import get_CSI_along_time

class UNet_model(pl.LightningModule):
    """
    UNet Model of Radar
    """
    def __init__(
            self,
            input_shape,
            n_steps = 1000,
            batch_size: int = 32,
            **kwargs
    ):
        super().__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.n_steps = n_steps
        #self.data_type = "MNIST"
        self.data_type = "FASION"

        backbone = UNet_DDPM(input_shape=input_shape,
                             n_steps=n_steps,
                             layer_depth=4,
                             drop_rate=0.2)

        self.ddpm_model = DDPM(backbone, input_shape, n_steps=n_steps)
        self.valid_plot_flag = True
        self.mse_loss = nn.MSELoss()
        
        ## Important: This property activates manual optimization.
        self.automatic_optimization = False
        #torch.autograd.set_detect_anomaly(True)

    ## X dims -> (batch, prev_step, channels, width, height)
    ## Y dims -> (batch, pred_step, channels, width, height)
    def prepare_data(self):
        transform = Compose([
            ToTensor(),
            Lambda(lambda x: (x - 0.5) * 2)
        ])
        if self.data_type == "MNIST":
            ds_fn = MNIST
            self.dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
        elif self.data_type == "FASION":
            ds_fn = FashionMNIST 
            self.dataset = ds_fn("./datasets", download=True, train=True, transform=transform)

    def setup(self, stage):
        ## Set up train & val dataset
        self.train_set = self.dataset
        #train_size = int(len(self.dataset) * 0.99)
        #valid_size = len(self.dataset) - train_size

        #self.train_set, self.valid_set = random_split(self.dataset, 
        #        [train_size, valid_size],
        #        generator=torch.Generator().manual_seed(42))
        
        print('Training set size:', len(self.train_set))
        #print('Validation set size:', len(self.valid_set))

    def train_dataloader(self):
        train_loader = DataLoader(
                 dataset=self.train_set,
                 batch_size=self.batch_size,
                 num_workers=2,
                 shuffle=True)

        return train_loader

    """
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
                 dataset=self.valid_set,
                 batch_size=self.batch_size,
                 num_workers=4,
                 shuffle=False)

        return val_loader
    """
    
    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        ## X,Y dims -> (batch, depth, channel, height, width)
        X0, Y = batch
        opt = self.optimizers()
        nn = X0.shape[0]

        ## sample noise
        eta = torch.randn_like(X0)
        tstep = torch.randint(0, self.n_steps, (nn,))
        #### model ####
        ## forward step
        X_noise = self.ddpm_model(X0, tstep, eta)

        ## backward
        ## getting estimated noise from model
        eta_estimate = self.ddpm_model.backward(
                                X_noise, 
                                tstep.reshape(nn, -1)
                        )

        mseLoss = self.mse_loss(eta_estimate, eta)
        opt.zero_grad()
        self.manual_backward(mseLoss)
        opt.step()

        self.log_dict(
            {
                'mse_loss': mseLoss,
            },
            prog_bar = True
        )

        if self.global_step % 100 == 0:
            ####
            print(f'epoch:{self.current_epoch},global_step:{self.global_step},mse_loss:{mseLoss}', flush=True)

        return {"loss": mseLoss}

    def training_epoch_end(self, training_step_outputs):
        all_loss = [i['loss'].detach().cpu().numpy() for i in training_step_outputs]
        all_mean_loss = np.mean(all_loss)
        print(f'EPOCH:{self.current_epoch} END, mean mse loss:{all_mean_loss:.4f}')
        gif_name = f"train_log/pics/v1_fasion/ep{self.current_epoch}_generated_imgs.gif"
        self.generate_imgs(gif_name=gif_name)

    ## Validation set
    #def validation_step(self, batch, batch_idx):
    #    X, Y = batch

    #def validation_epoch_end(self, all_loss):
    #    pass

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.ddpm_model.parameters())

        return opt

    def generate_imgs(self, samples=16, frames_per_gif=100, gif_name="test.gif"):
        interval = self.n_steps // frames_per_gif
        frames = []
        
        with torch.no_grad():
            x = torch.randn(samples, *self.input_shape) 

            for idx, t in enumerate(list(range(self.n_steps))[::-1]):
                time_tensor = (torch.ones(samples, 1) * t).long()
                eta_theta = self.ddpm_model.backward(x, time_tensor)
                eta_theta = eta_theta.detach().cpu()

                alpha_t = self.ddpm_model.alphas[t]
                alpha_t_bar = self.ddpm_model.alpha_bars[t]

                a = 1 / alpha_t.sqrt()
                b = ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * eta_theta

                x = a * (x - b)

                if t > 0:
                    z = torch.randn(samples, *self.input_shape)

                    beta_t = self.ddpm_model.betas[t]
                    sigma_t = beta_t.sqrt()

                    x = x + sigma_t * z

                ## Adding frames to the GIF
                if idx % interval == 0 or t == 0:
                    out = x.clone()
                    out -= torch.amin(out, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)
                    out *= 255 / torch.amax(out, dim=[1, 2, 3]).reshape(-1, 1, 1, 1)

                    frame = einops.rearrange(out, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(samples**0.5))
                    frame = frame.cpu().numpy().astype(np.uint8)

                    ## Rendering frame
                    frames.append(frame)

        ## store gif
        with imageio.get_writer(gif_name, mode="I") as writer:
            for idx, frame in enumerate(frames):
                writer.append_data(frame)
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(frames[-1])


if __name__ == '__main__':
    input_shape = (1, 28, 28)
    model = UNet_model(input_shape=input_shape)
    model.generate_imgs()
