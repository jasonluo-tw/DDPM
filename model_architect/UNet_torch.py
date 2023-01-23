import torch
import torch.nn as nn
import torch.nn.functional as F

class DownConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ksize, drop_rate=0.2):
        super().__init__()

        self.activation = nn.ReLU()
        self.out_c = out_c
        ## 
        self.conv2d_1 = nn.Conv2d(in_c, out_c, kernel_size=ksize, padding=1)
        self.drop_1 = nn.Dropout(drop_rate)

        self.conv2d_2 = nn.Conv2d(out_c, out_c, kernel_size=ksize, padding=1)
        self.drop_2 = nn.Dropout(drop_rate)

        ## conv1x1
        self.conv2d_1x1 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x: torch.Tensor, training=True):
        if x.shape[1] != self.out_c:
            res = self.conv2d_1x1(x)
        else:
            res = x.clone()

        x = self.conv2d_1(x)
        if training:
            x = self.drop_1(x)
        x = self.activation(x)

        x = self.conv2d_2(x)
        if training:
            x = self.drop_2(x)
        x = self.activation(x)

        return x + res

class UpConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ksize, stride=2, drop_rate=0.2):
        super().__init__()

        self.activation = nn.ReLU()
        
        self.upconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=ksize, stride=stride)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x, training=True):
        x = self.upconv(x)
        if training:
            x = self.drop(x)
        x = self.activation(x)

        return x

def CropNConcat(x1, x2):
    row_diff = x2.shape[3] - x1.shape[3]
    col_diff = x2.shape[2] - x1.shape[2]

    x1 = F.pad(x1, [row_diff // 2, row_diff - row_diff // 2,
                     col_diff // 2, col_diff - col_diff // 2])

    out = torch.cat([x1, x2], dim=1)

    return out

class UNet(nn.Module):
    def __init__(self, input_shape, root_filter=64, layer_depth=5, drop_rate=0.2, training=True):
        super().__init__()

        self.input_shape = input_shape
        self.filter_list = [input_shape[0]] + [root_filter*(2**i) for i in range(layer_depth)]
        self.pooling2d = nn.MaxPool2d(2)

        self.downlist = nn.ModuleList()

        for i in range(layer_depth-1):
            self.downlist.append(
                DownConvBlock(self.filter_list[i], self.filter_list[i+1], 
                              3, drop_rate=drop_rate)
            )

        self.uplist = nn.ModuleList()
        self.downlist2 = nn.ModuleList()

        for i in range(layer_depth, 1, -1):
            if i == layer_depth:
                up_inc = self.filter_list[i-1]
            else:
                up_inc = self.filter_list[i]

            self.uplist.append(
                UpConvBlock(up_inc, self.filter_list[i-1], 
                            3, 2, drop_rate=drop_rate)
            )
            self.downlist2.append(
                DownConvBlock(self.filter_list[i], self.filter_list[i-1], 
                              3, drop_rate=drop_rate)
            )

        self.out_conv = nn.Conv2d(self.filter_list[1], 1, kernel_size=1)

    def forward(self, x):
        skip_connects = []

        for dd in self.downlist:
            x = dd(x)
            skip_connects.append(x)
            x = self.pooling2d(x)

        for index in range(len(self.uplist)):
            x = self.uplist[index](x)
            x = CropNConcat(x, skip_connects[len(skip_connects)-1-index])
            x = self.downlist2[index](x)

        out = self.out_conv(x)

        return out

if __name__ == '__main__':
    from torchinfo import summary
    from torch.utils.tensorboard import SummaryWriter
    import torch
    model = UNet(input_shape=(1, 28, 28), layer_depth=4, drop_rate=0.2)
    ##summary(model, input_size=(16, 4, 256, 256))
    fake_input = torch.rand((16, 1, 28, 28))
    values = model(fake_input)
    print('input:', fake_input.shape)
    print('output:', values.shape)
    #writer = SummaryWriter()
    #writer.add_graph(model, fake_input)
    #writer.close()
