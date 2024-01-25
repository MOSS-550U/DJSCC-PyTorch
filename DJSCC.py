import torch
import torch.nn as nn


class Deep_JSCC(nn.Module):
    def __init__(self, filter_k):
        super(Deep_JSCC, self).__init__()

        self.En_Conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.En_Conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.En_Conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.En_Conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.En_Conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=filter_k, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(filter_k),
            nn.ReLU()
        )

        self.De_TransConv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filter_k, out_channels=128, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.De_TransConv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.De_TransConv_layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.De_TransConv_layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.De_TransConv_layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x, SNR):
        x_En1 = self.En_Conv_layer1(x)
        x_En2 = self.En_Conv_layer2(x_En1)
        x_En3 = self.En_Conv_layer3(x_En2)
        x_En4 = self.En_Conv_layer4(x_En3)
        x_En5 = self.En_Conv_layer5(x_En4)
        x_2_Signal = torch.reshape(x_En5, (2, -1))
        Complex_signal = torch.complex(x_2_Signal[0], x_2_Signal[1])
        Channel_normalization = channel_normalization(Complex_signal, 1)
        AWGN = add_complex_gaussian_noise(Channel_normalization, SNR)
        Channel_output = torch.cat([AWGN.real, AWGN.imag], 0)
        Channel_output = Channel_output.float()
        Channel_output = torch.reshape(Channel_output, [x_En5.size(0), x_En5.size(1), x_En5.size(2), x_En5.size(3)])
        x_De1 = self.De_TransConv_layer1(Channel_output)
        x_De2 = self.De_TransConv_layer2(x_De1)
        x_De3 = self.De_TransConv_layer3(x_De2)
        x_De4 = self.De_TransConv_layer4(x_De3)
        Output = self.De_TransConv_layer5(x_De4)

        return Output