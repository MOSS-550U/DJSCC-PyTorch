import torch
import torch.nn as nn

import WC


class Deep_JSCC(nn.Module):
    def __init__(self, filter_k):
        super(Deep_JSCC, self).__init__()

        self.En_Conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.PReLU())
        self.En_Conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.PReLU())
        self.En_Conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.PReLU())
        self.En_Conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.PReLU())
        self.En_Conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=filter_k, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.PReLU())

        self.De_TransConv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filter_k, out_channels=32, kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2)),
            nn.PReLU())
        self.De_TransConv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.PReLU())
        self.De_TransConv_layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.PReLU())
        self.De_TransConv_layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2),
                               output_padding=(1, 1)),
            nn.PReLU())
        self.De_TransConv_layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2),
                               output_padding=(1, 1)),
            nn.Sigmoid())

    def forward(self, Inpute_image, Channel, Power, SNR, **kwargs):
        x_En = self.En_Conv_layer1(Inpute_image)
        x_En = self.En_Conv_layer2(x_En)
        x_En = self.En_Conv_layer3(x_En)
        x_En = self.En_Conv_layer4(x_En)
        x_En = self.En_Conv_layer5(x_En)

        Channel_output = WC.wireless_channel(x_En, Channel, Power, SNR, **kwargs)

        x_De = self.De_TransConv_layer1(Channel_output)
        x_De = self.De_TransConv_layer2(x_De)
        x_De = self.De_TransConv_layer3(x_De)
        x_De = self.De_TransConv_layer4(x_De)
        Reconstructed_image = self.De_TransConv_layer5(x_De)

        return Reconstructed_image
