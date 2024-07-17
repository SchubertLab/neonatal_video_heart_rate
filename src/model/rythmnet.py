import torch
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class RhythmNet(nn.Module):
    def __init__(self, pretrained=True, ir_channel=False, num_input_channel=4):
        super(RhythmNet, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        resnet = models.resnet18(weights=weights)
        if ir_channel:
            resnet.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        modules = list(resnet.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        self.resnet_linear = nn.Linear(512, 1000)

        # FC Layer
        self.fc_regression = nn.Linear(1000, 1)

        # GRU Layer
        self.rnn = nn.GRU(input_size=1000, hidden_size=1000, num_layers=1)

        # Output after GRU
        self.gru_fc_out = nn.Linear(1000, 1)

    def forward(self, st_maps, frame_rate=30, conversion_to_bpm=0.2):
        batched_output_per_clip = []
        gru_input_per_clip = []
        hr_per_clip = []

        for t in range(st_maps.size(0)):
            temp_st_maps = st_maps[t, :, :, :]
            temp_st_maps = torch.reshape(
                    temp_st_maps, (1, st_maps.shape[3], st_maps.shape[2], st_maps.shape[1])
            )
            x = self.resnet18(temp_st_maps)
            x = x.view(x.size(0), -1)
            x = self.resnet_linear(x)
            gru_input_per_clip.append(x.squeeze(0))

            # Final regression layer for CNN features -> HR (per clip)
            x = self.fc_regression(x)
            x = x * frame_rate * conversion_to_bpm
            batched_output_per_clip.append(x.squeeze(0))

        # the features extracted from the backbone CNN are fed to a one-layer GRU structure.
        regression_output = torch.stack(batched_output_per_clip, dim=0).permute(1, 0)

        # Trying out GRU in addition to the regression now.
        gru_input = torch.stack(gru_input_per_clip, dim=0)
        gru_output, h_n = self.rnn(gru_input.unsqueeze(1))

        for i in range(gru_output.size(0)):
            hr = self.gru_fc_out(gru_output[i, :, :])
            # option added by me ->
            hr = hr * frame_rate * conversion_to_bpm
            hr_per_clip.append(hr.flatten())

        gru_output_seq = torch.stack(hr_per_clip, dim=0).permute(1, 0)
        gru_output_seq = gru_output_seq.squeeze(0)[:gru_output.size(0)]

        return regression_output, gru_output_seq

    def name(self):
        return "RhythmNet"


if __name__ == '__main__':
    TEST_MODEL = True
    if TEST_MODEL:
        model = RhythmNet()
        print(model)
