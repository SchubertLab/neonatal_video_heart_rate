import torch
from torch import nn
import src.config as config


class MyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hr_t, hr_outs, T):
        ctx.hr_outs = hr_outs
        ctx.hr_mean = hr_outs.mean()
        ctx.T = T
        ctx.save_for_backward(hr_t)
        if hr_t > ctx.hr_mean:
            loss = hr_t - ctx.hr_mean
        else:
            loss = ctx.hr_mean - hr_t
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        output = torch.zeros(1).to(config.params['DEVICE'])
        hr_t, = ctx.saved_tensors
        hr_outs = ctx.hr_outs

        # create a list of hr_outs without hr_t
        for hr in hr_outs:
            if hr == hr_t:
                pass
            else:
                output = output + (1/ctx.T)*torch.sign(ctx.hr_mean - hr)

        output = (1/ctx.T - 1)*torch.sign(ctx.hr_mean - hr_t) + output

        return output, None, None


class RhythmNetLoss(nn.Module):
    def __init__(self, weight_reg=1, weight_gru=10, gru_output=True):
        super(RhythmNetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.weight_gru = weight_gru
        self.weight_reg = weight_reg
        self.gru_outputs_considered = None
        self.custom_loss = MyLoss()
        self.device = config.params['DEVICE']
        self.gru_output = gru_output

    def forward(self, resnet_outputs, gru_outputs, target):
        if self.gru_output:
            # MSE loss with GRU output
            reg_loss = self.mse_loss(gru_outputs, target)
            # Smooth Loss
            smooth_loss_component = self.smooth_loss(gru_outputs)
            loss = self.weight_reg * reg_loss + self.weight_gru * smooth_loss_component
        else:
            # loss with reg output only
            loss = self.mse_loss(resnet_outputs, target)
        return loss

    def smooth_loss(self, gru_outputs):
        smooth_loss = torch.zeros(1).to(device=self.device)
        self.gru_outputs_considered = gru_outputs.flatten()
        for hr_t in self.gru_outputs_considered:
            smooth_loss = smooth_loss + self.custom_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                               self.gru_outputs_considered,
                                                               self.gru_outputs_considered.shape[0])
        return smooth_loss / self.gru_outputs_considered.shape[0]
