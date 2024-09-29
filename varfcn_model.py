import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
import torch.nn.functional as F
import math
from torchvision.models.segmentation.fcn import FCNHead
import torch.nn.init as init


class PriorEncoder(nn.Module):
    def __init__(self):
        super(PriorEncoder, self).__init__()
        weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = fcn_resnet50(weights=weights)

        self.backbone = model.backbone

        self.mu = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # spatial size becomes 32x32
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),  # maintain 32x32 size
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # spatial size becomes 16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten for linear layer
            nn.Linear(256 * 16 * 16, 512, bias = False)  # Output size is 512
        )

        # Define the sequence for logvar
        self.logvar = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # spatial size becomes 32x32
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),  # maintain 32x32 size
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # spatial size becomes 16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten for linear layer
            nn.Linear(256 * 16 * 16, 512, bias = False)  # Output size is 512
        )
        for module in self.logvar:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  
        
        for module in self.mu:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  


    def forward(self, x):
        out = self.backbone(x)['out']

        mu_ = self.mu(out)
        logvar_ = self.logvar(out)
        z = Independent(Normal(loc=mu_, scale=torch.exp(logvar_)), 1)

        return out, z, mu_, logvar_


class PosteriorEncoder(nn.Module):
    def __init__(self):
        super(PosteriorEncoder, self).__init__()
        weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = fcn_resnet50(weights=weights)
        # model = fcn_resnet50()
        
        self.backbone = model.backbone
        weight_clone = self.backbone.conv1.weight.clone()
        self.backbone.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight[:,:3,:,:] = weight_clone
            init.xavier_uniform_(self.backbone.conv1.weight[:, 3:, :, :])

        self.mu = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # spatial size becomes 32x32
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),  # maintain 32x32 size
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # spatial size becomes 16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten for linear layer
            nn.Linear(256 * 16 * 16, 512, bias = False)  # Output size is 512
        )

        # Define the sequence for logvar
        self.logvar = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # spatial size becomes 32x32
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),  # maintain 32x32 size
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # spatial size becomes 16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten for linear layer
            nn.Linear(256 * 16 * 16, 512, bias = False)  # Output size is 512
        )
        for module in self.logvar:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  
        
        for module in self.mu:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  


    def forward(self, x):
        out = self.backbone(x)['out']
        mu_ = self.mu(out)
        logvar_ = self.logvar(out)
        z = Independent(Normal(loc=mu_, scale=torch.exp(logvar_) ), 1)
        return out, z, mu_, logvar_


class FCNDecoder(nn.Module):
    def __init__(self, num_classes):
        super(FCNDecoder, self).__init__()
        model = fcn_resnet50()
        model.classifier[-1] =  nn.Conv2d(model.classifier[-1].weight.shape[1], num_classes, kernel_size=1, stride=1, bias = False).cuda()
        
        self.decode = model.classifier

    def forward(self, x):
        H, W = x.shape[-2:]
        x = F.interpolate(x, size=(2*H,2*W), mode='bilinear', align_corners=False)
        x = self.decode(x)
        x = F.interpolate(x, size=(520,520), mode='bilinear', align_corners=False)
        return x


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
            )
            * scale
        )

    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale

   
class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
       

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

 
    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)


            out = F.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(
            input, weight, padding=self.padding, groups=batch
        )
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out

class CVAE_SE_FCN_Modulation(nn.Module):
    def __init__(self, num_classes=5):
        super(CVAE_SE_FCN_Modulation, self).__init__()
        
        self.priorNet = PriorEncoder()
        self.postNet = PosteriorEncoder()
        self.decoder = FCNDecoder(num_classes)
        self.modulate = ModulatedConv2d(2048,2048,3,512)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def resampling(self, mu, logvar, device):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_().to(device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_mask= None, training = False, num_ensembles= 8, device= 'cuda'):

        prior_out, prior_z, prior_mu_, prior_logvar_ = self.priorNet(x)
        if training:
            post_out, post_z, post_mu_, post_logvar_ = self.postNet(x_mask)
            kl_loss = torch.mean(self.kl_divergence(post_z, prior_z))
        multi_out_prior = []
        for i in range(num_ensembles):
            prior_resample = self.resampling(prior_mu_, prior_logvar_, device)
            prior_to_decoder = self.modulate(prior_out, prior_resample) 
            multi_out_prior.append(self.decoder(prior_to_decoder))
          

        decoded_out_prior = torch.stack(multi_out_prior)
     
        if training:
            return decoded_out_prior, kl_loss
        else:
            return decoded_out_prior


if __name__ == "__main__":
    import ipdb; ipdb.set_trace()
    FCNNet = CVAE_SE_FCN_Modulation(num_classes = 5).cuda()
    img = torch.randn(1,3,512,512).cuda()
    out_image = FCNNet(img, training = True,  num_ensembles= 3)
    import ipdb; ipdb.set_trace()


