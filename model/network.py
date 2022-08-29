import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from .grl import GradientReverseFunction
from .models import register_model


class TaskNet(nn.Module):
    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    def __init__(self, num_cls=10, normalize=False, temp=0.05):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls
        self.setup_net()
        self.criterion = nn.CrossEntropyLoss()
        self.normalize = normalize
        self.temp = temp

    def forward(self, x, with_emb=False, reverse_grad=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        #x = x.clone()
        emb = self.fc_params(x)

        if isinstance(self.classifier, nn.Sequential):  # LeNet
            emb = self.classifier[:-1](emb)
            if reverse_grad: emb = GradientReverseFunction.apply(emb)
            if self.normalize: emb = F.normalize(emb) / self.temp
            score = self.classifier[-1](emb)
        else:  # ResNet
            if reverse_grad: emb = GradientReverseFunction.apply(emb)
            if self.normalize: emb = F.normalize(emb) / self.temp
            score = self.classifier(emb)

        if with_emb:
            return score, emb
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict, strict=False)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def parameters(self, lr, lr_scalar=0.1):
        parameter_list = [
            {'params': self.conv_params.parameters(), 'lr': lr * lr_scalar},
            {'params': self.fc_params.parameters(), 'lr': lr},
            {'params': self.classifier.parameters(), 'lr': lr},
        ]

        return parameter_list


@register_model('ResNet34Fc')
class ResNet34Fc(TaskNet):
    num_channels = 3
    name = 'ResNet34Fc'

    def setup_net(self):
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, self.num_cls, bias=False)


class BatchNorm1d(nn.Module):
    def __init__(self, dim):
        super(BatchNorm1d, self).__init__()
        self.BatchNorm1d = nn.BatchNorm1d(dim)

    def __call__(self, x):
        if x.size(0) == 1:
            x = torch.cat((x,x), 0)
            x = self.BatchNorm1d(x)[:1]
        else:
            x = self.BatchNorm1d(x)
        return x


@register_model('ResNet50Fc')
class ResNet50Fc(TaskNet):
    num_channels = 3
    name = 'ResNet50Fc'

    def setup_net(self):
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Sequential(nn.Linear(2048, 256), BatchNorm1d(256))
        self.classifier = nn.Linear(256, self.num_cls)


