# model.py

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIZE = 256  # Default resizing size

# ----------------- Dataset -----------------
class ColorizationAugmentor(Dataset):
    def __init__(self, paths, split='train', num_augmentations=1):
        self.paths = paths
        self.split = split
        self.num_augmentations = num_augmentations

    def get_transforms(self, random=True):
        transform_list = [transforms.Resize((SIZE, SIZE), Image.BICUBIC)]
        if self.split == 'train' and random:
            transform_list.append(transforms.RandomRotation(degrees=(0, 180)))
        return transforms.Compose(transform_list)

    def load_original_image(self, path):
        return Image.open(path).convert("RGB")

    def transform_and_convert_to_lab(self, img, transform):
        img_transformed = transform(img)
        img_np = np.array(img_transformed)
        img_lab = rgb2lab(img_np).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        return L, ab

    def __len__(self):
        return len(self.paths) * self.num_augmentations

    def __getitem__(self, idx):
        image_idx = idx // self.num_augmentations
        img = self.load_original_image(self.paths[image_idx])
        transform = self.get_transforms(random=True)
        L, ab = self.transform_and_convert_to_lab(img, transform)
        return {'L': L, 'ab': ab}

def make_dataloaders(paths, split='train', batch_size=16, num_augmentations=1, num_workers=4):
    dataset = ColorizationAugmentor(paths, split=split, num_augmentations=num_augmentations)
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

# ----------------- Model Utilities -----------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    return {name: AverageMeter() for name in ['loss_D_fake', 'loss_D_real', 'loss_D', 'loss_G_GAN', 'loss_G_L1', 'loss_G']}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    return np.clip(np.stack([lab2rgb(img) for img in Lab]), 0, 1)

# ----------------- Model Components -----------------
def build_res_unet(n_input=1, n_output=2, size=256):
    model = resnet18(pretrained=True)
    body = create_body(model, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
    net.apply(init_func)
    print(f"Model initialized with {init} initialization")
    return net

def init_model(model, device):
    return init_weights(model.to(device))

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        return self.real_label.expand_as(preds) if target_is_real else self.fake_label.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        return self.loss(preds, labels)

class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2**i, num_filters * 2**(i+1), s=1 if i==(n_down-1) else 2) for i in range(n_down)]
        model += [self.get_layers(num_filters * 2**n_down, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers.append(nn.BatchNorm2d(nf))
        if act: layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda_L1 = lambda_L1
        self.net_G = net_G.to(self.device) if net_G else init_model(build_res_unet(), self.device)
        self.net_D = init_model(PatchDiscriminator(3), self.device)
        self.GANcriterion = GANLoss('vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)

        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

# ----------------- Colorization Inference -----------------
def colorize_image(pil_img, model_path="final_model_weights.pt", size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_G = build_res_unet(n_input=1, n_output=2, size=size)

    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('net_G.', ''): v for k, v in checkpoint.items() if k.startswith('net_G.')}
    net_G.load_state_dict(new_state_dict)
    net_G.to(device)
    net_G.eval()

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    img = pil_img.convert("RGB")
    img = transform(img)

    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_lab = rgb2lab(img_np).astype("float32")

    L = img_lab[:, :, 0]
    L = (L / 50.) - 1.0
    L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        ab_pred = net_G(L)

    L = (L + 1.) * 50.
    ab_pred = ab_pred * 110.

    Lab = torch.cat([L, ab_pred], dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb_result = lab2rgb(Lab)
    return np.clip(rgb_result, 0, 1)
