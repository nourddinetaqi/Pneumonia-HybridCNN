import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Creating the Residual Block.
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=stride, 
            padding=1, 
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
    


# Creating the Hybrid CNN.
class HybridPneumoniaCNN(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.gradcam_target_layer = self.layer3[-1].conv2



    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x



def load_trained_model(weights_path:str, device:torch.device)-> HybridPneumoniaCNN:
    model = HybridPneumoniaCNN(num_classes=2).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def gradcam_for_images(model, image_tensor, device, class_names=None):

    model.eval()
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = model.gradcam_target_layer.register_forward_hook(fwd_hook)
    handle_bwd = model.gradcam_target_layer.register_backward_hook(bwd_hook)

    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)
    pred_idx = outputs.argmax(dim=1).item()
    score = outputs[0, pred_idx]

    model.zero_grad()
    score.backward(retain_graph = True)


    act = activations[0]          
    grad = gradients[0]           

    
    weights = grad.mean(dim=(2, 3), keepdim=True)    

    
    cam = (weights * act).sum(dim=1, keepdim=True)    

    cam = F.relu(cam)

    
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    
    cam = F.interpolate(
        cam,
        size=(image_tensor.shape[2], image_tensor.shape[3]),
        mode='bilinear',
        align_corners=False
    )

    cam = cam.squeeze().cpu().numpy()


    img_np = image_tensor[0].detach().cpu().numpy()
    img_np = np.transpose(img_np, (1,2,0))
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    handle_fwd.remove()
    handle_bwd.remove()

    return img_np, cam, pred_idx