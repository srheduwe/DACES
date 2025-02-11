import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from torch import nn
import detectors
import os

def load_imagenet(dim=224):
    transform = transforms.Compose([transforms.Resize([dim, dim]), transforms.ToTensor()])
    trainset = torchvision.datasets.ImageNet(root=os.getcwd() + "/data/raw", split='train', transform=transform)    
    testset = torchvision.datasets.ImageNet(root=os.getcwd() + "/data/raw", split='val', transform=transform)
    return trainset, testset

def load_cifar100(dim=32):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root=os.getcwd() + "/data/raw", train=True, transform=transform, download=True)    
    testset = torchvision.datasets.CIFAR100(root=os.getcwd() + "/data/raw", train=False, transform=transform, download=True)
    return trainset, testset

def loader(model, split: str ="test"):
    '''Returns the model, the respective dataset and size of the dimension and channel.'''
    if model == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if torch.cuda.is_available():
            net = net.cuda()

        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()

        dim = 224
        trainset, testset = load_imagenet(dim=dim)

    elif model == "resnet101":
        net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        if torch.cuda.is_available():
            net = net.cuda()

        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()

        dim = 224
        trainset, testset = load_imagenet(dim=dim)

    elif model == "vgg16":
        net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        if torch.cuda.is_available():
            net = net.cuda()

        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()

        dim = 224
        trainset, testset = load_imagenet(dim=dim)

    elif model == "ViT":
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
        if torch.cuda.is_available():
            net = net.cuda()

        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()

        dim = 224
        trainset, testset = load_imagenet(dim=dim)

    elif model == "resnet34_cifar100":
        net = timm.create_model("resnet34_cifar100", pretrained=True)

        if torch.cuda.is_available():
            net = net.cuda()

        mean = torch.Tensor([0.5071, 0.4867, 0.4408])
        std = torch.Tensor([0.2675, 0.2565, 0.2761])
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()

        dim = 32
        trainset, testset = load_cifar100(dim=dim)

    elif model == "resnet50_cifar100":
        net = timm.create_model("resnet50_cifar100", pretrained=True)
        if torch.cuda.is_available():
            net = net.cuda()

        mean = torch.Tensor([0.5071, 0.4867, 0.4408])
        std = torch.Tensor([0.2675, 0.2565, 0.2761])
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()

        dim = 32
        trainset, testset = load_cifar100(dim=dim)

    elif model == "vgg16_cifar100":
        net = timm.create_model("vgg16_bn_cifar100", pretrained=True)
        if torch.cuda.is_available():
            net = net.cuda()

        mean = torch.Tensor([0.5071, 0.4867, 0.4408])
        std = torch.Tensor([0.2675, 0.2565, 0.2761])
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()

        dim = 32
        trainset, testset = load_cifar100(dim=dim)

    elif model == "ViT_cifar100":
        net = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k",
        pretrained=False)
        net.head = nn.Linear(net.head.in_features, 100)
        net.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
                map_location="cpu",
                file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
            )
        )

        if torch.cuda.is_available():
            net = net.cuda()

        mean = torch.Tensor([0.5071, 0.4867, 0.4408])
        std = torch.Tensor([0.2675, 0.2565, 0.2761])
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()

        dim = 224
        trainset, testset = load_cifar100(dim=dim)

    if split == "test":
        return net, testset, dim
    else:
        return net, trainset, dim