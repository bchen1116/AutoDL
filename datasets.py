import torchvision.datasets as ds
import torchvision.transforms as transforms

root = "./datasets/"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def fashionMNIST():
    train = ds.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test = ds.FashionMNIST(root=root, train=False, download=True, transform=transform)
    return (train, test)

def MNIST():
    train = ds.MNIST(root=root, train=True, download=True, transform=transform)
    test = ds.MNIST(root=root, train=False, download=True, transform=transform)
    return (train, test)

def CIFAR():
    train = ds.CIFAR10(root=root, train=True, download=True, transform=transform)
    test = ds.CIFAR10(root=root, train=False, download=True, transform=transform)
    return (train, test)
