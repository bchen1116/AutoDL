from abc import abstractmethod
import torchvision.models as mod
import copy


class ModelBase():
    # this is the base class
    @property
    @abstractmethod
    def name(cls):
        "The name of the model"

    @property
    def is_fitted(cls):
        "Whether the model is fitted"
        return False

    @abstractmethod
    def best_state_dict(self):
        "Return the best state dict of the model"

    def get_model(self):
        return self.model

    def update_best_state_dict(self):
        self.best_state_dict = copy.deepcopy(self.model.state_dict())

    def get_model_best_state_dict(self):
        return self.model.load_state_dict(self.best_state_dict)


class ResNet18(ModelBase):
    name = "ResNet-18"
    def __init__(self, pretrained=True):
        self.model = mod.resnet18(pretrained=pretrained)


class VGG16(ModelBase):
    name = "VGG-16"
    def __init__(self, pretrained=True):
        self.model = mod.vgg16(pretrained=pretrained)


class DenseNet(ModelBase):
    name = "DenseNet"
    def __init__(self, pretrained=True):
        self.model = mod.densenet161(pretrained=pretrained)


class Inception(ModelBase):
    name = "Inception"
    def __init__(self, pretrained=True):
        self.model = mod.inception_v3(pretrained=pretrained)


class GoogleNet(ModelBase):
    name = "Google Net"
    def __init__(self, pretrained=True):
        self.model = mod.googlenet(pretrained=pretrained)
