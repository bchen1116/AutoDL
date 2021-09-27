import torch.optim as optim
import torch.nn as nn

def bce_loss():
    return nn.BCELoss()


def crossentropy_loss():
    return nn.CrossEntropyLoss()


def bce_logit_loss():
    return nn.BCEWithLogitsLoss()


def sgd_optim(model_params, lr, momentum=0):
    return optim.SGD(model_params, lr=lr, momentum=momentum)


def adam_optim(model_params, lr):
    return optim.Adam(model_params, lr=lr)


def step_LR(optimizer, step_size=50, gamma=0.1):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
