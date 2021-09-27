import numpy as np
from torch.utils.data import DataLoader
import torch


def channels_to_first(input_matrix):
    # puts the channels at the end
    # pytorch requires all inputs to use (3 x H x W) for the input shapes
    transformed_matrix = input_matrix
    shape = input_matrix.shape
    if len(shape) == 2:
        transformed_matrix = np.stack((input_matrix, input_matrix, input_matrix), axis=0)
    elif len(shape) == 3:
        if shape[0] == 1:
            in_matrix = input_matrix.reshape(shape[1], shape[2])
            transformed_matrix = np.stack((in_matrix, in_matrix, in_matrix), axis=0)
    return transformed_matrix


def normalize(input_image):
    # normalize the input image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if np.max(input_image) > 127:
        # we need to divide by 255 to normalize the image
        input_image = np.divide(input_image, 255.)
    # now, we can normalize all the images
    for i in range(3):
        input_image[i, :, :]  -= mean[i]
        input_image[i, :, :] /= std[i]
    return input_image


def put_data_to_dataloader(data, batch_size, shuffle=True):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader


def prep_dataloaders(train, val_split=0.2, batch_size=16, shuffle=True):
    t, v = torch.utils.data.random_split(train,
                                         [int(len(train) * (1 - val_split)), int(len(train) * val_split)],
                                         torch.Generator().manual_seed(10))
    dataloader_dict = {"train": put_data_to_dataloader(t, batch_size, shuffle), 
                       "val": put_data_to_dataloader(v, batch_size, shuffle)}
    return dataloader_dict