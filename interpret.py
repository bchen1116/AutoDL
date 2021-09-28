from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from numpy import moveaxis
import matplotlib.pyplot as plt
import cv2


def unnormalize(input_tensor, multiple=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # now, we can unnormalize all the images
    for i in range(3):
        input_tensor[i, :, :] *= std[i]
        input_tensor[i, :, :]  += mean[i]
    if multiple:
        input_tensor = input_tensor.multiply(255)
        input_tensor = input_tensor.type(torch.int64)
    return input_tensor

def plot_grad_cams(model, input_image_tensor, use_cuda=True):
    rgb_img = unnormalize(input_image_tensor.squeeze(0)).cpu().numpy()
    rgb_img = moveaxis(rgb_img, 0, 2)
    input_image_tensor = unnormalize(input_image_tensor.squeeze(0)).unsqueeze(0)
    if use_cuda:
        input_image_tensor = input_image_tensor.cuda()
    if "VGG" in model.name or "DenseNet" in model.name:
        target_layer = model.model.features[-1]
    elif "ResNet" in model.name:
        target_layer = model.model.layer4[-1]
    else:
        raise ValueError("Grad Cam only supported for ResNet, VGG, and DenseNet models!")

    cam = GradCAM(model=model.model, target_layer=target_layer)
    grayscale_cam = cam(input_tensor=input_image_tensor, target_category=None, eigen_smooth=False, aug_smooth=False)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
    fig, ax = plt.subplots(1, 3)
    RGB_im = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)

    ax[0].imshow(rgb_img)
    ax[1].imshow(visualization)
    ax[2].imshow(RGB_im)
