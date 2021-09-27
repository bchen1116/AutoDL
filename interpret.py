from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def plot_grad_cam(model, input_image_tensor):
    if "VGG" in model.name or "DenseNet" in model.name:
        target_layer = model.features[-1]
    elif "ResNet" in model.name:
        target_layer = model.layer4[-1]
    else:
        raise ValueError("Grad Cam only supported for ResNet, VGG, and DenseNet models!")

    cam = GradCam(model=model, target_layer=target_layer)
    grayscale_cam = cam(input_tensor=input_image_tensor, target_category=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
    return visualization
