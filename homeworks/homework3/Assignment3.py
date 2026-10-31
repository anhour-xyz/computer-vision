import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, torch_img):
        rgb_img = cv.cvtColor(torch_img, cv.COLOR_BGR2RGB)
        torch_img = torch.from_numpy(rgb_img).to(torch.float32)
        return torch_img

    def brighten(self, torch_img):
        bright_img = torch_img + 100.0
        return bright_img

    def saturation_arithmetic(self, img):
        rgb_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb_img).to(torch.int16)
        saturated_img = (img_tensor + 100).clamp(0,255).to(torch.uint8)
        return saturated_img

    def add_noise(self, torch_img):
        noise = torch.randn_like(torch_img) * 100.0
        noisy_img = ((torch_img + noise) / 255.0).clamp(0.0,1.0)  
        return noisy_img

    def normalization_image(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb_img).to(torch.float64) / 255.0
        mean = img_tensor.mean(dim=(0,1))
        std = img_tensor.std(dim=(0,1), correction = 0)
        image_norm = (img_tensor - mean) / std
        return image_norm

    def Imagenet_norm(self, img):
        rgb_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb_img).to(torch.float64) / 255.0
        mean = torch.tensor([0.485,0.456,0.406], dtype = torch.float64)
        std = torch.tensor([0.229, 0.224, 0.225], dtype = torch.float64)
        ImageNet_norm = ((img_tensor - mean) / std).clamp(0.0,1.0)
        return ImageNet_norm

    def dimension_rearrange(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb_img).to(torch.float32)
        rearrange = img_tensor.permute(2,0,1).unsqueeze(0)
        return rearrange

    def stride(self, img):
        img_tensor = torch.from_numpy(img).to(torch.float32)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        kernel = torch.tensor([
            [-3,0,3],
            [-10,0,10],
            [-3,0,3]
        ],dtype = torch.float32)
        kernel = torch.flip(kernel, dims=(0, 1))
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        res = F.conv2d(img_tensor, kernel,stride= 2, padding = 1)
        stride_imgage = res[0,0]
        return stride_imgage


if __name__ == "__main__":
    img = cv.imread("original_image.png")
    img_cat = cv.imread("cat_eye.jpg", cv.IMREAD_GRAYSCALE)
    assign = Assignment3()
    torch_img = assign.torch_image_conversion(img)
    bright_img = assign.brighten(torch_img)
    saturated_img = assign.saturation_arithmetic(img)
    noisy_img = assign.add_noise(torch_img)
    image_norm = assign.normalization_image(img)
    ImageNet_norm = assign.Imagenet_norm(img)
    rearrange = assign.dimension_rearrange(img)
    stride_imgage = assign.stride(img_cat)

    plt.imshow(ImageNet_norm.numpy())
    plt.axis("off")
    plt.savefig("output_image.png", bbox_inches="tight", pad_inches=0)
    plt.show()

