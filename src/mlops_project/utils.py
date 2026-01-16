import math

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import ScaleIntensity

def show_image_and_target(images, targets, show=True):
    """Display images with their corresponding targets in a single grid."""

    n = len(images)
    cols = int(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for ax, image, target in zip(axes, images, targets):
        if hasattr(image, "permute"):  # torch.Tensor
            image = ScaleIntensity()(image)
            image = image.permute(1, 2, 0)  # -> (H, W, 1)
            image = image.numpy()

        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"{int(target)}")
        ax.axis("off")

    # Hide unused subplots if grid is larger than n
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()

    if show:
        plt.show()

def crop_img(img: np.ndarray) -> np.ndarray:
    """Crop image to remove black borders using contour detection.

    Finds the extreme points on the image and crops the rectangular out of them.

    Args:
        img: Input image as numpy array in RGB format.

    Returns:
        Cropped image as numpy array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[
        extTop[1] - ADD_PIXELS : extBot[1] + ADD_PIXELS, extLeft[0] - ADD_PIXELS : extRight[0] + ADD_PIXELS
    ].copy()

    return new_img

class ArrayPreprocessing():
    def __init__(self, img_size, crop_img = True):
        self.img_size = img_size
        self.crop_img = crop_img
    
    def __call__(self, img_path):
        img = self.load(img_path)
        
        if self.crop_img == True:
            img = self.crop(img)
            
        img = self.gray_scale(img)
        img = self.resize(img)
        img = img[None,:,:] #shape (1,H,W)
        return img
        
    def load(self, img_path):
        return cv2.imread(str(img_path))
    
    def crop(self, img):
        return crop_img(img)
    
    def gray_scale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def resize(self, img):
        return cv2.resize(img, (self.img_size, self.img_size))
    
class TensorsPreprocessing():
    def __call__(self, imgs):
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs).float() / 255.0
        imgs = (imgs - imgs.mean()) / (imgs.std())
        return imgs