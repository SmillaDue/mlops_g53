import math

import cv2
import imutils
import matplotlib.pyplot as plt
import monai
import torch
from monai.transforms import Transform


class NormalizeImage(Transform):
    def __call__(self, img):
        return self.normalize(img)

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images."""
        return (images - images.mean()) / images.std()


class CropImage(Transform):
    """Crop image using extreme points."""

    def __call__(self, img):
        return self.crop(img)

    def crop(self, img):
        """
        Finds the extreme points on the image and crops the rectangular out of them
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


class LoadImageFromCV(Transform):
    """Load image using OpenCV."""

    def __call__(self, img_path):
        return cv2.imread(str(img_path))


class ToGrayCHW(Transform):
    """Ensure output is grayscale with shape (1, H, W)."""

    def __call__(self, img):
        x = torch.as_tensor(img)

        # If RGB/RGBA with channels last: (H, W, C)
        if x.ndim == 3 and x.shape[-1] in (3, 4):
            x = x[..., :3].float().mean(dim=-1)  # -> (H, W)

        # If channels first: (C, H, W)
        elif x.ndim == 3 and x.shape[0] in (3, 4):
            x = x[:3].float().mean(dim=0)  # -> (H, W)

        # If already (H, W), keep it
        if x.ndim == 2:
            x = x.unsqueeze(0)  # -> (1, H, W)

        return x


def describe_compose(c: monai.transforms.Compose) -> None:
    """
    Utility function to describe the transforms in a Compose object
    """
    return "".join([f"- {getattr(t, '__name__', t.__class__.__name__)}\n" for t in c.transforms])


def show_image_and_target(images, targets, show=True):
    """Display images with their corresponding targets in a single grid."""

    n = len(images)
    cols = int(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for ax, image, target in zip(axes, images, targets):
        if hasattr(image, "permute"):  # torch.Tensor
            image = image.permute(1, 2, 0)  # -> (H, W, 3)

        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"{int(target)}")
        ax.axis("off")

    # Hide unused subplots if grid is larger than n
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()

    if show:
        plt.show()
