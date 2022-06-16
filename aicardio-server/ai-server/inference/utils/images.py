import numpy as np
import cv2


__all__ = ["resize", "crop_around_center"]


def resize(image, target_size):
    r"""Resize image to some target size.
    
    Args:
        image (np.array): Input BGR image, i.e. np.array of shape (h, w, 3).
        target_size (tuple): Target size, i.e. (w, h).
    Returns:
        np.array: Output BGR image, i.e. np.array of shape (h, w, 3).
    """
    image = _pad_to_ratio(image, target_size[0] / target_size[1])
    image = cv2.resize(image, tuple(target_size))
    return image

def _pad_to_ratio(image, ratio):
    r"""Zero-pad an image to a given width-height ratio.
    
    Args:
        image (np.array): Input BGR image, i.e. np.array of shape (h, w, 3).
        ratio (float): Width / height ratio. 
    Returns:
        np.array: Padded image, i.e. np.array of shape (h, w, 3).
    """
    h, w, _ = image.shape
    w_pad, h_pad = int(max(w, h * ratio)), int(max(h, w / ratio))
    pad_top, pad_left = (h_pad - h) // 2, (w_pad - w) // 2
    pad_bottom, pad_right = h_pad - h - pad_top, w_pad - w - pad_left
    image = np.pad(
        image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant"
    )
    return image

def crop_around_center(image, center, box_size):
    r"""Crop a patch of size `box_size` around a point `center` from `image`"""
    x1, y1 = center[0] - (box_size[0] - 1)//2, center[1] - (box_size[1] - 1)//2
    x2, y2 = x1 + box_size[0], y1 + box_size[1]
    cropped = image[y1:y2, x1:x2]
    return cropped