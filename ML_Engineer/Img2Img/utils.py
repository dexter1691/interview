import cv2
import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from torchvision.utils import make_grid

def visualize(batch_dict):
    """Visualize a batch of images side by side for easy comparison
    
    Args: 
        batch_dict: {key: (B, C, H, W) tensor containing image data}.
    """ 
    num_cols = len(batch_dict.keys())
    
    plt.figure(figsize=(5, 5*num_cols), dpi=300)
    
    for i, k in enumerate(batch_dict.keys()):
        plt.subplot(1, num_cols, i + 1)
        batch_size = batch_dict[k].shape[0]
        num_row_tiles = int(np.sqrt(batch_size))
        # TODO: This doesn't work great for Grayscale images, but we can make do.
        img = make_grid(batch_dict[k], num_row_tiles).permute(1, 2, 0).to(torch.float)
        
        plt.imshow(img/img.max())
        plt.title(k)
        plt.axis("off")
    
    plt.show()

def process_image(example):
    """ Process (a batch of) images with a custom kernel. 
    Args:
        example: dict containing image data.
    
    Returns:
        output dict containing image and output data.
    """

    example['image'] = np.array(example['image'])
    # Only convert, if the image has three channels. This assumes that images do not have alpha channel.
    if example['image'].shape[2] == 3:
        example['image'] = cv2.cvtColor(example['image'], cv2.COLOR_RGB2GRAY)
    
    # resize to a fixed small size. This should be sufficient to predict the Sobel filter.
    example['image'] = cv2.resize(example['image'], (64, 64), cv2.INTER_AREA)
    sb_img_x = cv2.Sobel(example['image'], dx=1, dy=0, ddepth=cv2.CV_64F, ksize=3, borderType=cv2.BORDER_CONSTANT)
    sb_img_y = cv2.Sobel(example['image'], dx=0, dy=1, ddepth=cv2.CV_64F, ksize=3, borderType=cv2.BORDER_CONSTANT)
    example['image_np'] = torch.Tensor(example['image'])[None]
    example['sb_img_x'] = torch.Tensor(sb_img_x)[None]
    example['sb_img_y'] = torch.Tensor(sb_img_y)[None]
    
    example['sb_img'] = torch.sqrt(torch.sum(torch.concat([torch.pow(example['sb_img_x'], 2), torch.pow(example['sb_img_y'], 2)]), axis=0))[None]
    return example

def convolve_with_kernel(img, kernel):
    """Convolve image with the kernel in batched mode.
    Args:
        img: 4D Tensor (B, C, H, W)
        kernel: 4D Tensor (C_out, C_in, H, W)
    
    Returns:
        output image Tensor (B, C, H, W)
    """
    output = nn.functional.conv2d(img, kernel, None, padding='same')
    return output
    
def process_image_with_custom_kernel(kernel, example):
    """ Process (a batch of) images with a custom kernel. 
    Args:
        kernel: 4D Tensor (C_out, C_in, H, W).
        example: dict containing image data.
    
    Returns:
        output dict containing image and output data.
    """
    kernel = torch.Tensor(kernel)
    kernel = torch.flip(kernel, [2, 3])
    
    example['image'] = np.array(example['image'])
    # Only convert, if the image has three channels. This assumes that images do not have alpha channel.
    if example['image'].shape[2] == 3:
        example['image'] = cv2.cvtColor(example['image'], cv2.COLOR_RGB2GRAY)
    
    # resize to a fixed small size. This should be sufficient to predict the Sobel filter.
    example['image_np'] = cv2.resize(example['image'], (64, 64), cv2.INTER_AREA)
    example['image_np'] = torch.Tensor(example['image_np'])[None]
    output = convolve_with_kernel(example['image_np'][None], kernel)
    example['output'] = output[0]
    return example