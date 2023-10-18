# %%    IMPORT PACKAGES
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_matching_images(img0: np.ndarray, img1: np.ndarray,img0_to_show: np.ndarray, width_len: int = 480, keep_aspect_ratio: bool = True):
    """
    Resize images to a given height length, while keeping the aspect ratio.
    """
    new_img0_h, new_img1_h = get_new_height_for_img(img0.shape, img1.shape, width_len,keep_aspect_ratio)

    # resize images
    img0_resize = cv2.resize(img0, (new_img0_h, width_len))
    img1_resize = cv2.resize(img1, (new_img1_h, width_len))
    img0_to_show = cv2.resize(img0_to_show, (new_img1_h, width_len))

    # padding images
    pad_size = abs(new_img1_h - new_img0_h)
    if new_img0_h < new_img1_h:
        # pad image 0
        img0_with_padding = np.pad(img0_resize, ((0, 0), (0, pad_size)), 'constant', constant_values=(0))
        return img0_resize, img1_resize,img0_to_show, img0_with_padding, img1_resize
    else:
        # pad image 1
        img1_with_padding = np.pad(img1_resize, ((0, 0), (0, pad_size)), 'constant', constant_values=(0))
        return img0_resize, img1_resize,img0_to_show, img0_resize, img1_with_padding
def get_new_height_for_img(img0_shape, img1_shape, width_len: int = 480,
                               keep_aspect_ratio: bool = True):
    """
    Calculate new height for images to a given width length, while keeping the aspect ratio.
    """
    if keep_aspect_ratio:
        h0, w0 = img0_shape
        h1, w1 = img1_shape
        new_img0_h = int(((w0 / h0) * width_len) // 8 * 8)
        new_img1_h = int(((w1 / h1) * width_len) // 8 * 8)
    else:
        new_img0_h, new_img1_h = width_len, width_len
    return new_img0_h, new_img1_h

def normalize_image(image):
    """
    Normalize image to [0, 1]
    """
    image = image - np.min(image)
    image = image / np.max(image)
    return image
def equalize_hist(image):
    """
    Equalize image histogram
    """
    equalized_image = cv2.equalizeHist(image.astype(np.uint8))
    return (normalize_image(equalized_image) * 255.).astype(np.uint8)
def get_divisible_wh(w, h, df=None):
    """
    Get divisible width and height
    """
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new
def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def pad_match_images(h0, h1, img0, img1):
    # padding images
    pad_size = abs(h1 - h0)
    if h0 < h1:
        # pad image 0
        img0 = np.pad(img0, ((0, 0), (0, pad_size)), 'constant', constant_values=(0))
    else:
        # pad image 1
        img1 = np.pad(img1, ((0, 0), (0, pad_size)), 'constant', constant_values=(0))
    return img0, img1


def img2net_input(img):
    """
    Convert image to network input
    """
    return torch.from_numpy(img)[None][None].cuda() / 255.


def matching_images_preprocess(img0_raw: np.ndarray, img1_raw: np.ndarray,equaliz_depth: bool = True,  is_indoor: bool = False,width_len: int = 640, debug=False, plot_equalized_hist: bool = False):
    """
    Preprocess images for matching inference
    """

    if equaliz_depth:
        # equalize depth image
        img1_raw = equalize_hist(img1_raw)

        if plot_equalized_hist:
            # Plot the equalized image
            plt.subplot(2, 1, 1)
            plt.title('Equalized Image')

            plt.imshow(normalize_image(img1_raw.astype(np.float32)), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 1, 2)
            histogram, bins = np.histogram(img1_raw.flatten(), bins=256, range=[1, 256])
            plt.title('Equalized Image Pixel Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.bar(bins[:-1], histogram, width=1, align='center', color='b')
            plt.xlim([0, 256])

            plt.tight_layout()
            plt.show()

    print(f"origin image 0 shape: {img0_raw.shape}\norigin image 1 shape: {img1_raw.shape}")

    # Preprocess images
    if is_indoor:
        img0_resize = cv2.resize(img0_raw, (640, 480))
        img1_resize = cv2.resize(img1_raw, (640, 480))

        if debug:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img0_resize, cmap='gray')
            ax[1].imshow(img1_resize, cmap='gray')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            plt.show()
    else:
        new_h0, new_h1 = get_new_height_for_img(img0_raw.shape, img1_raw.shape, width_len, keep_aspect_ratio=True)
        img0_resize = cv2.resize(img0_raw, (new_h0, width_len))
        img1_resize = cv2.resize(img1_raw, (new_h1, width_len))
        print(f"resize image 0 shape: {img0_resize.shape}\nresize  image 1 shape: {img1_resize.shape}")
    # prepare images for inference
    img0_torch = img2net_input(img0_resize)
    img1_torch = img2net_input(img1_resize)
    return img0_torch, img1_torch, img0_resize, img1_resize