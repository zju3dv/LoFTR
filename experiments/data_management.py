
import matplotlib.pyplot as plt
import cv2
import numpy as np

CONF_FACTOR = 0.0
RUN_EDGES = False
DRAW_EPIPOLAR_LINES = False
from skimage import filters
import h5py
import io
try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None
def normalize_image(image):
    """
    Normalize image to [0, 1]
    """
    image = image - np.min(image)
    image = image / np.max(image)
    return image

def load_image(img_path, normalize_img: bool = True, max2zero : bool = False):
    """
    Load image from path
    """
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if normalize_img:
        img = normalize_image(img)

    if max2zero: # convert all max values to 0
        img[np.max(img) == img] = 0

    # convert to [0, 255] range
    img = (img * 255.).astype(np.uint8)
    return img

def get_resized_wh(w, h, resize=None):
    """
    Get resized width and height
    """
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new
def load_depth_file_as_image(path, resize=None, debug=False):
    """
    Read depth file as image
    """
    if str(path).startswith('s3://'):
        image = load_array_from_s3(path, None, None, use_h5py=True)
    else:
        image = np.array(h5py.File(path, 'r')['depth'])

    # normalize image
    image = normalize_image(image)

    # resize image
    if resize is not None:
        w, h = image.shape[1], image.shape[0]
        w_new, h_new = get_resized_wh(w, h, resize)

        image = cv2.resize(image, (w_new, h_new))
    # show depth image for debug
    if debug:
        plt.imshow(image, cmap='jet')
        plt.show()

    return image * 255.
def load_array_from_s3(path, client, cv_type,use_h5py=False,):
    """
    Load array from s3 path
    """
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data

def read_float32_image(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    # img = img / 65535.0
    # image histogram
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.figure()
    plt.imshow(img.astype(np.uint8), cmap='gray')
    plt.show()
    return img