import os
# os.chdir("..")
from copy import deepcopy
from experiments.matching import *
from experiments.preprocessing import *
from experiments.drawing import *
from experiments.data_management import *
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from src.loftr import LoFTR, default_cfg
import time
RUN_EDGES = False
DRAW_EPIPOLAR_LINES = False
CONF_FACTOR = 0.7


os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

BASE_DIR ="/home/avichaih/Projects/MyLoFTR"
BASE_DATA_DIR = os.path.join(BASE_DIR, 'data')

def remove_outliers_by_epipolar_constraint(kpts0: np.ndarray, kpts1: np.ndarray, conf: np.ndarray):
    """
    Remove outliers by epipolar constraint
    """
    Fm, inliers = cv2.findFundamentalMat(kpts0, kpts1, cv2.FM_RANSAC, 0.5, 0.999, 100000)
    kpts0 = kpts0[inliers.flatten() == 1, :]
    kpts1 = kpts1[inliers.flatten() == 1, :]
    conf = conf[inliers.flatten() == 1]
    return kpts0, kpts1, conf, Fm

def run_inference(img0: np.ndarray, img1: np.ndarray, weights_path: str = None, width_len: int =640, is_indoor: bool = False, debug: bool = False,):
    # initiate LoFTR model
    matcher = init_model(weights_path=weights_path, is_indoor=is_indoor)

    # Preprocess images
    img0_torch, img1_torch, img0_resize, img1_resize = matching_images_preprocess(img0_raw=img0, img1_raw=img1, is_indoor=is_indoor, debug=debug, width_len=width_len)

    # run matcher inference
    mkpts0, mkpts1, mconf = matching_by_loftr(img0_torch, img1_torch, matcher)

    return mkpts0, mkpts1, mconf, img0_resize, img1_resize

def run_demo(img0: np.ndarray, img1: np.ndarray, weights_path: str = None,width_len: int =640, remove_outliers: bool = False, is_indoor: bool = False, debug: bool = False,save_path: str = None):
    # inference
    kpts0_loftr, kpts1_loftr, conf_loftr, img0_resize, img1_resize = run_inference(img0=img0, img1=img1, weights_path=weights_path, is_indoor=is_indoor, debug=debug, width_len=width_len)

    # remove outliers by epipolar constraint
    if remove_outliers:
        kpts0_loftr, kpts1_loftr, conf_loftr, Fm = remove_outliers_by_epipolar_constraint(kpts0_loftr, kpts1_loftr, conf_loftr)
    else:
        Fm = None

    # draw matches
    draw_matches_on_images(img0_resize, img1_resize, kpts0_loftr, kpts1_loftr, conf_loftr, title='LoFTR Matcher - reference (left) to query (right)',
                       draw_epipolar_lines=DRAW_EPIPOLAR_LINES, f_matrix=Fm, conf_factor=CONF_FACTOR, save_path=save_path)


def load_outdoor_demo_data(debug: bool  = False):
    # outdoor demo
    # img0_outdoor = "/home/ubuntu/projects/lang-segment-anything/results/gaza1/0.png"
    # img1_outdoor = "/home/ubuntu/projects/lang-segment-anything/results/gaza1/10.png"
    img0_outdoor = "/home/ubuntu/Data/source.png"
    img1_outdoor = "/home/ubuntu/Data/target.png"
    weights_outdoor = "weights/outdoor_ds.ckpt"

    # load images
    img0 = load_image(img0_outdoor)
    img1 = load_image(img1_outdoor)

    if debug:
        eq_img0 = equalize_hist(img0)
        eq_img1 = equalize_hist(img1)
        # plot image histogram
        his1 = cv2.calcHist([img0], [0], None, [256], [0, 256])
        his2 = cv2.calcHist([eq_img0], [0], None, [256], [0, 256])

        plt.plot(his1, color='r', label='source original')
        plt.plot(his2, color='b', label='source equalized')
        plt.legend()
        plt.show()
        # plot original images and equalized images
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].imshow(img0, 'gray')
        ax[0, 0].set_title('original image source')
        ax[0, 1].imshow(eq_img0, 'gray')
        ax[0, 1].set_title('equalized image source')
        ax[1, 0].imshow(img1, 'gray')
        ax[1, 0].set_title('original image target')
        ax[1, 1].imshow(eq_img1, 'gray')
        ax[1, 1].set_title('equalized image target')
        plt.show()
    return img0, img1, weights_outdoor


def run_queries(base_path:str = '/home/ubuntu/Data/'):
    queries_files_path = base_path + 'W7_EDlXWTBiXAEEniNoMPwAAYamdpeGl2cXZqAYsGfNuqAYsGfNtTAAAAAQ/queries/'
    references_files_path = base_path +'/W7_EDlXWTBiXAEEniNoMPwAAYamdpeGl2cXZqAYsGfNuqAYsGfNtTAAAAAQ/references/'
    num_imgs = len(os.listdir(queries_files_path))
    for i in tqdm(range(num_imgs)):
        img0 = load_image(os.path.join(references_files_path, f'{i}.png'))
        img1 = load_image(os.path.join(queries_files_path, f'{i}.png'))
        save_path = 'results/' + queries_files_path.split('Data/')[-1].split('queries/')[
            0] + f'matching_frame_idx_{i}.png'

        run_demo(img0, img1, weights_path="weights/outdoor_ds.ckpt", is_indoor=False, remove_outliers=False,
                 width_len=512, save_path=save_path)



if __name__ == '__main__':
    from tqdm import tqdm
    from loftr_inference import *
    img0, img1, weights = load_outdoor_demo_data(False)
    run_inference(img0, img1, weights_path=weights, is_indoor=False, show_matches=True)
    # run_demo(img0, img1, weights_path=weights,is_indoor=False, remove_outliers=False, width_len=512, save_path=None)


