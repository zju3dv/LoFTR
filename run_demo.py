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
CONF_FACTOR = 0.0


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

def run_demo(img0: np.ndarray, img1: np.ndarray, weights_path: str = None,width_len: int =640, img_depth_path: str = None, remove_outliers: bool = False, is_indoor: bool = False, debug: bool = False,save_path: str = None):
    # inference
    kpts0_loftr, kpts1_loftr, conf_loftr, img0_resize, img1_resize = run_inference(img0=img0, img1=img1, weights_path=weights_path, is_indoor=is_indoor, debug=debug, width_len=width_len)

    # remove outliers by epipolar constraint
    if remove_outliers:
        kpts0_loftr, kpts1_loftr, conf_loftr, Fm = remove_outliers_by_epipolar_constraint(kpts0_loftr, kpts1_loftr, conf_loftr)
    else:
        Fm = None

    # draw matches
    if img_depth_path is not None:
        img_depth = cv2.resize(load_image(img_depth_path), (img1_resize.shape[1], img1_resize.shape[0]))
        draw_matches_on_images(img0_resize, img_depth, kpts0_loftr, kpts1_loftr, conf_loftr, title='LoFTR Matcher - Sensor To Reference Depth Model',
                               draw_epipolar_lines=DRAW_EPIPOLAR_LINES, f_matrix=Fm, conf_factor=CONF_FACTOR, save_path=save_path)
    else:
        draw_matches_on_images(img0_resize, img1_resize, kpts0_loftr, kpts1_loftr, conf_loftr, title='LoFTR Matcher - Sensor To Reference Depth Model',
                           draw_epipolar_lines=DRAW_EPIPOLAR_LINES, f_matrix=Fm, conf_factor=CONF_FACTOR, save_path=save_path)


def load_outdoor_demo_data():
    # outdoor demo
    img0_outdoor = "data/megadepth_test_1500/Undistorted_SfM/0015/images/570188204_952af377b3_o.jpg"
    # img0_outdoor = "data/megadepth_test_1500/Undistorted_SfM/0015/images/841149791_2ae77144de_o.jpg"
    img1_outdoor = "data/megadepth_test_1500/Undistorted_SfM/0015/images/841149791_2ae77144de_o.jpg"
    weights_outdoor = "weights/outdoor_ds.ckpt"

    # load images
    img0 = load_image(img0_outdoor)
    img1 = load_image(img1_outdoor)

    run_demo(img0, img1, weights_path=weights_outdoor,
             is_indoor=False,remove_outliers=False)

def load_outdoor_depth_data(depth2edges: bool = False):
    data_path = '/opt/Data/megadepth'
    # file_name = '86236224_a6c9ec375d_o'
    file_name = '3399431445_233fbedd3c_o'
    depth_different_angle ='3594397839_f9db736dff_o'
    depth_path = os.path.join(data_path, 'train/phoenix/S6/zl548/MegaDepth_v1/0063/dense0/depths',depth_different_angle+'.h5')
    img_depth_path = os.path.join(data_path, 'train/phoenix/S6/zl548/MegaDepth_v1/0063/dense0/imgs', depth_different_angle+'.jpg')
    sensor_path = os.path.join(data_path, 'train/phoenix/S6/zl548/MegaDepth_v1/0063/dense0/imgs', file_name+'.jpg')

    img0 = load_image(sensor_path)
    img1 = load_depth_file_as_image(depth_path)
    weights = WEIGHTS_PATH
    if depth2edges:
        weights = "weights/depthAsEdges.ckpt"

    return img0, img1, weights, img_depth_path


def load_indoor_data():
    # indoor demo
    img0_idoor = "data/scannet_test_1500/scene0711_00/color/1680.jpg"
    img1_idoor = "data/scannet_test_1500/scene0711_00/color/1995.jpg"
    weights = "weights/indoor_ds_new.ckpt"
    img0 = load_image(img0_idoor)
    img1 = load_image(img1_idoor)
    return img0, img1, weights


def load_ir_data():
    # outdoor demo
    img0 = load_image("data/ir/028360.tiff")
    img1 = load_image("data/ir/028370.tiff")
    weights = "weights/outdoor_ds.ckpt"
    return img0, img1, weights
def load_rahfanim_data():
    # indoor demo
    img0_idoor = "/home/avichaih/Desktop/rahfanim data/flight3-1-A40-R50/DJI_0374.JPG"
    img1_idoor = "/home/avichaih/Desktop/rahfanim data/flight3-1-A40-R50/DJI_0380.JPG"
    weights = "weights/outdoor_ds.ckpt"
    img0 = load_image(img0_idoor)
    img1 = load_image(img1_idoor)
    return img0, img1, weights


def load_render_data():
    # outdoor demo
    img0_outdoor = BASE_DATA_DIR + "/rf_data/warpedImage.tif"
    # img1_outdoor = BASE_DIR + BASE_DATA_DIR + "/rf_data/model_image.tif"
    img1_outdoor = BASE_DATA_DIR + "/rf_data/depth_image.tif"
    img_depth_path =  BASE_DATA_DIR + "/rf_data/model_image.tif"
    img0 =  load_image(img0_outdoor)
    img1 = load_image(img1_outdoor)

    weights = WEIGHTS_PATH
    # weights_outdoor = "weights/depth.ckpt"
    return img0, img1, weights,img_depth_path

def load_mala_data(depth2edges=False):
    # outdoor demo
    img0_outdoor = BASE_DATA_DIR + "/mala/02_original.tiff"
    img1_outdoor = BASE_DATA_DIR + "/mala/render_dsm_depth.tif"

    img0 = load_image(img0_outdoor)
    img1 = load_image(img1_outdoor, max2zero=True)
    weights = WEIGHTS_PATH
    if depth2edges:
        weights = "weights/depthAsEdges.ckpt"
    return img0, img1, weights

if __name__ == '__main__':
    WEIGHTS_PATH = "logs/tb_logs/outdoor-ds-480-bs=8-edges-2023-08-13/version_0/checkpoints/epoch=29-auc@5=0.089-auc@10=0.208-auc@20=0.370.ckpt"
    # WEIGHTS_PATH = "logs/tb_logs/outdoor-ds-640-bs=4-edges-2023-08-13/version_0/checkpoints/epoch=14-auc@5=0.110-auc@10=0.231-auc@20=0.395.ckpt"
    # WEIGHTS_PATH = "weights/outdoor_ds.ckpt"
    # save_path = 'experiments/viz_results/mala_case.png'
    img_depth_path = None
    # img0, img1, weights, img_depth_path = load_outdoor_depth_data()
    
    # img0, img1, weights = load_indoor_data()
    # img0, img1, weights = load_outdoor_demo_data()
    # img0, img1, weights = load_ir_data()
    # img0, img1, weights = load_rahfanim_data()
    img0, img1, weights, img_depth_path = load_render_data()

    # img0, img1, weights = load_mala_data(depth2edges=False)
    run_demo(img0, img1, weights_path=weights,is_indoor=False, remove_outliers=False, width_len=640, img_depth_path=img_depth_path, save_path=None)


