import os
os.chdir("..")
from copy import deepcopy
 
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
 
 
from src.loftr import LoFTR, default_cfg
 
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
 
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("/mnt/d/GitWarehouse/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()
 
 
default_cfg['coarse']
 
# Load example images
img0_pth = "/mnt/d/GitWarehouse/LoFTR/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
img1_pth = "/mnt/d/GitWarehouse/LoFTR/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))
 
img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}
 
# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
 
# Draw
color = cm.jet(mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text,path="/mnt/d/GitWarehouse/LoFTR/LoFTR-colab-demo.pdf")