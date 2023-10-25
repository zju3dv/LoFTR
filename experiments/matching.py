import torch
import cv2

from src.loftr import LoFTR, default_cfg
import time



def matching_by_loftr(img0, img1, matcher):
    # inputs to matcher
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        # measure time
        start_time = time.time()
        matcher(batch)
        end_time = time.time()
        print(f"running duration {(end_time - start_time):.2f} seconds")
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    return mkpts0, mkpts1, mconf

def matching_by_sift(img0, img1):
    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(img0, None)
    kp1, des1 = sift.detectAndCompute(img1, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.78 * n.distance:
            good.append([m])
    return kp0, kp1, good

def init_model(weights_path):
    """
    Init LoFTR model
    """
    # init model
    model = LoFTR(config=default_cfg)
    # load the pretrained model
    model.load_state_dict(torch.load(weights_path)['state_dict'])
    model = model.eval().cuda()
    return model