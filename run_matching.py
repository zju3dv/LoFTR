from typing import Union, List, Dict, Tuple
import os
from collections import defaultdict
from copy import deepcopy
import glob
import sqlite3

import torch
import cv2
import numpy as np
import numpy as np
import paralleldomain as pd
from paralleldomain import AnyPath

from loftr.utils.colmap_database import (
    COLMAPDatabase, array_to_blob, blob_to_array, image_ids_to_pair_id, pair_id_to_image_ids
)
from loftr.utils.plotting import make_matching_figure
from loftr.loftr import LoFTR, default_cfg

# os.chdir("..")

def parse_image_names(image_map):
    for idx,(img0_id, img0_name) in enumerate(image_map):
        for img1_id, img1_name in image_map[idx+1:]:
            pair_id = image_ids_to_pair_id(img0_id,img1_id)
            yield pair_id, img0_id, img0_name, img1_id, img1_name

def preprocess_images(img_dir: Union[AnyPath,str], img0_name: Union[AnyPath,str], img1_name: Union[AnyPath,str], dims=(1152,768)) -> Dict[str,torch.Tensor]:
    img0_pth = AnyPath(img_dir) / img0_name
    img1_pth = AnyPath(img_dir) / img1_name
    img0_raw = cv2.resize(cv2.cvtColor(pd.fsio.read_image(img0_pth), cv2.COLOR_BGR2GRAY),dims)
    img1_raw = cv2.resize(cv2.cvtColor(pd.fsio.read_image(img1_pth), cv2.COLOR_BGR2GRAY),dims)
    # img0_raw = cv2.resize(cv2.imread(str(img0_pth), cv2.IMREAD_GRAYSCALE),dims)
    # img1_raw = cv2.resize(cv2.imread(str(img1_pth), cv2.IMREAD_GRAYSCALE),dims)
    # Check that input size is divisible by 8
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))
    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    return batch

if __name__ == "__main__":
    
    db_path = "/home/michael/datasets/zkm_garage_downscaled/database.db"

    db = COLMAPDatabase.connect(db_path)
    # Create all the tables up front
    db.create_tables()

    # Insert the camera into the db
    #TODO: Check if these cam_params change for different datasets?
    cam_params = np.array([2667.27421246, 2667.27421246, 3000., 2000., 0., 0., 0., 0.])
    camera_id = db.add_camera(camera_id=1,model=4,width=6000,height=4000,params=cam_params,prior_focal_length=1)

    # Populate images table
    img_dir = AnyPath("s3://pd-internal-ml/location_creation_tests/2024_01_22_zkm_garage_downscaled/originalImages/")
    # image_paths = glob.glob(img_dir+"*.JPG")
    camera_id=1
    for id,image_path in enumerate(img_dir.iterdir()):
        db.add_image(image_id=id,name=image_path.name,camera_id=camera_id)

    image_map = db.execute("SELECT image_id, name FROM images").fetchall()

    # Instantiate the model
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("/home/michael/other_repos/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()

    # Collect keypoint entries for each image
    # TODO: This might be too memory intensive, we could remove completed image entries based on the image_map loop
    keypoint_blob_dict = defaultdict(dict)
    #TODO: rescale to full images?
    for pair_count, (pair_id, img0_id, img0_name, img1_id, img1_name) in enumerate(parse_image_names(image_map)):

        batch = preprocess_images(img_dir=img_dir, img0_name=img0_name, img1_name=img1_name)
        # batch = preprocess_images(img_dir=img_dir, img0_name="IMG_8236.JPG", img1_name="IMG_8237.JPG")
        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        
        print(f"Matches: {len(mkpts0)}")
        print(f"Conf score mean: {mconf.mean()}\tMin: {mconf.min()}\tMax: {mconf.max()}")
        print(f"Confidence scores: {mconf[:100]}")
        break
    """
        match_table_entry = []
        idx0 = 0
        idx1 = 0
        for coords0,coords1 in zip(mkpts0.astype(int),mkpts1.astype(int)):
            # Each keypoint is added to keypoint_blob_dict[img_id].
            # The point's index in that list is entered into the matches table.
            feat_idx0 = keypoint_blob_dict[img0_id].setdefault(tuple(coords0),idx0)
            feat_idx1 = keypoint_blob_dict[img1_id].setdefault(tuple(coords1),idx1)
            if idx0 == feat_idx0:
                idx0 += 1
            if idx1 == feat_idx1:
                idx1 += 1
            # Add to matching table blob
            match_table_entry.append([feat_idx0,feat_idx1])
        # Write matching blob to matches table
        db.execute(
                "INSERT INTO matches VALUES (?, ?, ?, ?)",
                (pair_id,) + mkpts0.shape + (array_to_blob(np.array(match_table_entry)),
                ),)
        if pair_count % 100 == 0:
            print(f"{pair_count} / {len(image_map)} pairs processed.")
            db.commit()

        # Store confidences
        db.add_confidence(img0_id,img1_id,mconf)

    # Write the keypoint table
    for image_id,keypoint_dict in keypoint_blob_dict.items():
        keypoints = np.array(list(keypoint_dict.keys()))
        db.execute(
                "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
                (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
            )
        
    db.commit()
    """