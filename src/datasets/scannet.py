from os import path as osp
import numpy as np
import torch
import torch.utils as utils
from src.utils.dataset import read_scannet_gray, read_scannet_depth


class ScanNetDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 intrinsic_path,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode

        # prepare data_names, intrinsics and extrinsics(T)
        with np.load(npz_path) as data:
            self.data_names = data['name']
            self.T_1to2s = data['rel_pose']
            # min_overlap_score criterion
            if 'score' in data.keys() and mode not in ['val' or 'test']:
                kept_mask = data['score'] > min_overlap_score
                self.data_names = self.data_names[kept_mask]
                self.T_1to2s = self.T_1to2s[kept_mask]
        self.intrinsics = dict(np.load(intrinsic_path))

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        # read the grayscale image which will be resized to (1, 480, 640)
        img_name0 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
        img_name1 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_1}.jpg')
        image0 = read_scannet_gray(img_name0, resize=(640, 480),
                                   augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1 = read_scannet_gray(img_name1, resize=(640, 480),
                                   augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read the depthmap which is stored as (480, 640)
        if self.mode in ['train', 'val']:
            depth0 = read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_0}.png'))
            depth1 = read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_1}.png'))
        else:
            depth0 = depth1 = torch.tensor([])

        # read the intrinsic of depthmap
        K_0 = K_1 = torch.tensor(self.intrinsics[scene_name].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T_0to1 = torch.tensor(self.T_1to2s[idx].copy(), dtype=torch.float).reshape(3, 4)
        T_0to1 = torch.cat([T_0to1, torch.tensor([[0., 0., 0., 1.]])], dim=0).reshape(4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,   # (1, h, w)
            'depth0': depth0,   # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'dataset_name': 'scannet',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (osp.join(scene_name, 'color', f'{stem_name_0}.jpg'),
                           osp.join(scene_name, 'color', f'{stem_name_1}.jpg'))
        }

        return data
