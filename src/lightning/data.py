from loguru import logger
from tqdm import tqdm
from os import path as osp

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler

from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split
from src.datasets.megadepth import MegaDepthDataset
from src.datasets.scannet import ScanNetDataset


class MultiSceneDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined 
    only a part of the training scenes to reduce memory overhead.
    """

    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH

        # 2. dataset config
        # general options
        self.min_overlap_score = config.DATASET.MIN_OVERLAP_SCORE  # 0.4, omit data with overlap_score < min_overlap_score
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

        # MegaDepth options
        self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
        self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD   # True
        self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD   # True
        self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
        self.coarse_scale = 1 / config.LOFTR.RESOLUTION[0]  # 0.125. for training loftr.

        # 3.loader parameters
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """ 
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage == 'test', "only support testing yet"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        self.test_dataset = self._setup_dataset(self.test_data_root,
                                                self.test_npz_root,
                                                self.test_list_path,
                                                self.test_intrinsic_path,
                                                mode='test')
        logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self, data_root, split_npz_root, scene_list_path, intri_path, mode='train'):
        """ Setup train / val / test set"""
        with open(scene_list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        if mode == 'train':
            local_npz_names = get_local_split(npz_names, self.world_size, self.rank, self.seed)
        else:
            local_npz_names = npz_names
        logger.info(f'[rank {self.rank}]: {len(local_npz_names)} scene(s) assigned.')

        return self._build_concat_dataset(data_root, local_npz_names, split_npz_root, intri_path, mode=mode)

    def _build_concat_dataset(self, data_root, npz_names, npz_dir, intrinsic_path, mode):
        datasets = []
        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
        for npz_name in tqdm(npz_names, desc=f'[rank:{self.rank}], loading {mode} datasets', disable=int(self.rank) != 0):
            # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
            npz_path = osp.join(npz_dir, npz_name)
            if data_source == 'ScanNet':
                datasets.append(
                    ScanNetDataset(data_root,
                                   npz_path,
                                   intrinsic_path,
                                   mode=mode,
                                   min_overlap_score=self.min_overlap_score,
                                   augment_fn=augment_fn))
            elif data_source == 'MegaDepth':
                datasets.append(
                    MegaDepthDataset(data_root,
                                     npz_path,
                                     mode=mode,
                                     min_overlap_score=self.min_overlap_score,
                                     img_resize=self.mgdpt_img_resize,
                                     df=self.mgdpt_df,
                                     img_padding=self.mgdpt_img_pad,
                                     depth_padding=self.mgdpt_depth_pad,
                                     augment_fn=augment_fn,
                                     coarse_scale=self.coarse_scale))
            else:
                raise NotImplementedError()
        return ConcatDataset(datasets)

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)
