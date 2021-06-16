from configs.data.base import cfg


TRAIN_BASE_PATH = "data/scannet/index"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "ScanNet"
cfg.DATASET.TRAIN_DATA_ROOT = "data/scannet/train"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_data/train"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/scene_data/train_list/scannet_all.txt"
cfg.DATASET.TRAIN_INTRINSIC_PATH = f"{TRAIN_BASE_PATH}/intrinsics.npz"

TEST_BASE_PATH = "assets/scannet_test_1500"
cfg.DATASET.TEST_DATA_SOURCE = "ScanNet"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/scannet/test"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = TEST_BASE_PATH
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/scannet_test.txt"
cfg.DATASET.VAL_INTRINSIC_PATH = cfg.DATASET.TEST_INTRINSIC_PATH = f"{TEST_BASE_PATH}/intrinsics.npz"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val
