from configs.data.base import cfg

TEST_BASE_PATH = "assets/scannet_test_1500"

cfg.DATASET.TEST_DATA_SOURCE = "ScanNet"
cfg.DATASET.TEST_DATA_ROOT = "data/scannet/test"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/scannet_test.txt"
cfg.DATASET.TEST_INTRINSIC_PATH = f"{TEST_BASE_PATH}/intrinsics.npz"

cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
