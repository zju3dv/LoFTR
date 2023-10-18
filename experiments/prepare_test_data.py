import os
import numpy as np


if __name__ == '__main__':
    # load npz file
    npz_path = '/home/avichaih/Projects/MyLoFTR/assets/megadepth_test_1500_scene_info/0015_0.1_0.3.npz'
    npz = np.load(npz_path, allow_pickle=True)
    print(npz)