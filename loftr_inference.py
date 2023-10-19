from experiments.matching import *
from experiments.preprocessing import *
from experiments.drawing import *

def run_inference(img0: np.ndarray, img1: np.ndarray, weights_path: str = None, is_indoor: bool = False, show_matches: bool = False):
    # initiate LoFTR model
    matcher = init_model(weights_path=weights_path, is_indoor=is_indoor)

    # Preprocess images
    img0_torch, img1_torch,  = img2net_input(img0), img2net_input(img1)

    # run matcher inference
    mkpts0, mkpts1, mconf = matching_by_loftr(img0_torch, img1_torch, matcher)

    if show_matches:
        draw_matches_on_images(img0, img1, mkpts0, mkpts1, mconf,
                               title='LoFTR Matcher - reference (left) to query (right)',
                               draw_epipolar_lines=DRAW_EPIPOLAR_LINES, f_matrix=None, conf_factor=CONF_FACTOR,
                               save_path=None)

    return mkpts0, mkpts1, mconf

