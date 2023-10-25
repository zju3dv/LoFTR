import torch
import numpy as np

from src.loftr import LoFTR, default_cfg
import time

class LoFTR_Inference:
    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path
        self.matcher = self.init_model()

    def init_model(self,):
        """
        Init LoFTR model
        """
        # init model
        model = LoFTR(config=default_cfg)
        # load the pretrained model
        model.load_state_dict(torch.load(self.weights_path)['state_dict'])
        model = model.eval().cuda()
        return model

    @staticmethod
    def img2net_input(img: np.ndarray):
        """
        Convert image to network input
        """
        return torch.from_numpy(img).cuda() / 255.

    def run_matching(self, img0: torch.tensor, img1: torch.tensor):
        # inputs to matcher
        batch = {'image0': img0, 'image1': img1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            # measure time
            start_time = time.time()
            self.matcher(batch)
            end_time = time.time()
            print(f"running duration {(end_time - start_time):.2f} seconds")
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            m_bids = batch['m_bids'].cpu().numpy()
        return mkpts0, mkpts1, mconf, m_bids

    def predict(self, img0: np.ndarray, img1: np.ndarray):
        # Preprocess images
        img0_torch, img1_torch = self.img2net_input(img0), self.img2net_input(img1)

        # run matcher inference
        mkpts0, mkpts1, mconf, mbids = self.run_matching(img0_torch, img1_torch, self.matcher)

        return mkpts0, mkpts1, mconf, mbids

