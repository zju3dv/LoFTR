import pprint
from loguru import logger
from pathlib import Path
import numpy as np

import torch
import pytorch_lightning as pl

from src.loftr import LoFTR
from src.utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors, aggregate_metrics

from src.utils.comm import gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):

        super().__init__()
        # Misc
        self.config = config  # full config
        self.loftr_cfg = lower_config(self.config.LOFTR)
        self.profiler = profiler or PassThroughProfiler()
        self.dump_dir = dump_dir

        # Matcher: LoFTR
        self.matcher = LoFTR(config=self.loftr_cfg)

        # Pretrained weights
        if pretrained_ckpt:
            self.matcher.load_state_dict(torch.load(pretrained_ckpt, map_location='cpu')['state_dict'])
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)
