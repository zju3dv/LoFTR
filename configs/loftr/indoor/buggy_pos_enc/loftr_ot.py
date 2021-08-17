from src.config.default import _CN as cfg

cfg.LOFTR.COARSE.TEMP_BUG_FIX = False
cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'sinkhorn'

cfg.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 17, 20, 23, 26, 29]
