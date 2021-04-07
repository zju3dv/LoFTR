from loguru import logger
from yacs.config import CfgNode as CN
from itertools import chain


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def log_on(condition, message, level):
    if condition:
        assert level in ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
        logger.log(level, message)


def flattenList(x):
    return list(chain(*x))


if __name__ == '__main__':
    _CN = CN()
    _CN.A = CN()
    _CN.A.AA = CN()
    _CN.A.AA.AAA = CN()
    _CN.A.AA.AAA.AAAA = "AAAAA"

    _CN.B = CN()
    _CN.B.BB = CN()
    _CN.B.BB.BBB = CN()
    _CN.B.BB.BBB.BBBB = "BBBBB"

    print(lower_config(_CN))
    print(lower_config(_CN.A))
