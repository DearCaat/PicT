from .base import BaseTrainer
from .iNet_cls import *
from .pict import *


def build_trainer(config):
    if config.TRAINER.NAME.lower() == 'pict':
        engine = PicTEngine(config)
    elif config.TRAINER.NAME.lower() == 'inet_cls':
        engine = INetClsEngine(config)
    else:
        raise NotImplementedError
    base = BaseTrainer(engine=engine,config=config)
    return base.train_one_epoch,base.predict,base.validate,base.best_metrics