import logging
import os
import os.path as osp

def set_logging(name=None, verbose=True):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Sets level and returns logger
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val=None):
        self.reset()
        if val is not None:
            self.update(val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

class SummaryLogger:

    def __init__(self, cfg_all) -> None:
        cfg = cfg_all.train.logger
        self.use_wandb = cfg.use_wandb
        self.use_tensorboard = cfg.use_tensorboard
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            save_dir = osp.join(cfg.log_dir, cfg_all.train.save_name)
            self.writer = SummaryWriter(save_dir)
        elif self.use_wandb:
            import wandb
            run_id = cfg.wandb.run_id
            if run_id:
                resume = 'must'
            else:
                resume = 'allow'
                run_id = None
            self.wandb = wandb.init(project=cfg.wandb.project, config=cfg_all, resume=resume, id=run_id)
        self.cmd_logger = LOGGER

    def summary(self, log_dict, global_iter):
        if self.use_wandb:
            wandb_dict = {}
            for tag, metrics in log_dict.items():
                for name, metric in metrics.items():
                    wandb_dict[tag + '/' + name] = metric
            self.wandb.log(wandb_dict)
        
        elif self.use_tensorboard:
            for tag, metrics in log_dict.items():
                for name, metric in metrics.items():
                    self.writer.add_scalars(tag + '_metrics/' + name, {tag: metric}, global_iter)
            self.writer.flush()

    def info(self, *args, **kwargs):
        return self.cmd_logger.info(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.cmd_logger.error(*args, **kwargs)

    def warn(self, *args, **kwargs):
        return self.cmd_logger.warn(*args, **kwargs)