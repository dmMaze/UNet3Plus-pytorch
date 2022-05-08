
from yacs.config import CfgNode as CN

cfg = CN()

# MODEL
cfg.model = CN()
cfg.model.encoder = 'resnet18'
cfg.model.pretrained = False
cfg.model.skip_ch = 64
cfg.model.use_cgm = False
cfg.model.aux_losses = 2

# DATA
cfg.data = CN()
cfg.data.type = 'voc2012_aug'
cfg.data_root = './data'
cfg.data.num_classes = 21
cfg.data.batch_size = 1
cfg.data.num_workers = 0
cfg.data.max_training_samples = -1

# HYPERPARAMETERS
cfg.train = CN()
cfg.train.seed = 42
cfg.train.num_epochs = 10

cfg.train.lr = 0.001
cfg.train.lrf = 0.0005  # final lr
cfg.train.scheduler = 'linear'
cfg.train.warmup_iters = 500

cfg.train.optimizer = 'adamw'
cfg.train.weight_decay = 0.0001
cfg.train.momentum = 0.9
cfg.train.nesterov = True

cfg.train.accum_steps = 2
cfg.train.resume = ''
cfg.train.epochs = 120
cfg.train.eval_interval = 1
cfg.train.device = 'cuda'

cfg.train.aux_weight = 0.4
cfg.train.loss_type = 'focal'

# LOGGING
cfg.train.logger = CN()
cfg.train.logger.log_dir = './logs'
cfg.train.logger.tensorboard = True

