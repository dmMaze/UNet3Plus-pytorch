import argparse
import math
from tqdm import tqdm
import numpy as np

import torch
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp

from model import build_unet3plus, UNet3Plus
from torch.utils.data import DataLoader
from datasets import build_data_loader
from config.config import cfg
from utils.losses.losses import build_loss
from utils.log import AverageMeter

def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

class Trainer:

    global_iter = 0
    start_epoch = 0
    epoch = 0   # current epoch
    loss = AverageMeter()
    
    def __init__(self, cfg, model, train_loader, val_loader):
        self.cfg = cfg
        self.model: UNet3Plus = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader

        # build loss
        self.loss_func = build_loss(cfg.loss_type, cfg.aux_weight)
        self.scaler = amp.GradScaler(enabled=cfg.device == 'cuda')  # mixed precision training

        # build optimizer
        if cfg.optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum, nesterov=cfg.nesterov)
        elif cfg.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            raise ValueError('Unknown optimizer')
        if cfg.scheduler == 'linear':
            self.lr_func = lambda x: (1 - x / (cfg.epochs - 1)) * (1.0 - cfg.lrf) + cfg.lrf  # linear
        elif cfg.scheduler == 'cyclic':
            self.lr_func = one_cycle(1, cfg.lrf, cfg.epochs)
        else:
            raise ValueError('Unknown scheduler')

        # build scheduler
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_func)

        if cfg.logger.tensorboard:
            self.writer = SummaryWriter(log_dir=cfg.logger.log_dir)
        else:
            self.writer = None

        if cfg.resume:
            self.resume(cfg.resume)
        
        self.model.to(cfg.device)
        
    def resume(self, resume_path):
        print('resuming from {}'.format(resume_path))
        saved = torch.load(resume_path, map_location='cpu')
        self.model.load_state_dict(saved['state_dict'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.scheduler.load_state_dict(saved['scheduler'])
        self.scheduler.step()
        self.epoch = saved['epoch'] + 1
        self.start_epoch = saved['epoch'] + 1
        self.global_iter = saved['global_iter']

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.train_epoch(epoch)


    def train_epoch(self, epoch):
        model = self.model
        device = self.cfg.device
        pbar = enumerate(self.train_loader)
        num_batches = len(self.train_loader)
        batch_size = self.train_loader.batch_size
        accum_steps = self.cfg.accum_steps

        self.loss.reset()

        pbar = tqdm(pbar, total=num_batches, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        
        for i, batch in pbar:
            self.warmup()

            imgs, masks = batch[0].to(device), batch[1].to(device, dtype=torch.long)
            print(masks.shape)
            self.global_iter += batch_size
            with amp.autocast():
                preds = model(imgs)
            
            loss = self.loss_func(preds, masks)
            self.loss.update(loss.detach().item())
            self.scaler.scale(loss / accum_steps).backward()
            if (i+1) % accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        self.epoch += 1

    def warmup(self):
        ni = self.global_iter
        if ni <= self.cfg.warmup_iters:
            xi = [0, self.cfg.warmup_iters]  # x interp
            for j, x in enumerate(self.optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lr_func(self.epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.8, self.cfg.momentum])


def main(args):

    cfg.merge_from_file(args.cfg)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
    if args.resume:
        cfg.train.resume = args.resume
    cfg.freeze()
    print(cfg)
    model, data = cfg.model, cfg.data
    model = build_unet3plus(data.num_classes, model.encoder, model.skip_ch, model.aux_losses, model.use_cgm, model.pretrained)
    
    if data.type in ['voc2012', 'voc2012_aug']:
        train_loader, val_loader = build_data_loader(data.batch_size, data.num_workers, data.max_training_samples)
    else:
        raise NotImplementedError
    
    trainer = Trainer(cfg.train, model, train_loader, val_loader)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="config/test_voc.yaml",
                        type=str)
    parser.add_argument('--seed',
                        help='random seed',
                        default=None)
    parser.add_argument('--resume',
                        help='resume from checkpoint',
                        default='',
                        type=str)

    args = parser.parse_args()
    main(args)