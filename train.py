import argparse
import math
from tqdm import tqdm
import numpy as np
import os
import os.path as osp

import torch
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.cuda import amp

from model import build_unet3plus, UNet3Plus
from model.unet3plusoriginal import UNet_3Plus_DeepSup
from torch.utils.data import DataLoader
from datasets import build_data_loader
from config.config import cfg
from utils.loss import build_u3p_loss
from utils.logging import AverageMeter, SummaryLogger
from utils.metrics import StreamSegMetrics

def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

class Trainer:

    global_iter = 0
    start_epoch = 0
    epoch = 0   # current epoch
    loss_dict = dict()
    val_loss_dict = dict()
    val_score_dict = None
    best_val_score_dict = None
    
    def __init__(self, cfg, model, train_loader, val_loader):
        self.cfg_all = cfg

        # build metrics
        self.metrics = StreamSegMetrics(cfg.data.num_classes)

        os.makedirs(cfg.train.save_dir, exist_ok=True)
        hyp_path = osp.join(cfg.train.save_dir, cfg.train.save_name+'.yaml')
        with open(hyp_path, "w") as f: 
            f.write(cfg.dump())

        cfg = self.cfg = cfg.train
        self.model: UNet3Plus = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader

        # build loss
        self.criterion = build_u3p_loss(cfg.loss_type, cfg.aux_weight)
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

        self.logger = SummaryLogger(self.cfg_all)

        self.model.to(cfg.device)
        if cfg.resume:
            self.resume(cfg.resume)
        
        
    def resume(self, resume_path):
        print('resuming from {}'.format(resume_path))
        saved = torch.load(resume_path, map_location=self.cfg.device)
        self.model.load_state_dict(saved['state_dict'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.scheduler.load_state_dict(saved['scheduler'])
        self.scheduler.step()
        self.epoch = saved['epoch'] + 1
        self.start_epoch = saved['epoch'] + 1
        self.global_iter = saved['global_iter']

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.train_one_epoch()
            self.end_train_epoch()

    def train_one_epoch(self):
        model = self.model
        model.train()
        device = self.cfg.device
        pbar = enumerate(self.train_loader)
        num_batches = len(self.train_loader)
        batch_size = self.train_loader.batch_size
        accum_steps = self.cfg.accum_steps
        
        pbar = tqdm(pbar, total=num_batches, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        # with torch.profiler.profile(
        #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler(self.cfg.logger.log_dir),
        #         record_shapes=True
        # ) as prof:
        for i, batch in pbar:
            self.warmup()
            imgs, masks = batch[0].to(device), batch[1].to(device, dtype=torch.long)
            self.global_iter += batch_size
            with amp.autocast():
                preds = model(imgs)
                loss, batch_loss_dict = self.criterion(preds, masks)
            self.update_loss_dict(self.loss_dict, batch_loss_dict)
            self.scaler.scale(loss).backward()
            if (i+1) % accum_steps == 0 or i == num_batches - 1:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                    # prof.step()
        # self.scheduler.step()
        pbar.close()

    def end_train_epoch(self):
        self.epoch += 1
        if self.epoch % self.cfg.val_interval == 0 or self.epoch == self.cfg.epochs:
            val_dict = self.val_score_dict = self.validate()
            miou = val_dict['Mean IoU']
            if self.best_val_score_dict is None or miou > self.best_val_score_dict['Mean IoU']:
                self.best_val_score_dict = val_dict
                self.save_checkpoint(self.cfg.save_name + '_best.ckpt')
            self.log_results()
        self.save_checkpoint(self.cfg.save_name + '_last.ckpt')
        self.scheduler.step()
    
    def save_checkpoint(self, save_name):
        state = {
            'epoch': self.epoch,
            'global_iter': self.global_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(state, osp.join(self.cfg.save_dir, save_name))

    def warmup(self):
        ni = self.global_iter
        # if ni <= self.cfg.warmup_iters:
        #     xi = [0, self.cfg.warmup_iters]  # x interp
        #     for j, x in enumerate(self.optimizer.param_groups):
        #         x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lr_func(self.epoch)])
        #         if 'momentum' in x:
        #             x['momentum'] = np.interp(ni, xi, [0.8, self.cfg.momentum])

        warmup_iters = max(self.cfg.warmup_iters, len(self.train_loader.dataset) * 3)
        if ni <= warmup_iters:
            xi = [0, warmup_iters]  # x interp
            for j, x in enumerate(self.optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lr_func(self.epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.8, self.cfg.momentum])

    def update_loss_dict(self, loss_dict, batch_loss_dict=None):
        if batch_loss_dict is None:
            if loss_dict is None:
                return
            for k in loss_dict:
                loss_dict[k].reset()
        elif len(loss_dict) == 0:
            for k, v in batch_loss_dict.items():
                loss_dict[k] = AverageMeter(val=v)
        else:
            for k, v in batch_loss_dict.items():
                loss_dict[k].update(v)

    def log_results(self):
        log_dict = {
            'Train': {},
            'Val': {}
        }

        for k, v in self.loss_dict.items():
            log_dict['Train'][k] = v.avg
        self.update_loss_dict(self.loss_dict, None)
        log_dict['Train']['lr'] = self.optimizer.param_groups[0]['lr']

        for k, v in self.val_loss_dict.items():
            log_dict['Val'][k] = v.avg
        self.update_loss_dict(self.val_loss_dict, None)
        
        for k, v in self.val_score_dict.items():
            if k == 'Class IoU':
                print(v)
                # self.logger.cmd_logger.info(v)
                continue
            log_dict['Val'][k] = v
        self.logger.summary(log_dict, self.global_iter)


    def validate(self):
        """Do validation and return specified samples"""
        self.metrics.reset()
        self.model.eval()
        device = self.cfg.device
        pbar = enumerate(self.val_loader)
        pbar = tqdm(pbar, total=len(self.val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        with torch.no_grad():

            for i, (images, labels) in pbar:

                images = images.to(device)
                labels = labels.to(device, dtype=torch.long)

                outputs = self.model(images)
                
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                self.metrics.update(targets, preds)

                _, batch_loss_dict = self.criterion(outputs, labels)
                self.update_loss_dict(self.val_loss_dict, batch_loss_dict)

            score = self.metrics.get_results()
            pbar.close()
        return score

def main(args):

    cfg.merge_from_file(args.cfg)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
    if args.resume:
        cfg.train.resume = args.resume
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.use_tensorboard is not None:
        cfg.train.logger.use_tensorboard = args.use_tensorboard == 1
    elif args.use_wandb is not None:
        cfg.train.logger.use_wandb = args.use_wandb == 1
    cfg.freeze()
    print(cfg)

    import torch
    import random
    import numpy as np
    seed = 42
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model, data = cfg.model, cfg.data
    model = build_unet3plus(data.num_classes, model.encoder, model.skip_ch, model.aux_losses, model.use_cgm, model.pretrained)
    # model = UNet_3Plus_DeepSup()
    if data.type in ['voc2012', 'voc2012_aug']:
        train_loader, val_loader = build_data_loader(data.data_dir, data.batch_size, data.num_workers, data.max_training_samples)
    else:
        raise NotImplementedError
    
    trainer = Trainer(cfg, model, train_loader, val_loader)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="config/u3p_resnet34_voc.yaml",
                        type=str)
    parser.add_argument('--seed',
                        help='random seed',
                        default=None)
    parser.add_argument('--resume',
                        help='resume from checkpoint',
                        default=None,
                        type=str)
    parser.add_argument('--data_dir',
                        default=None,
                        type=str)
    parser.add_argument('--use_wandb',
                        default=None,
                        type=int)
    parser.add_argument('--use_tensorboard',
                        default=None,
                        type=int)

    args = parser.parse_args()
    main(args)