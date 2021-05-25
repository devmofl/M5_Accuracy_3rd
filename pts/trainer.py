import time
from typing import List, Optional, Tuple
import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pts.core.component import validated

import os
logger = logging.getLogger("mofl").getChild("trainer")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:    
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        num_workers: int = 0, # TODO worker>0 causes performace drop if uses iterable dataset
        pin_memory: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        betas: Tuple[float, float] = (0.9, 0.999),
        device: Optional[torch.device] = None,
        log_path: Optional[str] = None,
        use_lr_scheduler: bool = False,
        lr_warmup_period: int = 0,  # num of iterations for warmup
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.log_path = log_path
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_warmup_period = lr_warmup_period
        
        self.roll_mat_csr = None
        
    def __call__(
        self, net: nn.Module, input_names: List[str], training_data_loader: DataLoader, validation_period: int = 1
    ) -> None:
        # loggin model size
        net_name = type(net).__name__
        num_model_param = count_parameters(net)
        logger.info(
            f"Number of parameters in {net_name}: {num_model_param}"
        )

        if torch.cuda.device_count() > 1:
            logger.info("Training with %d gpus..." % (torch.cuda.device_count()))
            net = nn.DataParallel(net)

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas=self.betas, #eps=1e-9,
        )

        if self.use_lr_scheduler:
            total_iter = self.epochs * self.num_batches_per_epoch
            scheduler_cos = CosineAnnealingLR(optimizer, total_iter, eta_min=1e-6)        
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.lr_warmup_period, after_scheduler=scheduler_cos)

        writer = SummaryWriter(log_dir=self.log_path)
        #writer.add_graph(net)

        def loop(
            epoch_no, data_loader, is_training: bool = True
        ) -> float:
            tic = time.time()
            avg_epoch_loss = 0.0
            cumlated_sqerr = 0.0
            total_seq = 0
            errors = []

            if is_training:
                net.train()
            else:
                net.eval()

            with tqdm(data_loader, total=float("inf"),disable=os.environ.get("DISABLE_TQDM", False)) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    inputs = [data_entry[k].to(self.device) for k in input_names]

                    output = net(*inputs)
                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                        error = output[1]
                        cumlated_sqerr += (error ** 2).sum()
                        total_seq += len(inputs[0])

                        if not is_training:
                            errors.append(error)
                    else:
                        loss = output

                    loss = loss.sum()

                    avg_epoch_loss += loss.item()

                    lr = optimizer.param_groups[0]['lr']

                    it.set_postfix(
                        ordered_dict={
                            "lr": lr,
                            ("" if is_training else "validation_")
                            + "avg_epoch_loss": avg_epoch_loss / batch_no,
                            "epoch": epoch_no,                            
                        },
                        refresh=False,
                    )
                    n_iter = epoch_no*self.num_batches_per_epoch + batch_no

                    if n_iter % 20 == 0:
                        if is_training:
                            writer.add_scalar('Loss/train', loss.item(), n_iter)
                            writer.add_scalar('Learning rate', lr, n_iter)
                        else:
                            writer.add_scalar('Loss/validation', loss.item(), n_iter)

                    if is_training:
                        loss.backward()
                        optimizer.step()

                        if self.use_lr_scheduler:
                            scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        #for name, param in net.named_parameters():
                        #    writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)
                        break

            # mark epoch end time and log time cost of current epoch
            toc = time.time()

            # logging
            '''logger.info(
                "Epoch[%d] Elapsed time %.3f seconds",
                epoch_no,
                (toc - tic),
            )'''

            lv = avg_epoch_loss / batch_no
            logger.info(
                "Epoch[%d] Evaluation metric '%s'=%f",
                epoch_no,
                ("" if is_training else "validation_") + "epoch_loss",
                lv,
            )

            writer.add_scalar('Loss_epoch/' + ("train" if is_training else "validation") , lv, epoch_no)

            if total_seq != 0:
                writer.add_scalar('MSE_epoch/' + ("train" if is_training else "validation") , cumlated_sqerr / total_seq, epoch_no)
                
            return lv

        for epoch_no in range(self.epochs):            
            # training
            epoch_loss = loop(epoch_no, training_data_loader)

            if epoch_no % validation_period == 0 and epoch_no != 0:
                # save model
                torch.save(net.state_dict(), f"{self.log_path}/trained_model/train_net_{epoch_no}")
        
        writer.close()
