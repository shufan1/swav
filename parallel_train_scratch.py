import os
import torch
import torch.cuda as cuda
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from ImageNetKaggle import ImageNetKaggle
from eval_linear import RegLog
import src.resnet50 as resnet_models
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import logging
from logging import getLogger
import os
import time
import wandb
import pickle
from src.logger import PD_Stats
from src.utils import (
    AverageMeter,
    accuracy,
)
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.datasets as datasets

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


def set_device(d):
    if d == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = d
    return device


def initialize_logger(dump_path, model_path, rank, *args):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=os.path.join(dump_path, f"train_{rank}.log"),
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    logging.info(
        f"Using pretrained ResNet50 from {model_path}, train on imagenet classifiaction, evaluate accuracy"
    )

    # dump parameters

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(dump_path, "stats" + str(rank) + ".pkl"), args
    )

    # create a logger

    logging.info("============ Initialized training logger ============")

    logging.info("The experiment will be stored in %s\n" % dump_path)
    logging.info("")
    return training_stats


def initialize_wandb(trainer, epochs, batch_size):
    # start wandb logging
    config_dict = {
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "weight_decay": trainer.optimizer.param_groups[0]["lr"],
        "epochs": epochs,
        "batch_size": batch_size,
    }

    wandb.init(project="multiprocess_scratch_imagnet", config=config_dict)
    wandb.watch(trainer.reglog, log="all")


def prepare_data(data_path):
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(os.path.join(data_path, "train"))
    train_dataset.transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            tr_normalize,
        ]
    )

    val_dataset = datasets.ImageFolder(os.path.join(data_path, "val"))
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            tr_normalize,
        ]
    )
    val_dataset.transform = val_transform

    return train_dataset, val_dataset


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        reglog: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        training_stats: PD_Stats,
        dumppath: str,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.reglog = reglog.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.dumppath = dumppath
        self.training_stats = training_stats
        self.reglog = torch.nn.parallel.DistributedDataParallel(
            self.reglog, device_ids=[gpu_id]
        )

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.reglog(self.model(source))
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        return loss, acc1, acc5

    def _run_epoch(self, epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        end = time.perf_counter()

        self.model.eval()
        self.reglog.train()

        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )

        for iter_epoch, (inp, targets) in tqdm(
            enumerate(self.train_data),
            total=len(self.train_data),
            desc=f"GPU {self.gpu_id} Epoch {epoch}",
        ):
            inp = inp.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, acc1, acc5 = self._run_batch(inp, targets)
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()
            # verbose
            if iter_epoch > 5:
                break
            if iter_epoch % 2 == 0:
                logging.info(
                    "Epoch[{0}] - Iter: [{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                    "LR {lr}".format(
                        epoch,
                        iter_epoch,
                        len(self.train_data),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1,
                        lr=self.optimizer.param_groups[0]["lr"],
                    )
                )
        # else:
        #     for iter_epoch, (inp, targets) in enumerate(self.train_data):
        #         inp = inp.to(self.gpu_id)
        #         targets = targets.to(self.gpu_id)
        #         loss, acc1, acc5 = self._run_batch(inp, targets)
        #         losses.update(loss.item(), inp.size(0))
        #         top1.update(acc1[0], inp.size(0))
        #         top5.update(acc5[0], inp.size(0))
        #         batch_time.update(time.perf_counter() - end)
        #         end = time.perf_counter()
        #         if iter_epoch > 3:
        #             return epoch, losses.avg, top1.avg.item(), top5.avg.item()
        return epoch, losses.avg, top1.avg.item(), top5.avg.item()

    def _eval_epoch(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        global best_acc

        self.model.eval()
        self.reglog.eval()

        with torch.no_grad():
            end = time.perf_counter()
            for i, (inp, targets) in enumerate(self.val_data):
                # move to gpu
                inp = inp.to(self.gpu_id)
                targets = targets.to(self.gpu_id)

                # compute output
                output = self.reglog(self.model(inp))
                loss = F.cross_entropy(output, targets)

                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                losses.update(loss.item(), inp.size(0))
                top1.update(acc1[0], inp.size(0))
                top5.update(acc5[0], inp.size(0))

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()

                if i > 3:
                    break
        if top1.avg.item() > best_acc:
            best_acc = top1.avg.item()

        logging.info(
            "GPU {gpu_id} \t"
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                gpu_id=self.gpu_id,
                batch_time=batch_time,
                loss=losses,
                top1=top1,
                acc=best_acc,
            )
        )

        return losses.avg, top1.avg.item(), top5.avg.item()

    def _save_checkpoint(self, epoch):
        ckp = self.reglog.module.state_dict()
        PATH = os.path.join(self.dumppath, f"checkpoint{epoch}.pt")
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        # training statistics
        global best_acc
        best_acc = 0

        for epoch in range(max_epochs):
            train_scores = self._run_epoch(epoch)
            val_scores = self._eval_epoch()
            self.training_stats.update(train_scores + val_scores)
            wandb.log(
                {
                    "train loss": train_scores[1],
                    "train accuracy": train_scores[2],
                    "validation loss": val_scores[0],
                    "validation accuracy": val_scores[1],
                }
            )

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

            # if self.gpu_id == 0:
            #     # only evaluate at first gpu
            #     val_scores = self._eval_epoch()
            #     self.training_stats.update(train_scores + val_scores)
            #     wandb.log(
            #         {
            #             "train loss": train_scores[1],
            #             "train accuracy": train_scores[2],
            #             "validation loss": val_scores[0],
            #             "validation accuracy": val_scores[1],
            #         }
            #     )

            #     if epoch % self.save_every == 0:
            #         self._save_checkpoint(epoch)
            # else:
            #     self.training_stats.update(train_scores)
            #     wandb.log(
            #         {"train loss": train_scores[1], "train accuracy": train_scores[2],}
            #     )

        logging.info(
            "Training of the supervised linear classifier on frozen features completed.\n"
            f"training loss: {train_scores[1]:.3f}, validation loss: {val_scores[0]:.3f}"
            f"Top-1 training accuracy: {train_scores[2]:.3f}, Top-1 validation accuracy: {val_scores[1]:.3f}"
        )
        # save final model
        if self.gpu_id == 0:
            torch.save(
                self.reglog.module.state_dict,
                os.path.join(self.dumppath, f"{epoch}final.pt"),
            )


def load_train_objs(data_path, model_path, lr, wd):
    train_dataset, val_dataset = prepare_data(data_path)
    print(
        "train dataset total size:",
        len(train_dataset),
        "validation dataset total size:",
        len(val_dataset),
    )
    state_dict = torch.load(model_path)
    model = resnet_models.__dict__["resnet50"](output_dim=0, eval_mode=True)
    msg = model.load_state_dict(state_dict, strict=False)
    logging.info(f"load pretrained ResNet50 weights : {msg}")
    reglog = RegLog(1000, global_avg=True, use_bn=False)
    # if args.linear_classifier_initial_weights != None:
    #     logging.info("linear classifier initial wieghts provided, loading ")
    #     state_dict = torch.load(args.linear_classifier_initial_weights)
    #     msg = reglog.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.SGD(
        reglog.parameters(), lr=lr, momentum=0.9, weight_decay=wd,
    )

    return train_dataset, val_dataset, model, reglog, optimizer


def main(
    rank: int,
    world_size: int,
    save_every: int,
    total_epochs: int,
    batch_size: int,
    data_path: str,
    dump_path: str,
    model_path: str,
    lr: float,
    wd: float,
):
    ddp_setup(rank, world_size)
    training_stats = initialize_logger(
        dump_path,
        model_path,
        rank,
        "epoch",
        "loss",
        "prec1",
        "prec5",
        "loss_val",
        "prec1_val",
        "prec5_val",
    )

    train_dataset, val_dataset, model, reglog, optimizer = load_train_objs(
        data_path, model_path, lr, wd
    )
    train_loader = prepare_dataloader(train_dataset, batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,)

    trainer = Trainer(
        model,
        reglog,
        train_loader,
        val_loader,
        optimizer,
        rank,
        save_every,
        training_stats,
        dump_path,
    )
    initialize_wandb(trainer, total_epochs, batch_size)

    trainer.train(total_epochs)

    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "--total_epochs", default=30, type=int, help="Total epochs to train the model"
    )
    parser.add_argument(
        "--save_every", default=10, type=int, help="How often to save a snapshot"
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    parser.add_argument(
        "--dump_path",
        type=str,
        default=".",
        help="experiment dump path for checkpoints and log",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/imagenet",
        help="path to dataset repository",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/resenetmodel",
        help="path to resnet50 weight file",
    )
    parser.add_argument("--lr", default=0.3, type=float, help="initial learning rate")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    args = parser.parse_args()

    torch.cuda.empty_cache()

    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(
            world_size,
            args.save_every,
            args.total_epochs,
            args.batch_size,
            args.data_path,
            args.dump_path,
            args.model_path,
            args.lr,
            args.wd,
        ),
        nprocs=world_size,
    )

