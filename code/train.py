import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data_utils import RASampler, load_dataset
from model import get_model
from optimizer import Muon

warnings.filterwarnings('ignore')


def set_logger(model_name):
    # 获取一个日志记录器实例，通常以模块名命名
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # 为日志文件创建目录（如果不存在）
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    # 使用当前时间戳命名日志文件，避免覆盖
    log_filename = f"{model_name}_{datetime.now().strftime('app_%Y%m%d_%H%M%S.log')}"
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件中记录所有 DEBUG 及以上级别>的消息
    logger.addHandler(file_handler)
    return logger, file_handler


def accelerate_train(model, model_name, train_dataset, val_dataset, output_path, fold=None,
                     ema_model=None):  # Removed 'device' parameter

    logger, file_handler = set_logger(model_name)
    accelerator = Accelerator(mixed_precision="fp16")  # Added for mixed precision training
    set_seed(42)
    if accelerator.is_main_process:
        if fold is not None:
            print(f"Fold {fold}")
            logger.info(f"Fold {fold}")
        print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
        logger.info(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    # mixup and cutmix
    cutmix = v2.CutMix(num_classes=54)
    mixup = v2.MixUp(alpha=0.2, num_classes=54)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    sampler = RASampler(train_dataset, seed=42, num_replicas=1)
    # sampler = None
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16,
                              sampler=sampler,
                              shuffle=sampler is None,
                              pin_memory=True,
                              num_workers=4,
                              prefetch_factor=2,
                              persistent_workers=True,
                              collate_fn=collate_fn
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            pin_memory=True,
                            num_workers=4,
                            prefetch_factor=2,
                            persistent_workers=True
                            )

    epochs = 30
    lr = 3e-5

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )
    # optimizer = optim.AdamW(model.parameters(), lr=lr)
    adamw_param = []
    muon_param = []
    for name, param in model.named_parameters():
        if (
                'token' in name or 'embed' in name or
                param.ndim != 2
        ):
            adamw_param.append(param)
        else:
            muon_param.append(param)
    optimizer = Muon(lr=lr, wd=0.01, muon_params=muon_param, adamw_params=adamw_param)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * epochs * len(train_loader)),
        num_training_steps=epochs * len(train_loader)
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    # Training loop
    best_val_acc = accelerate_evaluate(model, val_loader, accelerator)
    if accelerator.is_main_process:
        print(f'Zero-shot Val Acc: {best_val_acc:.4f}')
        logger.info(f'Zero-shot Val Acc: {best_val_acc:.4f}')
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_start = time.time()
        model.train()
        step = 0
        for (inputs, labels) in tqdm(train_loader,
                                     disable=not accelerator.is_main_process
                                     ):  # Disable tqdm for non-main processes
            # No need for inputs.to(device), labels.to(device) anymore!
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Use accelerator.backward for proper gradient handling in distributed training
            accelerator.backward(loss)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            scheduler.step()
            step += 1
            if ema_model and step % 32 == 0:
                ema_model.update_parameters(model)
                if epoch * len(train_loader) + step < int(0.05 * epochs * len(train_loader)):
                    # Reset ema buffer to keep copying weights during warmup period
                    ema_model.n_averaged.fill_(0)
        if ema_model:
            ema_val_acc = accelerate_evaluate(ema_model, val_loader, accelerator)
        else:
            val_acc = accelerate_evaluate(model, val_loader, accelerator)
        epoch_time = time.time() - epoch_start

        # Only print and log from the main process to avoid duplicate output
        if accelerator.is_main_process:
            os.makedirs(output_path, exist_ok=True)
            if ema_model:
                print(f'Epoch {epoch + 1}, EMA ACC: {ema_val_acc:.4f}, Time: {epoch_time:.2f}s')
                logger.info(
                    f'Epoch {epoch + 1}, EMA ACC: {ema_val_acc:.4f}, Time: {epoch_time:.2f}s')

                # Save the best model based on validation score
                if ema_val_acc > best_val_acc:
                    best_val_acc = ema_val_acc
                    if fold is None:
                        torch.save(ema_model.state_dict(), f'{output_path}/best_model_ema.pth')
                    else:
                        torch.save(ema_model.state_dict(), f'{output_path}/best_model_ema_{fold}.pth')
                    print(f'New best EMA model saved with val acc: {ema_val_acc:.4f}')
                    logger.info(f'New best EMA model saved with val acc: {ema_val_acc:.4f}')
            else:
                print(f'Epoch {epoch + 1}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s')
                logger.info(
                    f'Epoch {epoch + 1}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Unwrap the model before saving state_dict
                    if fold is None:
                        torch.save(accelerator.unwrap_model(model).state_dict(),
                                  f'../model/{model_name}/best_model.pth')
                    else:
                        torch.save(accelerator.unwrap_model(model).state_dict(),
                                   f'../model/{model_name}/best_model_{fold}.pth')
                    print(f'New best model saved with val acc: {val_acc:.4f}')
                    logger.info(f'New best model saved with val acc: {val_acc:.4f}')
    # 结束 logger
    logger.removeHandler(file_handler)
    file_handler.close()


def accelerate_evaluate(model, data_loader, accelerator):
    model.eval()
    correct = 0
    total = 0

    # 使用 accelerator.unwrap_model 访问原始模型
    model = accelerator.unwrap_model(model)
    model.to(accelerator.device)

    # 不要手动 .to(device)，accelerator 已经帮你处理了
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, disable=not accelerator.is_main_process):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 使用 accelerator.gather 来聚合所有进程的结果（多GPU时）
    correct_tensor = torch.tensor(correct).to(accelerator.device)
    total_tensor = torch.tensor(total).to(accelerator.device)

    # 聚合全体 GPU 结果
    correct_all = accelerator.gather(correct_tensor).sum().item()
    total_all = accelerator.gather(total_tensor).sum().item()

    accuracy = correct_all / total_all if total_all > 0 else 0.0
    return accuracy


def main(ema=True, use_5fold=True):
    pretrained_model = "convnextv2_huge.fcmae_ft_in22k_in1k_384" # "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k" #'caformer_b36.sail_in22k_ft_in1k_384' #

    model_name = 'convnextv2_ls_gc_re_muon_mix_ra_ema_5folds'
    model_path = f'../model/{model_name}/best_model_ema.pth'
    # 初始化模型
    model, ema_model = get_model(pretrained_model, model_path, ema=ema)
    full_dataset, _ = load_dataset(model)

    if use_5fold:
        targets = np.array(full_dataset.targets)  # 获取标签列表，确保是 array

        # 初始化 5-fold 分层划分器
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset.samples, targets)):
            model_path = f'../model/{model_name}/best_model_ema_{fold}.pth'
            model, ema_model = get_model(pretrained_model, model_path, ema=ema)
            # 用 Subset 构建训练和验证数据集
            train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
            val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

            accelerate_train(model, model_name, train_dataset, val_dataset, f'../model/{model_name}', fold, ema_model, )
    else:
        # Partition training and validation sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        print("train_size = ", train_size)
        print("val_size = ", val_size)
        accelerate_train(model, model_name, train_dataset, val_dataset, f'../model/{model_name}', ema_model=ema_model, )


if __name__ == '__main__':
    main(use_5fold=True)
    pass
