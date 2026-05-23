import math
import os
from PIL import Image
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
import torchvision
from torchvision.datasets import ImageFolder
import timm


class SafeImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        print("Validating image files...")

        valid_samples = []
        for path, label in self.samples:
            try:
                with Image.open(path) as img:
                    img.verify()  # 检查是否是完整图片
                valid_samples.append((path, label))
            except Exception as e:
                print(f"[Warning] Skipping corrupt image: {path} ({e})")

        self.samples = valid_samples
        self.imgs = valid_samples  # 兼容 torchvision 旧版本
        print(f"Remaining valid images: {len(self.samples)}")


class HerbalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.df = pd.read_csv(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path, label = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class RASampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=4):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for _ in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        if dist.get_world_size() > self.num_replicas:
            indices = indices[0: self.total_size: self.num_replicas]
        else:
            indices = indices[self.rank: self.total_size: self.num_replicas]

        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def load_dataset(model, train=True):
    # 数据预处理
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=True)
    transform = torchvision.transforms.Compose([transform, torchvision.transforms.RandomErasing(), ])

    dataset = SafeImageFolder(
        root='../data/train_set',
        transform=transform
    )

    real_class_to_idx = {v: k for k, v in
                         pd.read_csv('../data/chinese_herbal_medicine.csv').to_dict()['category'].items()}
    mapping = {v: real_class_to_idx[k] for k, v in dataset.class_to_idx.items()}
    if train:
        return dataset, mapping
    else:
        test_transform = timm.data.create_transform(**data_config, is_training=False)

        # 测试集评估
        test_dataset = HerbalDataset(
            df='../data/example_B.csv',
            img_dir='../data/test_set_B',
            transform=test_transform,
        )
        return test_dataset, mapping
