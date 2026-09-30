"""Stimulus dataset for the MEG-occlusion experiment (4 categories x 3 occlusion levels)."""
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

OCCLUSION_MAPPING = {'level_00': 0, 'level_60': 1, 'level_80': 2}
CLASS_MAPPING = {'camel': 0, 'car': 1, 'deer': 2, 'motor': 3}


class OcclusionDataset(Dataset):
    """Images laid out as ``root/<category>/level_<occlusion>/*.png``.

    Each item is a dict with the transformed image, the integer class and occlusion labels
    (see ``CLASS_MAPPING`` / ``OCCLUSION_MAPPING``), their names, and the file path.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """Collect image paths and labels from the category and occlusion-level folders."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name not in CLASS_MAPPING:
                continue
            for level_dir in sorted(class_dir.iterdir()):
                if not level_dir.is_dir() or level_dir.name not in OCCLUSION_MAPPING:
                    continue
                for img_path in sorted(level_dir.glob('*.png')):
                    self.samples.append({
                        'path': str(img_path),
                        'class': CLASS_MAPPING[class_dir.name],
                        'class_name': class_dir.name,
                        'occlusion': OCCLUSION_MAPPING[level_dir.name],
                        'occlusion_name': level_dir.name,
                    })

        if not self.samples:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'class': sample['class'],
            'class_name': sample['class_name'],
            'occlusion': sample['occlusion'],
            'occlusion_name': sample['occlusion_name'],
            'path': sample['path'],
        }


def imagenet_transform():
    """Resize(256) -> CenterCrop(224) -> ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_dataloader(root_dir, transform, batch_size=32, num_workers=4):
    """Return an unshuffled ``(DataLoader, OcclusionDataset)`` pair over ``root_dir``."""
    dataset = OcclusionDataset(root_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataloader, dataset


def batch_metadata(batch):
    """Label columns of a dataloader batch as a DataFrame (one row per image)."""
    return pd.DataFrame({
        'occlusion': batch['occlusion'].numpy(),
        'occlusion_name': batch['occlusion_name'],
        'class': batch['class'].numpy(),
        'class_name': batch['class_name'],
    })
