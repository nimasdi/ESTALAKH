import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMG_SIZE = 224

_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomAffine(degrees=4, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def _list_upright(split_dir: str | Path) -> list[Path]:
    d = Path(split_dir) / "0"
    return sorted(p for ext in _EXTS for p in d.glob(ext))


class RotationDataset(Dataset):
    def __init__(self, paths, transform=None, train: bool = True):
        self.paths = list(paths)
        self.transform = transform
        self.train = train

    def __len__(self) -> int:
        return len(self.paths) if self.train else len(self.paths) * 4

    def __getitem__(self, idx: int):
        if self.train:
            path = self.paths[idx]
            label = random.randint(0, 3)
        else:
            path = self.paths[idx // 4]
            label = idx % 4
        img = Image.open(path).convert("RGB")
        if label:
            img = img.rotate(-90 * label, expand=True)  # negative angle == CW
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def make_loaders(dataset_root, batch_size=32, val_fraction=0.15, num_workers=2):
    dataset_root = Path(dataset_root)

    train_paths = _list_upright(dataset_root / "train")
    if not train_paths:
        raise FileNotFoundError(
            f"no upright images under {dataset_root / 'train' / '0'} "
            f"(expected the *_r0 crops)"
        )

    paths = train_paths[:]
    random.Random(42).shuffle(paths)
    n_val = int(len(paths) * val_fraction)
    val_paths, fit_paths = paths[:n_val], paths[n_val:]

    train_ds = RotationDataset(fit_paths, transform=TRAIN_TRANSFORMS, train=True)
    val_ds = RotationDataset(val_paths, transform=EVAL_TRANSFORMS, train=False)
    test_ds = RotationDataset(
        _list_upright(dataset_root / "test"), transform=EVAL_TRANSFORMS, train=False
    )

    pin = torch.cuda.is_available()

    def loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin,
        )

    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)
