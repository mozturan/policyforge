from torch.utils.data import Dataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
from policyforge.train import TrainingConfig

def make_dataloaders(cfg: TrainingConfig):
    """Wraps LeRobot dataset into train/val DataLoaders."""
    dataset = LeRobotDataset(cfg.dataset_name)
 
    n_train = int(len(dataset) * cfg.train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
    )
    return train_loader, val_loader

