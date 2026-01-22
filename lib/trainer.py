import torch
import numpy as np
import pandas as pd
import lightning as L
import torch.optim as optim

from typing import Optional, Any
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from lib.utils import accuracy_metric
from lib.model.model import AgeEstimationModel
from lib.dataset import AdienceAgeEstimationDataset
from lib.loss import get_loss_function, ProbabilisticOrdinalLoss

# Lightning module for training age estimation model
class AgeEstimationLitModule(L.LightningModule):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

        # define model
        self.model = AgeEstimationModel(
            backbone_type=self.config['model']['backbone_type'],
            backbone_imagenet_pretrained=self.config['model']['backbone_imagenet_pretrained'],
            head_type=self.config['model']['head_type'],
            n_age_classes=self.config['data']['n_age_classes'],
            use_poe=self.config['model']['use_poe'],
            t_samples=self.config['training']['poe_config']['t_monte_carlo_samples'] if self.config['model']['use_poe'] else None,
        )

        # define loss function
        self.loss_fn = ProbabilisticOrdinalLoss(
            head_type=self.config['model']['head_type'],
            distance_name=self.config['training']['poe_config']['distance_metric'],
            alpha=self.config['training']['poe_config']['alpha'],
            beta=self.config['training']['poe_config']['beta'],
            delta=self.config['training']['poe_config']['delta'],
        ) if self.config['model']['use_poe'] else get_loss_function(self.config['model']['head_type'])

        self.save_hyperparameters()

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        inputs, targets = batch
        logits, ages, poe = self.model(inputs)

        if self.config['model']['use_poe']:
            assert isinstance(self.loss_fn, ProbabilisticOrdinalLoss), 'Loss function must be ProbabilisticOrdinalLoss when using POE model!'
            
            embeddings_mean, embeddings_log_var = poe
            head_loss, vib_loss, ordinal_loss, loss = self.loss_fn(logits, embeddings_mean, embeddings_log_var, targets)

            self.log('train/partial_head_loss', head_loss.item(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
            self.log('train/partial_vib_loss', vib_loss.item(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
            self.log('train/partial_ord_loss', ordinal_loss.item(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        else:
            loss = self.loss_fn(logits, targets)

        acc = accuracy_metric(ages, targets)

        self.log('train/loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log('train/accuracy', acc, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        inputs, targets = batch
        logits, ages, poe = self.model(inputs)

        if self.config['model']['use_poe']:
            assert isinstance(self.loss_fn, ProbabilisticOrdinalLoss), 'Loss function must be ProbabilisticOrdinalLoss when using POE model!'
            
            embeddings_mean, embeddings_log_var = poe
            head_loss, vib_loss, ordinal_loss, loss = self.loss_fn(logits, embeddings_mean, embeddings_log_var, targets)

            self.log('val/partial_head_loss', head_loss.item(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
            self.log('val/partial_vib_loss', vib_loss.item(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
            self.log('val/partial_ord_loss', ordinal_loss.item(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        else:
            loss = self.loss_fn(logits, targets)

        acc = accuracy_metric(ages, targets)

        self.log('val/loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log('val/accuracy', acc, on_epoch=True, on_step=False, prog_bar=False, logger=True)

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        loss = self.model(batch).sum()
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.config['training']['learning_rate'],
        )
    
        lr_decay_epochs = [int(i) for i in self.config['training']['learning_rate_decay_epoch']] + [int(np.inf)]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            gamma=self.config['training']['learning_rate_decay'],
            milestones=lr_decay_epochs,
            last_epoch=-1,
        )
    
        return [optimizer], [lr_scheduler]
    
# Lightning data module for Adience dataset
class AdienceAgeEstimationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root_dir: str,
        annotations_filepath: str,
        batch_size: int,
        n_dataloader_workers: int,
        train_ratio: float,
        random_seed: Optional[int] = None,
    ):
        super().__init__()

        self.data_root_dir = data_root_dir
        self.annotations_filepath = annotations_filepath

        self.batch_size = batch_size
        self.n_dataloader_workers = n_dataloader_workers
        self.train_ratio = train_ratio
        self.random_seed = random_seed

        self.transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_eval = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.train_ds: Optional[AdienceAgeEstimationDataset] = None
        self.val_ds: Optional[AdienceAgeEstimationDataset] = None
        self.test_ds: Optional[AdienceAgeEstimationDataset] = None

        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == 'fit':
            annotations = pd.read_csv(self.annotations_filepath).to_dict(orient='records')
            train_annotations, val_annotations = train_test_split(
                annotations,
                train_size=self.train_ratio,
                random_state=self.random_seed,
                shuffle=True,
            )
            self.train_ds = AdienceAgeEstimationDataset(
                self.data_root_dir,
                train_annotations,
                self.transform_train,
            )
            self.val_ds = AdienceAgeEstimationDataset(
                self.data_root_dir,
                val_annotations,
                self.transform_eval,
            )

    def train_dataloader(self):
        if self.train_ds is None:
            raise RuntimeError('Training dataset not set up')
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_dataloader_workers,
            persistent_workers=self.n_dataloader_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            raise RuntimeError('Validation dataset not set up.')
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_dataloader_workers,
            persistent_workers=self.n_dataloader_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            raise RuntimeError('Test dataset not set up.')
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_dataloader_workers,
            persistent_workers=self.n_dataloader_workers > 0,
            pin_memory=True,
        )