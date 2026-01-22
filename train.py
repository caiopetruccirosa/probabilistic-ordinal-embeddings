import os
import torch
import argparse
import numpy as np

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar

from lib.utils import yaml_load, yaml_dump, get_available_device
from lib.trainer import AgeEstimationLitModule, AdienceAgeEstimationDataModule

_EXPERIMENT_LIT_CHECKPOINT_FILENAME_TEMPLATE = 'epoch:{epoch}-val_loss:{val_loss:.5f}'

if __name__ == '__main__':
    # define CLI arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--experiment-filepath', type=str, required=True,
                            help='Experiment configuration file (e.g. configs/baseline.yaml).')
    arg_parser.add_argument('--dataset-data-root-dir', type=str, required=True,
                            help='Root directory of the dataset images.')
    arg_parser.add_argument('--dataset-annotations-filepath', type=str, required=True,
                            help='Dataset annotations CSV file (e.g. fold_0_data.txt).')
    arg_parser.add_argument('--device-type', type=str, choices=['cpu', 'gpu', 'mps'], required=False,
                            help='Type of device to use (e.g. "cpu", "gpu", or "mps").')
    arg_parser.add_argument('--n-dataloader-workers', type=int, default=4, required=False,
                            help='Number of worker processes per dataloader.')
    arg_parser.add_argument('--wandb-project', type=str, default='facial-age-estimation-benchmark', required=False,
                            help='Weights & Biases project name for logging.')
    arg_parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline', 'disabled'], default='online', required=False,
                            help='Weights & Biases run mode: "online" to sync, "offline" to log locally, "disabled" to turn off logging.')
    arg_parser.add_argument('--runs-dir', type=str, default='runs', required=False,
                            help='Directory where experiment runs are stored.')
    arg_parser.add_argument('--train-ratio', type=float, default=0.8, required=False,
                            help='Ratio of training samples when splitting the dataset into train and validation sets.')
    arg_parser.add_argument('--data-split-seed', type=int, default=0, required=False,
                            help='Random seed for dataset splitting into train and validation sets.')
    arg_parser.add_argument('--seed', type=int, default=0, required=False,
                            help='Random seed for reproducibility.')
    args = arg_parser.parse_args()

    # set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # load experiment configuration
    config = yaml_load(filepath=args.experiment_filepath)

    # create run directory and save config file
    experiment_name, _ = os.path.splitext(os.path.basename(args.experiment_filepath))
    run_dir = os.path.join(args.runs_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    yaml_dump(config, filepath=f'{run_dir}/config.yaml')

    # create lit module
    model_lit_module = AgeEstimationLitModule(config)

    # create lit data module
    data_lit_module = AdienceAgeEstimationDataModule(
        data_root_dir=args.dataset_data_root_dir,
        annotations_filepath=args.dataset_annotations_filepath,
        batch_size=config['training']['batch_size'],
        n_dataloader_workers=args.n_dataloader_workers,
        train_ratio=args.train_ratio,
        random_seed=args.data_split_seed,
    )

    # create wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        mode=args.wandb_mode,
        save_dir=run_dir,
        config=config,
    )

    # create checkpoint callbacks
    checkpoint_callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(run_dir, 'checkpoints', 'best'),
            filename=_EXPERIMENT_LIT_CHECKPOINT_FILENAME_TEMPLATE,
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            save_last='link',
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(run_dir, 'checkpoints', 'history'),
            filename=_EXPERIMENT_LIT_CHECKPOINT_FILENAME_TEMPLATE,
            save_last='link',
            save_on_exception=True,
            auto_insert_metric_name=False,
        ),
        RichModelSummary(max_depth=3),
        RichProgressBar(),
    ]
    
    # create trainer
    trainer = Trainer(
        default_root_dir=run_dir,
        max_epochs=config['training']['n_epochs'],
        logger=wandb_logger,
        callbacks=checkpoint_callbacks,
        accelerator=args.device_type or get_available_device(),
        deterministic=True,
    )

    trainer.fit(model_lit_module, datamodule=data_lit_module)