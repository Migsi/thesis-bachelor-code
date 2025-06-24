import os
import sys
import tempfile
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from args.dist import parse_args, map_strategy_accelerator
from data.module import CIFAR10, resnet18transform
from model.resnet import ResNet18, export_inference_model
from torchvision.models import ResNet18_Weights


def main(argv: List[str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(argv)

        # Init Data and Model
        weights = None
        if args.pretrained:
            weights = ResNet18_Weights.DEFAULT
        data_module = CIFAR10(batch_size=args.batch_size, transform=resnet18transform(weights=weights))
        model = ResNet18(lr=args.lr, weights=weights)

        # Fetch world size
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Create subdirectory based on model and world size
        model_name = f"{str(model.__class__.__name__)}_ws{world_size}_{args.strategy}"
        args.storage_path = os.path.join(args.storage_path, model_name)

        # Setup DeviceStatsMonitor callback
        callbacks: list[Callback] = [DeviceStatsMonitor(cpu_stats=True)]

        # Setup model checkpointing (optional)
        if args.storage_path:
            # NOTE: It is important that each rank behaves the same.
            # All ranks, or none of them should return ModelCheckpoint
            # otherwise, there will be deadlock for distributed training
            callbacks.append(ModelCheckpoint(dirpath=f"{args.storage_path}/checkpoints"))
        if args.load_path:
            print(f"Loading checkpoint: {args.load_path}")
            model = ResNet18.load_from_checkpoint(checkpoint_path=args.load_path)

        # Setup logger (optional)
        logger = TensorBoardLogger(
            save_dir=args.storage_path,
            name="logs"
        )

        # Trainer config (can be set to accelerator='gpu' if needed)
        trainer = pl.Trainer(
            num_nodes=world_size,
            max_epochs=args.epochs,
            accelerator=map_strategy_accelerator(args.strategy),
            devices="auto",  # If GPU, use all available
            strategy=args.strategy,
            logger=logger,
            callbacks=callbacks,
            # default_root_dir=args.storage_path, # Alternative to dedicated logger and callbacks (optional)
            # precision=16-mixed, # Mixed precision for speed (optional)
        )

        # Train and Validate
        trainer.fit(model, datamodule=data_module)

        # Export the inference model
        rank = trainer.global_rank
        if rank == 0 and not args.skip_export and args.storage_path:
            storage_path = f"{args.storage_path}/export"
            print(f"Saving model to {storage_path}")
            export_inference_model(model, storage_path, tmpdir)


if __name__ == "__main__":
    main(sys.argv[1:])
