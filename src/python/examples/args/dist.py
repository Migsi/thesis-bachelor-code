import argparse
from typing import List


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Lightning TorchX Example")
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size to use for training"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        help="training strategy, 'fsdp' is available for gpu only, defaults to data parallel"
    )
    parser.add_argument("--pretrained", action="store_true", help="use pretrained model weights")
    parser.add_argument("--skip_export", action="store_true", help="do not export model as archive")
    parser.add_argument("--load_path", type=str, help="path to load checkpoints from")
    parser.add_argument(
        "--storage_path",
        type=str,
        required=True,
        help="path to store logs, checkpoints and archived models",
    )
    return parser.parse_args(argv)


def map_strategy_accelerator(strategy_name: str) -> str:
    if strategy_name == "fsdp":
        return "gpu"
    return "auto"  # Use either 'gpu', 'tpu', or 'cpu'
