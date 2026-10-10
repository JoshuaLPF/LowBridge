from argparse import ArgumentParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def build_train_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train RecNet.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--decay-rate", type=float, default=0.1)
    parser.add_argument("--decay-epoch", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--train-images", type=Path,
                        default=PROJECT_ROOT / "train_datasets" / "MR" / "Edge")
    parser.add_argument("--train-targets", type=Path,
                        default=PROJECT_ROOT / "train_datasets" / "MR" / "Img")
    parser.add_argument("--val-images", type=Path,
                        default=PROJECT_ROOT / "train_datasets" / "MR" / "val" / "Edge")
    parser.add_argument("--val-targets", type=Path,
                        default=PROJECT_ROOT / "train_datasets" / "MR" / "val" / "Img")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "ckpts")
    parser.add_argument("--no-pretrained-backbone", action="store_false",
                        dest="pretrained_backbone")
    parser.set_defaults(pretrained_backbone=True)
    return parser


def build_test_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run RecNet inference on a test set.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--test-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "test_maps")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--gpu-id", default="0")
    return parser
