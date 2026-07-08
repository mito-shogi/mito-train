from .capture_dataset import CaptureDataset, build_transform
from .pieces_dataset import PiecesDataset
from .piyo_dataset import PiyoDataset

__all__ = [
    "PiyoDataset", "PiecesDataset", "CaptureDataset", "build_transform",
    "HFCaptureDataset",
]


def __getattr__(name):
    # Lazy import: HFCaptureDataset requires the optional `datasets` package.
    if name == "HFCaptureDataset":
        from .hf_capture_dataset import HFCaptureDataset
        return HFCaptureDataset
    raise AttributeError(f"module 'mito_train.datasets' has no attribute {name!r}")
