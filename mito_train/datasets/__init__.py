from .capture_dataset import CaptureDataset, build_transform
from .detector_dataset import DetectorDataset, build_detector_transform
from .pieces_dataset import PiecesDataset
from .piyo_dataset import PiyoDataset

__all__ = [
    "PiyoDataset", "PiecesDataset", "CaptureDataset", "build_transform",
    "DetectorDataset", "build_detector_transform",
    "HFCaptureDataset", "HFPairedDataset", "HFDetectorDataset",
]


def __getattr__(name):
    # Lazy import: HF datasets require the optional `datasets` package.
    if name == "HFCaptureDataset":
        from .hf_capture_dataset import HFCaptureDataset
        return HFCaptureDataset
    if name == "HFPairedDataset":
        from .hf_paired_dataset import HFPairedDataset
        return HFPairedDataset
    if name == "HFDetectorDataset":
        from .hf_detector_dataset import HFDetectorDataset
        return HFDetectorDataset
    raise AttributeError(f"module 'mito_train.datasets' has no attribute {name!r}")
