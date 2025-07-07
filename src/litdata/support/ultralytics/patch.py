# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from litdata.constants import _ULTRALYTICS_AVAILABLE
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import StreamingDataset


def patch_ultralytics() -> None:
    """Patch Ultralytics to use the LitData optimize function."""
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics is not available. Please install it to use this functionality.")

    # import sys

    # if "ultralytics" in sys.modules:
    #     raise RuntimeError("patch_ultralytics() must be called before importing 'ultralytics'")

    # Patch detection dataset loading
    from ultralytics.data.utils import check_det_dataset

    check_det_dataset.__code__ = patch_check_det_dataset.__code__

    # Patch training visualizer (optional, but useful)
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    DetectionTrainer.plot_training_samples = patch_detection_plot_training_samples
    DetectionTrainer.plot_training_labels = patch_none_function

    # Patch BaseDataset globally
    import ultralytics.data.base as base_module
    import ultralytics.data.dataset as child_modules

    base_module.BaseDataset = PatchedUltralyticsBaseDataset
    base_module.BaseDataset.set_rectangle = patch_none_function
    child_modules.YOLODataset.__bases__ = (PatchedUltralyticsBaseDataset,)
    child_modules.YOLODataset.get_labels = patch_get_labels

    from ultralytics.data.build import build_dataloader

    build_dataloader.__code__ = patch_build_dataloader.__code__

    print("✅ Ultralytics successfully patched to use LitData.")


def patch_check_det_dataset(dataset: str, _: bool = True) -> Dict:
    if not (isinstance(dataset, str) and dataset.endswith(".yaml") and os.path.isfile(dataset)):
        raise ValueError("Dataset must be a string ending with '.yaml' and point to a valid file.")

    import yaml

    # read the yaml file
    with open(dataset) as file:
        data = yaml.safe_load(file)
    print(f"patch successful for {dataset}")
    return data


def patch_build_dataloader(
    dataset: Any, batch: int, workers: int, shuffle: bool = True, rank: int = -1, drop_last: bool = False
) -> StreamingDataLoader:
    """Create and return an InfiniteDataLoader or DataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker threads for loading data.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.

    Returns:
        (StreamingDataLoader): A dataloader that can be used for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    from litdata.streaming.dataloader import StreamingDataLoader

    print("litdata is rocking⚡️")
    if not hasattr(dataset, "streaming_dataset"):
        raise ValueError("The dataset must have a 'streaming_dataset' attribute.")

    from ultralytics.data.utils import PIN_MEMORY

    batch = min(batch, len(dataset))
    num_devices = torch.cuda.device_count()  # number of CUDA devices
    num_workers = min(os.cpu_count() // max(num_devices, 1), workers)  # number of workers
    return StreamingDataLoader(
        dataset=dataset.streaming_dataset,
        batch_size=batch,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        drop_last=drop_last,
    )


class TransformedStreamingDataset(StreamingDataset):
    def transform(self, x, *args, **kwargs):
        """Apply transformations to the data.

        Args:
            x: Data to transform.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Transformed data.
        """
        ...


if _ULTRALYTICS_AVAILABLE:
    from ultralytics.data.base import BaseDataset as UltralyticsBaseDataset
    from ultralytics.utils.plotting import plot_images

    class PatchedUltralyticsBaseDataset(UltralyticsBaseDataset):
        def __init__(self, img_path: str, classes: Optional[List[int]] = None, *args, **kwargs):
            print("patched ultralytics dataset: 🔥")
            self.litdata_dataset = img_path
            self.classes = classes
            super().__init__(img_path, classes=classes, *args, **kwargs)
            self.streaming_dataset = TransformedStreamingDataset(
                img_path,
                transform=[
                    ultralytics_detection_transform,
                    partial(self.transform_update_label, classes),
                ],
            )
            self.ni = len(self.streaming_dataset)
            self.buffer = list(range(len(self.streaming_dataset)))

        def __len__(self):
            """Return the length of the dataset."""
            return len(self.streaming_dataset)

        def get_image_and_label(self, index):
            # Your custom logic to load from .litdata
            # e.g. use `self.litdata_dataset[index]`
            raise NotImplementedError("Custom logic here")

        def get_img_files(self, img_path: Union[str, List[str]]) -> List[str]:
            """Let this method return an empty list to avoid errors."""
            return []

        def get_labels(self) -> List[Dict[str, Any]]:
            # this is used to get number of images (ni) in the BaseDataset class
            return []

        def cache_images(self) -> None:
            pass

        def cache_images_to_disk(self, i: int) -> None:
            pass

        def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
            """Check if the cache disk is available."""
            # This method is not used in the streaming dataset, so we can return True
            return True

        def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
            """Check if the cache RAM is available."""
            # This method is not used in the streaming dataset, so we can return True
            return True

        def update_labels(self, *args, **kwargs):
            """Do nothing, we will update labels when item is fetched in transform."""
            pass

        def transform_update_label(self, include_class: Optional[List[int]], label: Dict, *args, **kwargs) -> Dict:
            """Update labels to include only specified classes.

            Args:
                self: PatchedUltralyticsBaseDataset instance.
                include_class (List[int], optional): List of classes to include. If None, all classes are included.
                label (Dict): Label to update.
                *args: Additional positional arguments (unused).
                **kwargs: Additional keyword arguments (unused).
            """
            include_class_array = np.array(include_class).reshape(1, -1)
            if include_class is not None:
                cls = label["cls"]
                bboxes = label["bboxes"]
                segments = label["segments"]
                keypoints = label["keypoints"]
                j = (cls == include_class_array).any(1)
                label["cls"] = cls[j]
                label["bboxes"] = bboxes[j]
                if segments:
                    label["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    label["keypoints"] = keypoints[j]
            if self.single_cls:
                label["cls"][:, 0] = 0

            return label

        def __getitem__(self, index: int) -> Dict[str, Any]:
            """Return transformed label information for given index."""
            # return self.transforms(self.get_image_and_label(index))
            if not hasattr(self, "streaming_dataset"):
                raise ValueError("The dataset must have a 'streaming_dataset' attribute.")
            data = self.streaming_dataset[index]

            label = data["label"]
            # split label on the basis of `\n` and then split each line on the basis of ` `
            # first element is class, rest are bbox coordinates
            if isinstance(label, str):
                label = label.split("\n")
                label = [line.split(" ") for line in label if line.strip()]

                data = {
                    "batch_idx": torch.Tensor([index], dtype=torch.int32),  # ← add this!
                    "img": data["image"],
                    "cls": torch.Tensor([int(line[0]) for line in label]),
                    "bboxes": torch.Tensor([[float(coord) for coord in line[1:]] for line in label]),
                    "normalized": True,
                    "segments": [],
                    "keypoints": None,
                    "bbox_format": "xywh",
                }
            else:
                raise ValueError("Label must be a string in YOLO format.")

            data = self.transform_update_label(
                include_class=self.classes,
                label=data,
            )
            print("-" * 100)
            return self.transforms(data)

    def patch_detection_plot_training_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            self: DetectionTrainer instance.
            batch (Dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            labels=batch,
            images=batch["img"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def patch_detection_plot_training_labels(self) -> None:
        """Create a labeled training plot of the YOLO model."""
        pass

    def patch_get_labels(self) -> List[Dict[str, Any]]:
        # this is used to get number of images (ni) in the BaseDataset class
        return []

    def patch_none_function(*args, **kwargs):
        """A placeholder function that does nothing."""
        pass

# ------- helper transformations -------


def ultralytics_detection_transform(data: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Transform function for YOLO detection datasets.

    Args:
        data (Dict[str, Any]): Input data containing image and label.
        index (int): Index of the data item.

    Returns:
        Dict[str, Any]: Transformed data with image and label.
    """
    if index is None:
        raise ValueError("Index must be provided for YOLO detection transform.")
    label = data["label"]
    # split label on the basis of `\n` and then split each line on the basis of ` `
    # first element is class, rest are bbox coordinates
    if isinstance(label, str):
        label = label.split("\n")
        label = [line.split(" ") for line in label if line.strip()]
        print(f"label={label}")

        data = {
            "batch_idx": torch.Tensor([index]),  # ← add this!
            "img": data["image"],
            "cls": torch.Tensor([int(line[0]) for line in label]),
            "bboxes": torch.Tensor([[float(coord) for coord in line[1:]] for line in label]),
            "normalized": True,
            "segments": [],
            "keypoints": None,
            "bbox_format": "xywh",
        }
        print(f"{data=}")
    else:
        raise ValueError("Label must be a string in YOLO format.")

    return data
