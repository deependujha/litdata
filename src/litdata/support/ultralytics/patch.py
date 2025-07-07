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
from typing import Dict

from litdata.constants import _ULTRALYTICS_AVAILABLE


def patch_ultralytics() -> None:
    """Patch Ultralytics to use the LitData optimize function."""
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics is not available. Please install it to use this functionality.")

    import sys

    if "ultralytics" in sys.modules:
        raise RuntimeError("patch_ultralytics() must be called before importing 'ultralytics'")

    from ultralytics.data.utils import check_det_dataset

    check_det_dataset.__code__ = patch_check_det_dataset.__code__


def patch_check_det_dataset(dataset: str, _: bool = True) -> Dict:
    if not (isinstance(dataset, str) and dataset.endswith(".yaml") and os.path.isfile(dataset)):
        raise ValueError("Dataset must be a string ending with '.yaml' and point to a valid file.")

    import yaml

    # read the yaml file
    with open(dataset) as file:
        data = yaml.safe_load(file)
    print(f"patch successful for {dataset}")
    return data
