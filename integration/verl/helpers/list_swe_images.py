# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Emit the unique sandbox image refs from preprocessed SWE parquet files.

Feed the output to the pre-warm Job (see configs/swe_image_prewarm_job.yaml)
to populate the Artifact Registry remote-repo cache before training:

    python integration/verl/helpers/list_swe_images.py \
        ~/data/swe/train.parquet ~/data/swe/test.parquet > /tmp/swe_images.txt
    kubectl create configmap swe-image-list --from-file=images.txt=/tmp/swe_images.txt
    kubectl apply -f configs/swe_image_prewarm_job.yaml
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", help="Parquet files from prepare_swe_dataset.py")
    args = parser.parse_args()

    images: set[str] = set()
    for path in args.parquet:
        df = pd.read_parquet(path, columns=["extra_info"])
        images.update(info["docker_image"] for info in df["extra_info"])

    for image in sorted(images):
        print(image)
    print(f"{len(images)} unique images", file=sys.stderr)


if __name__ == "__main__":
    main()
