#!/usr/bin/env bash
# Downloads CUB-200-2011 (~1.1 GB) from Caltech into data/.
set -euo pipefail
mkdir -p data
curl -L -o data/CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
tar -xzf data/CUB_200_2011.tgz -C data
rm data/CUB_200_2011.tgz
