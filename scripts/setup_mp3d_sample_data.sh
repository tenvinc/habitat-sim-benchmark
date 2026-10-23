#!/usr/bin/env bash
# Downloads MP3D example scene + PointNav episodes for that scene only.
set -euo pipefail

DATA_ROOT="${1:-/workspace/data}"
MP3D_SCENE="17DRP5sb8fy"
MP3D_URL="http://dl.fbaipublicfiles.com/habitat/mp3d/mp3d_example_v1.1.zip"
POINTNAV_ZIP_URL="http://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip"

SCENES_DIR="${DATA_ROOT}/scene_datasets/mp3d"
DATASET_DIR="${DATA_ROOT}/datasets/pointnav/mp3d/v1/train"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${SCENES_DIR}" "${DATASET_DIR}/content"

if [[ ! -f "${SCENES_DIR}/${MP3D_SCENE}/${MP3D_SCENE}.glb" ]]; then
  echo "Downloading MP3D example scene (${MP3D_SCENE})..."
  curl -fsSL "${MP3D_URL}" -o "${TMP_DIR}/mp3d_example.zip"
  unzip -q "${TMP_DIR}/mp3d_example.zip" -d "${TMP_DIR}/mp3d_example"
  cp -a "${TMP_DIR}/mp3d_example/${MP3D_SCENE}" "${SCENES_DIR}/"
  cp "${TMP_DIR}/mp3d_example/mp3d.scene_dataset_config.json" "${SCENES_DIR}/"
fi

if [[ ! -f "${DATASET_DIR}/content/${MP3D_SCENE}.json.gz" ]]; then
  echo "Extracting PointNav episodes for ${MP3D_SCENE}..."
  curl -fsSL "${POINTNAV_ZIP_URL}" -o "${TMP_DIR}/pointnav_mp3d_v1.zip"
  unzip -p "${TMP_DIR}/pointnav_mp3d_v1.zip" "train/content/${MP3D_SCENE}.json.gz" \
    > "${DATASET_DIR}/content/${MP3D_SCENE}.json.gz"
fi

if [[ ! -f "${DATASET_DIR}/train.json.gz" ]]; then
  echo '{"episodes": []}' | gzip -c > "${DATASET_DIR}/train.json.gz"
fi

echo "MP3D sample data ready under ${DATA_ROOT}"
