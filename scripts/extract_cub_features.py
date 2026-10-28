"""Process CUB-200-2011 concepts and cache frozen ResNet-50 features.

    python scripts/extract_cub_features.py --cub-dir data/CUB_200_2011 --out-dir data/cub_features

Concept processing follows Koh et al. (2020): on the training split, a class has an
attribute if at least half of its images are labelled with it (majority vote); attributes
present in at least 10 classes are kept (112 in the paper). Every image then gets its
class's attribute vector ("denoised"). The per-image crowd-worker labels of the same
attributes are saved as "concepts_raw".
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


def read_table(path, n_cols):
    """Whitespace table of ints; tolerates the few malformed lines in CUB's attribute file."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= n_cols:
            rows.append([int(p) for p in parts[:n_cols]])
    return np.array(rows)


def load_cub(cub_dir: Path):
    names = dict(line.split(" ", 1) for line in (cub_dir / "images.txt").read_text().splitlines())
    paths = [names[str(i)] for i in range(1, len(names) + 1)]
    is_train = read_table(cub_dir / "train_test_split.txt", 2)[:, 1].astype(bool)
    labels = read_table(cub_dir / "image_class_labels.txt", 2)[:, 1] - 1

    attr_file = cub_dir / "attributes" / "attributes.txt"
    if not attr_file.exists():
        attr_file = cub_dir.parent / "attributes.txt"
    attr_names = [line.split(" ", 1)[1] for line in attr_file.read_text().splitlines()]

    table = read_table(cub_dir / "attributes" / "image_attribute_labels.txt", 3)
    raw = np.zeros((len(paths), len(attr_names)), dtype=np.float32)
    raw[table[:, 0] - 1, table[:, 1] - 1] = table[:, 2]
    return paths, is_train, labels, attr_names, raw


def select_concepts(raw, labels, train_idx, min_classes=10):
    n_classes = labels.max() + 1
    class_attr = np.zeros((n_classes, raw.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        class_attr[c] = raw[train_idx][labels[train_idx] == c].mean(0) >= 0.5
    keep = np.where(class_attr.sum(0) >= min_classes)[0]
    return keep, class_attr[:, keep]


@torch.no_grad()
def extract_features(cub_dir, paths, device, batch_size=64):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    backbone = models.resnet50(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([tf(Image.open(cub_dir / "images" / p).convert("RGB")) for p in paths[i:i + batch_size]])
        feats.append(backbone(batch.to(device)).cpu())
        print(f"\r{min(i + batch_size, len(paths))}/{len(paths)} images", end="", flush=True)
    print()
    return torch.cat(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-dir", type=Path, default=Path("data/CUB_200_2011"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/cub_features"))
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths, is_train, labels, attr_names, raw = load_cub(args.cub_dir)
    rng = np.random.default_rng(args.seed)
    train_all = np.where(is_train)[0]
    rng.shuffle(train_all)
    n_val = int(len(train_all) * args.val_frac)
    splits = {"train": np.sort(train_all[n_val:]), "val": np.sort(train_all[:n_val]), "test": np.where(~is_train)[0]}

    keep, class_concepts = select_concepts(raw, labels, splits["train"])
    concept_names = [attr_names[j] for j in keep]
    group_names = sorted({n.split("::")[0] for n in concept_names})
    concept_groups = [group_names.index(n.split("::")[0]) for n in concept_names]
    print(f"kept {len(keep)} concepts in {len(group_names)} groups")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    features = extract_features(args.cub_dir, paths, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, idx in splits.items():
        torch.save(
            {
                "features": features[idx],
                "concepts": torch.from_numpy(class_concepts[labels[idx]]),
                "concepts_raw": torch.from_numpy(raw[idx][:, keep]),
                "labels": torch.from_numpy(labels[idx]),
                "concept_names": concept_names,
                "concept_groups": concept_groups,
            },
            args.out_dir / f"{split}.pt",
        )
        print(f"{split}: {len(idx)} images")


if __name__ == "__main__":
    main()
