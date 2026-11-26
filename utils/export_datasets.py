#!/usr/bin/env python
"""Export small balanced subsets of image datasets as <class>/<n>.png folders.

The folders can be read directly by Orange's Import Images widget.  They are
built on demand and are not committed (see .gitignore).

Usage (run with the Orange env, which has keras + torchvision):
    python utils/export_datasets.py [mnist fashion-mnist cifar10 emnist-letters] [--per-class N] [--out datasets]

Output: <out>/<name>/train/<class>/<i>.png and <out>/<name>/test/<class>/<i>.png
Set KERAS_HOME / TORCH_HOME to control where the raw downloads are cached.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

FASHION = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]
CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
DIGITS = [str(i) for i in range(10)]
LETTERS = list("abcdefghijklmnopqrstuvwxyz")


def load(name):
    """Return (x_train, y_train, x_test, y_test, class_names); y is 0-based."""
    if name in ("mnist", "fashion-mnist", "cifar10"):
        from keras import datasets as d
        mod = {"mnist": d.mnist, "fashion-mnist": d.fashion_mnist, "cifar10": d.cifar10}[name]
        (xa, ya), (xb, yb) = mod.load_data()
        names = {"mnist": DIGITS, "fashion-mnist": FASHION, "cifar10": CIFAR}[name]
        return xa, ya.ravel(), xb, yb.ravel(), names
    if name == "emnist-letters":
        from torchvision.datasets import EMNIST
        root = os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
        out = []
        for train in (True, False):
            ds = EMNIST(root, split="letters", train=train, download=True)
            # EMNIST images are stored transposed; targets are 1..26
            out += [ds.data.numpy().transpose(0, 2, 1), ds.targets.numpy() - 1]
        return out[0], out[1], out[2], out[3], LETTERS
    raise SystemExit(f"unknown dataset {name}")


def write_split(x, y, names, per_class, dest, rng):
    for c, cname in enumerate(names):
        idx = np.flatnonzero(y == c)
        idx = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        folder = Path(dest) / cname
        folder.mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(sorted(idx)):
            Image.fromarray(x[i]).save(folder / f"{n:04d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["mnist", "fashion-mnist", "cifar10", "emnist-letters"])
    ap.add_argument("--per-class", type=int, default=500, help="training images per class (test gets a fifth, min 50)")
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for name in a.names:
        rng = np.random.default_rng(a.seed)
        xa, ya, xb, yb, names = load(name)
        base = Path(a.out) / name
        write_split(xa, ya, names, a.per_class, base / "train", rng)
        write_split(xb, yb, names, max(50, a.per_class // 5), base / "test", rng)
        print(f"{name}: {len(names)} classes -> {base}")


if __name__ == "__main__":
    main()
