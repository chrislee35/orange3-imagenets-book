#!/usr/bin/env python
"""Build the cats / dogs / ducks photo set from Google's Open Images dataset.

Usage:  python utils/export_animals.py [--per-class 50] [--test 20] [--out datasets] [--seed 0]

Output:  <out>/animals/train/<class>/<n>.jpg, <out>/animals/test/<class>/<n>.jpg
         <out>/animals/ATTRIBUTION.csv   (author, source page and licence of every image)

Why Open Images: every image in it is listed as CC BY 2.0 (attribution required), and each one comes with its author
and source page, so the set can be rebuilt and shared with proper credit.  Google says it cannot guarantee the
licence labels are correct, so keep ATTRIBUTION.csv with any copy you share.  The images are downloaded from the
public Open Images mirror, shrunk so the longer side is at most 256 pixels (an adaptation, as CC BY requires us to
say), and saved as JPEG.  The folders are not committed (see .gitignore).

After you delete pictures by hand, run  python utils/export_animals.py --sync  to update ATTRIBUTION.csv and record the
rejected ids in utils/animals_exclude.txt (which is committed, so a rebuild skips them).

Selection: only the validation and test splits are used (they are small), and only images that
  * contain exactly one of the three animals (no mixed cat-and-dog pictures) and no goose or swan,
  * show the animal large (its biggest box covers at least 10% of the picture),
  * are not drawings or toys ("depiction") and not a group of animals, and
  * need no rotation fix.
"""
import argparse
import csv
import io
import random
import urllib.request
from pathlib import Path

from PIL import Image

OI = "https://storage.googleapis.com/openimages"
S3 = "https://open-images-dataset.s3.amazonaws.com"
CLASSES = {"/m/01yrx": "cat", "/m/0bt9lr": "dog", "/m/09ddx": "duck"}
# Open Images often puts "Duck" boxes on geese and swans (or tags one picture with all three); such pictures are
# left out so that the duck class means ducks.
LOOKALIKES = {"/m/0dbvp": "goose", "/m/0dftk": "swan"}
SPLITS = {  # split -> (boxes csv, image-info csv)
    "validation": (f"{OI}/v5/validation-annotations-bbox.csv", f"{OI}/2018_04/validation/validation-images-with-rotation.csv"),
    "test": (f"{OI}/v5/test-annotations-bbox.csv", f"{OI}/2018_04/test/test-images-with-rotation.csv"),
}
CC_BY = "https://creativecommons.org/licenses/by/2.0/"
# Pictures we looked at and rejected (a puffin or a gull labelled "duck", and so on), one Open Images id per line.
EXCLUDE = Path(__file__).with_name("animals_exclude.txt")
# The exact pictures of the book's set (split, class, id), written by --sync once the set has been cleaned by hand.
# When it exists the builder downloads exactly these pictures, so everyone gets the set we reviewed.
SELECTION = Path(__file__).with_name("animals_selection.csv")
MIN_AREA = 0.1
MAX_SIDE = 256


def fetch(url, cache):
    """Download ``url`` into the cache folder once and return the local path."""
    path = cache / url.rsplit("/", 1)[1]
    if not path.exists():
        print("downloading", url)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return path


def excluded():
    if not EXCLUDE.exists():
        return set()
    return {line.split("#")[0].strip() for line in EXCLUDE.read_text().splitlines() if line.split("#")[0].strip()}


def candidates(cache):
    """{class: [(split, image id, info row)]} for every image that passes the selection above."""
    found = {c: [] for c in CLASSES.values()}
    skip = excluded()
    for split, (boxes_url, info_url) in SPLITS.items():
        boxes = {}
        with open(fetch(boxes_url, cache), newline="") as f:
            for r in csv.DictReader(f):
                cls = CLASSES.get(r["LabelName"]) or LOOKALIKES.get(r["LabelName"])
                if cls:
                    boxes.setdefault(r["ImageID"], {}).setdefault(cls, []).append(r)
        keep = {}
        for image_id, by_class in boxes.items():
            if len(by_class) != 1 or next(iter(by_class)) in LOOKALIKES.values():
                continue
            (cls, rows), = by_class.items()
            if any(r["IsGroupOf"] == "1" or r["IsDepiction"] == "1" for r in rows):
                continue
            area = max((float(r["XMax"]) - float(r["XMin"])) * (float(r["YMax"]) - float(r["YMin"])) for r in rows)
            if area >= MIN_AREA:
                keep[image_id] = cls
        with open(fetch(info_url, cache), newline="") as f:
            for r in csv.DictReader(f):
                if r["ImageID"] in keep and r["ImageID"] not in skip and r["License"] == CC_BY and r["Rotation"] in ("", "0", "0.0"):
                    found[keep[r["ImageID"]]].append((split, r["ImageID"], r))
    return found


def download(split, image_id):
    """The picture as a PIL image, shrunk to MAX_SIDE, or None if it cannot be fetched or read."""
    try:
        with urllib.request.urlopen(f"{S3}/{split}/{image_id}.jpg", timeout=30) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception as e:
        print("  skipped", image_id, type(e).__name__)
        return None
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def sync(base, balance=True, seed=0):
    """After you delete pictures by hand: forget them in ATTRIBUTION.csv and remember their ids in animals_exclude.txt."""
    table = base / "ATTRIBUTION.csv"
    with open(table, newline="") as f:
        rows = list(csv.DictReader(f))
    kept, removed = [], []
    for r in rows:
        (kept if (base / r["split"] / r["class"] / r["file"]).exists() else removed).append(r)
    known = {(r["split"], r["class"], r["file"]) for r in rows}
    stray = [p for p in base.glob("*/*/*.jpg") if (p.parent.parent.name, p.parent.name, p.name) not in known]
    if stray:
        print("warning: pictures that ATTRIBUTION.csv does not list (renamed or added by hand?):", [str(p) for p in stray][:5])
    old = excluded()
    new = [r for r in removed if r["open_images_id"] not in old]
    if balance:  # every class gets as many pictures as the smallest one in its split (the surplus is not "bad", just dropped)
        rng = random.Random(seed)
        by = {}
        for r in kept:
            by.setdefault((r["split"], r["class"]), []).append(r)
        for split in {k[0] for k in by}:
            target = min(len(v) for k, v in by.items() if k[0] == split)
            for (sp, cls), v in by.items():
                if sp == split and len(v) > target:
                    for r in rng.sample(sorted(v, key=lambda r: r["file"]), len(v) - target):
                        (base / sp / cls / r["file"]).unlink()
                        kept.remove(r)
    if new:
        with open(EXCLUDE, "a") as f:
            for r in new:
                f.write(f"{r['open_images_id']}  # {r['class']}, rejected by hand\n")
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(kept)
    with open(SELECTION, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "open_images_id"])
        w.writerows(sorted((r["split"], r["class"], r["open_images_id"]) for r in kept))
    counts = {}
    for r in kept:
        counts.setdefault(r["split"], {}).setdefault(r["class"], 0)
        counts[r["split"]][r["class"]] += 1
    print(f"removed {len(removed)} rows ({len(new)} new exclusions); {len(kept)} pictures left: {counts}")


def reviewed(found):
    """{class: [(part, split, id, info)]}: exactly the pictures listed in animals_selection.csv."""
    chosen = {}
    with open(SELECTION, newline="") as f:
        for r in csv.DictReader(f):
            chosen[r["open_images_id"]] = (r["split"], r["class"])
    plan = {}
    for cls, items in found.items():
        for split, image_id, info in sorted(items, key=lambda t: t[1]):
            if image_id in chosen and chosen[image_id][1] == cls:
                plan.setdefault(cls, []).append((chosen[image_id][0], split, image_id, info))
    missing = len(chosen) - sum(len(v) for v in plan.values())
    if missing:
        print(f"warning: {missing} selected pictures are no longer among the candidates")
    return plan


def fresh_draw(found, per_class, test, seed):
    """{class: [(part, split, id, info)]}: a random draw (test pictures first), before any pictures are checked."""
    rng = random.Random(seed)
    plan = {}
    for cls, items in found.items():
        rng.shuffle(items)
        plan[cls] = [("test" if k < test else "train", split, image_id, info)
                     for k, (split, image_id, info) in enumerate(items[:per_class + test + 10])]  # a few spares for failed downloads
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=50, help="training images per class")
    ap.add_argument("--test", type=int, default=20, help="test images per class")
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reselect", action="store_true", help="ignore animals_selection.csv and draw a fresh random set")
    ap.add_argument("--sync", action="store_true", help="after deleting pictures by hand, update ATTRIBUTION.csv, the exclude list and the selection")
    a = ap.parse_args()
    base = Path(a.out) / "animals"
    if a.sync:
        return sync(base)
    cache = Path(a.out) / "_openimages_cache"
    cache.mkdir(parents=True, exist_ok=True)
    found = candidates(cache)
    print({c: len(v) for c, v in found.items()}, "candidate images")
    if SELECTION.exists() and not a.reselect:
        print("using the reviewed selection in", SELECTION.name)
        plan = reviewed(found)
    else:
        plan = fresh_draw(found, a.per_class, a.test, a.seed)
    rows = []
    for cls, items in plan.items():
        quota = {"train": a.per_class, "test": a.test} if not (SELECTION.exists() and not a.reselect) else None
        saved = {"train": 0, "test": 0}
        for part, split, image_id, info in items:
            if quota and saved[part] >= quota[part]:
                continue
            img = download(split, image_id)
            if img is None:
                continue
            folder = base / part / cls
            folder.mkdir(parents=True, exist_ok=True)
            img.save(folder / f"{image_id}.jpg", quality=90)
            saved[part] += 1
            rows.append([part, cls, f"{image_id}.jpg", image_id, info["Author"], info["AuthorProfileURL"],
                         info["OriginalLandingURL"], info["Title"], info["License"]])
        print(f"{cls}: {saved['train']} train, {saved['test']} test")
    with open(base / "ATTRIBUTION.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "file", "open_images_id", "author", "author_url", "source_page", "title", "license"])
        w.writerows(sorted(rows))
    print(f"wrote {len(rows)} images and {base / 'ATTRIBUTION.csv'}")


if __name__ == "__main__":
    main()
