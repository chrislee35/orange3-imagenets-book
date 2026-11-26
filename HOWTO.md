# How to build the book

Requirements: `make`, `pandoc`, and Python with `pillow` (only for `utils/resize_imgs.py`).

```
make          # writes public/index.html (plus public/imgs, public/css)
make clean
```

Only HTML is produced. `public/` is gitignored build output and is the website root; the GitHub Actions workflow builds and deploys it to Pages.

Chapter order is set in `chapters.conf`. Image filenames must be unique across chapters because the build copies them all into `imgs/`.
- Datasets: run `make datasets` (Orange env with keras + torchvision; EMNIST letters is a 562 MB download).
