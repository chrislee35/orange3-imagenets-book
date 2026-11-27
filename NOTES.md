# Research notes

## Orange3-ImageAnalytics 0.13.0 (checked 2026-11-24)

Location: `~/code/miniconda3/envs/orange/lib/python3.12/site-packages/orangecontrib/imageanalytics`

Widgets: Import Images, Image Viewer, Image Grid, Image Embedding, Save Images. No training or model building.

- **Import Images**: folder tree -> image table. Subfolder name = `category` class. Image column has `attributes["type"]="image"` and `["origin"]=<start dir>`; cell value is the path relative to origin. orange3-imagenets already relies on this.
- **Image Embedding** embedders: inception-v3, painters, vgg16, vgg19, openface, squeezenet, yeast. Only squeezenet is local; the rest use Orange's embedding server (needs internet).
- **Image Viewer / Image Grid / Save Images** reusable for inspecting data, mistakes, and GAN/diffusion output (Generate Images can emit an image table pointing at a folder).

### Consequences for the book / add-on (all to be addressed)

1. Dataset loaders must write `<class>/<n>.png` folders so Import Images works unchanged.
2. Add a transfer-learning lab: Import Images -> Image Embedding (squeezenet, offline) -> Logistic Regression -> Test and Score, as a baseline for the CNN labs.
3. Reuse Image Viewer for generated images instead of writing a viewer.
4. **Bug/limit in Train ImageNet** (`ow_image_train_and_score.py` `prepare_data`): `cv2.imread` is always 3-channel BGR and always resized to 224x224. MNIST-family data (28x28x1) would be upscaled 8x and tripled. Make input size and channels settings (and match in Build ImageNet and Classify Images).
5. Chapter 1 must say "ImageNet" here means "an image network", not the ImageNet dataset or ImageAnalytics' ImageNet-trained embedders.
6. Offline caveat: only squeezenet embeds offline.
7. Train ImageNet: unchecked for >10 classes (EMNIST) and validation split. Train hardcodes Adam + categorical cross-entropy.

## Dataset sources (checked 2026-11-24)

Env: `~/code/miniconda3/envs/orange` has keras 3.11.3 (TensorFlow 2.20 backend), torch 2.7.1, torchvision 0.22.1, scikit-learn, Pillow, OpenCV. Orange 3.x itself bundles no image datasets; datasets.biolab.si/core is reachable but I found no MNIST-family sets there.

| Dataset | Source | Result |
|---|---|---|
| MNIST (60k+10k, 28x28x1) | `keras.datasets.mnist` -> storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz (11 MB) | downloaded OK |
| Fashion-MNIST (60k+10k, 28x28x1) | `keras.datasets.fashion_mnist` (same host, 4 small gz files) | downloaded OK, shapes verified |
| CIFAR-10 (50k+10k, 32x32x3) | `keras.datasets.cifar10` -> cs.toronto.edu (170 MB, slow, ~10 min here) | downloaded OK, shape (50000,32,32,3) verified; cifar100 also loads |
| EMNIST | NOT in Keras. `torchvision.datasets.EMNIST` (splits: byclass, bymerge, balanced, letters, digits, mnist) downloads biometrics.nist.gov/cs_links/EMNIST/gzip.zip, 562 MB, reachable | usable but heavy |
| sklearn `load_digits` | bundled, offline, 8x8 | tiny offline fallback |

Plan: a one-time script (`utils/export_datasets.py`) loads each dataset and writes a SMALL balanced subset as `<class>/<n>.png` folders (e.g. 500-1000 per class for MNIST/Fashion/EMNIST-balanced letters, 500 per class for CIFAR-10). Committing the subsets keeps the book self-contained (no TensorFlow or 560 MB download for students) and Import Images reads them directly. Full-size versions can be rebuilt by the script. Keras's loader is Keras-only, so the script itself is a dev tool, not an add-on dependency; if the add-on should offer an in-widget loader, call `keras.datasets` for MNIST/Fashion/CIFAR and torchvision (optional) or a pre-exported subset for EMNIST.

Licensing/citation to put in chapter Notes (verify wording before release): MNIST (LeCun, Cortes, Burges), Fashion-MNIST (Zalando, MIT license), CIFAR-10 (Krizhevsky 2009), EMNIST (Cohen et al. 2017, NIST). Cat/dog/duck set: license unverified, do not commit until confirmed.

### Decisions (2026-11-25)

- Dataset subsets are built **on demand**, not committed: `make datasets` / `utils/export_datasets.py [names] [--per-class N] [--out datasets]` writes `datasets/<name>/{train,test}/<class>/<n>.png` (default 500 train per class, 100 test). `datasets/*/` is gitignored.
- EMNIST variant: **letters** (26 classes, a-z, case merged). Script transposes EMNIST images and maps targets 1..26 to a..z.
- Tested: mnist, fashion-mnist, cifar10 exports (6000 PNGs each, 10 classes) and Import Images reads mnist/train as a 5000-row table (10 classes from the folder names).

## Add-on changes made (2026-11-25, in dev/orange3-imagenets, uncommitted, 39 tests pass)

- New `util/image_loader.py`: `load_image(path, (h,w,c))` (grayscale or RGB, raw 0..255 float32) and `model_input_shape(model)`. Train and Classify share it.
- **Build ImageNet**: Input Images box (Height, Width, Channels 1/3) -> prepends `keras.Input`, so the model knows its shape. Added Flatten and AveragePooling2D buttons (Flatten was missing, so a CNN couldn't be built from scratch). Presets are deep-copied (editing a layer used to mutate the preset).
- **Double-scaling bug fixed**: Clear added a Rescaling(1/255) layer while Train also divided by 255. Now loaders return raw 0..255 and every preset starts with Rescaling. Models saved before this change expect 0..1 input and will mis-score.
- **Train ImageNet**: reads size/channels from the model (fallback 224x224x3); error if last Dense units != number of classes; Validation split (%) with dashed validation curves; split is stratified (Keras's own validation_split takes the last rows, and sorted image folders would hold out whole classes, which I hit: val acc 0.0 and digits 8/9 never trained on).
- **Classify Images**: uses model input shape, batch prediction, and the trained model's `class_names` (previously indexed the data's class values).
- End-to-end check: MNIST subset (5000 train), SimpleCNN, 5 epochs, 6 s on CPU: train 0.961, validation 0.945, test (1000 held-out images) 0.932.

Still open: Classify drops unreadable images, which would misalign the added column; MiniResNet preset has no skip connections (no Add); no tests for >10 classes with EMNIST; GAN/diffusion widgets; Conv2DTranspose/UpSampling in builder.

## Generative widgets (2026-11-25, dev/orange3-imagenets, uncommitted, 54 tests pass)

New widgets: **Train GAN**, **Train Diffusion**, **Generate Images** (+ docs pages, placeholder icons copied from train/classify svgs). Qt-free cores in `util/gan.py`, `util/diffusion.py`; helpers in `util/image_loader.py` (`load_table_images`, `image_grid`, `to_uint8/to_signed`), `util/image_table.py` (`images_to_table`), `widgets/gen_common.py`.

- Both trainers take an image table, resize to Height/Width/Channels settings (divisible by 4), run in the GUI thread with processEvents per epoch (same pattern as Train ImageNet), have Stop, live plots. Models are wrapper objects (`GANModel`, `DiffusionModel`, `.kind`, `.sample()`), NOT Keras models: Save ImageNet / Load ImageNet do not handle them yet.
- Gotcha: Keras 3 does not honor `trainable=False` after compile; the classic compiled "combined GAN" left the discriminator frozen (D acc 0.0, noise images). Fixed with explicit tf.GradientTape steps (tf.function). Regression test exists.
- Measured on CPU with MNIST 5000 images: GAN 15 epochs ~40 s, stroke-like digits (see below). Diffusion base_filters 32: ~40 s/epoch (20 epochs = 10 min, digit-like but rough); base_filters 16: ~12 s/epoch (default). Sampling 20 images x 200 steps ~20-26 s. Diffusion quality at 20 epochs: digit-like strokes, class conditioning only partly right (1s and 7s ok). More epochs/data will help; book labs should say so honestly.
- Generate Images writes PNGs to a new temp dir each run.

Open: Save/Load for GAN/diffusion models; real icons; Build ImageNet still has no Conv2DTranspose/UpSampling (not needed now because the generative widgets build their own nets); not yet run in the real Orange canvas (tested offscreen via pytest and widget screenshots only); diffusion on color (CIFAR) untested; GAN/diffusion widgets block the GUI during training.

## Save/Load and workflows (2026-11-25)

- `util/generative_io.py`: `.imagenet` file = zip of `meta.json` + `net.keras`; Save ImageNet and Load ImageNet got "Save/Load GAN / diffusion model" buttons (h5/JSON save refuses these models). Round trip verified: same seed gives identical images after reload. Uncommitted in dev/orange3-imagenets (59 tests pass).
- `utils/make_ows.py` (book repo) generates `workflows/mnist-gan.ows`, `mnist-diffusion.ows`, `mnist-cnn.ows`; `--check` passes (parses, loads the dev widgets). They point at `datasets/mnist/{train,test}` (built with `make datasets`/export_datasets.py; gitignored).
- Launch canvas with dev widgets: scratchpad `launch_canvas.py [workflow.ows]` (sets `orangecontrib.__path__`); the installed 0.0.5 lacks the new widgets.

## Chapters 1-3 drafted (2026-11-26)

Numbers in the text come from scripted runs (scratchpad exp*.py; same data, SimpleCNN, Adam, batch 32) plus one widget run; see each chapter's Notes. Widget changes made while drafting (dev/orange3-imagenets, uncommitted, 67 tests pass):

- **Augment Images** mutated the INPUT table's `origin` (shared domain), breaking the original table downstream. Fixed via `util/image_table.retarget_images`. Also: Horizontal Flip now off by default (flips ruin digits), new "Include the original images" option (default on), no hardcoded `~/Downloads/test` save folder, preview files in the system temp dir.
- **Preprocess Images**: same origin mutation fixed; Width/Height minimum lowered from 32 to 4 (MNIST 14x14 was impossible). It only runs when an output folder is chosen, not when options change (documented in Ch 2).
- **Classify Images** now shows accuracy when the data has a class.
- **Build ImageNet**: shows "Weights / Output"; float spin boxes show 6 decimals (Rescaling scale 1/255 showed 0.00); default input is still 224x224x3 (Ch 3 warns).
- **Train ImageNet**: removed the Dropout Rate combo (it was never wired to the model; dropout is a layer in Build ImageNet); legend moved off the curves.
- Gotchas documented in Ch 3: Train retrains on every model edit; GUI blocks during training. Validation split is taken after augmentation, so augment-then-validate leaks (Ch 2 says judge on the test set).
- Workflows added: `mnist-explore.ows` (Ch 1), `datasets-tour.ows` (Ch 2). `datasets/{mnist,fashion-mnist,emnist-letters,cifar10}` copied from the scratchpad exports (gitignored).
- Still open: cat/dog/duck set not built (Ch 2 does not use it); labs 2B-2D and 3A-3E workflows were verified by script/widget run, not click-by-click in the canvas.

## Title image (2026-11-26)

`imgs/Title-Image.png` (1200 x 400): a real MNIST 7 noised with the book's 200-step schedule (steps 0, 30, 60, 100, 199), then a class-7 sample denoised by the 30-epoch diffusion model (6 snapshots, seed 21), with the title in DejaVu Sans. Built with Pillow from the add-on's `make_schedule` and `load_generative`; the model file was a scratch `.imagenet` (retrain with Lab 9A to reproduce).

## Cats, dogs, ducks (2026-11-26)

The original set (Kaggle mixtures: Bing thumbnails, Pixabay/Unsplash-style names, unknown cat numbers, Open Images ducks) has no clear licence and is not used. Replacement: `utils/export_animals.py` (`make animals`) builds it from Open Images validation + test boxes (CC BY 2.0, author per picture in `datasets/animals/ATTRIBUTION.csv`; Google does not guarantee the licence labels). Filters: one of cat/dog/duck only, no goose/swan boxes, biggest box >= 10% of the picture, not a depiction or group, no rotation fix needed. Only 71 duck candidates exist in those splits (63 after the hand review removed 8 non-ducks: gulls, puffins); `utils/animals_exclude.txt` lists the rejected ids and `utils/animals_selection.csv` the exact 186 pictures (42 train + 20 test per class), which the builder downloads so a rebuild is identical (checked). Pictures are fetched from the public Open Images S3 mirror (Flickr hosts refuse the sandbox). `--sync` updates the metadata after deleting pictures by hand; `--reselect` draws a fresh random set (larger train split needs a 2 GB CSV, not done).
Results (Lab 7F): SimpleCNN from scratch 56.7%, 60.0%; frozen MobileNet 81.7-88.3% in five runs.
Lab 7F (`workflows/animals-transfer.ows`) was opened and checked by hand in the Orange canvas on 2026-11-27: everything worked.
