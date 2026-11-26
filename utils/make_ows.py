#!/usr/bin/env python
"""Generate Orange .ows workflows from a small Python description.

Usage:  python utils/make_ows.py [name ...] [--check]     (writes workflows/<name>.ows)

Run with the conda env that has Orange and orange3-imagenets installed (the new widgets must be importable; see the
make-ows skill).  Node settings are written as pickled dicts (format="pickle"), the same way Orange saves them.  Anything
not set here falls back to the widget's defaults.
"""
import base64
import pickle
import sys
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

BOOK = Path(__file__).resolve().parent.parent
DEV = Path.home() / "code/orange/dev/orange3-imagenets"  # checkout with the widgets, until a release has them

IA = "Orange3-ImageAnalytics"
IN = "orange3-imagenets"
# widget name -> (qualified class name, add-on project name)
WIDGETS = {
    "Import Images": ("orangecontrib.imageanalytics.widgets.owimageimport.OWImportImages", IA),
    "Image Viewer": ("orangecontrib.imageanalytics.widgets.owimageviewer.OWImageViewer", IA),
    "Image Embedding": ("orangecontrib.imageanalytics.widgets.owimageembedding.OWImageEmbedding", IA),
    "Build ImageNet": ("orangecontrib.imagenets.widgets.ow_imagenet_builder.OWImageNetBuilder", IN),
    "Train ImageNet": ("orangecontrib.imagenets.widgets.ow_image_train_and_score.OWImageTrainAndScore", IN),
    "Classify Images": ("orangecontrib.imagenets.widgets.ow_imagenet_classify.OWImageNetClassify", IN),
    "Save ImageNet": ("orangecontrib.imagenets.widgets.ow_save_imagenet.OWSaveImageNet", IN),
    "Train GAN": ("orangecontrib.imagenets.widgets.ow_train_gan.OWTrainGAN", IN),
    "Train Diffusion": ("orangecontrib.imagenets.widgets.ow_train_diffusion.OWTrainDiffusion", IN),
    "Generate Images": ("orangecontrib.imagenets.widgets.ow_generate_images.OWGenerateImages", IN),
    "Pretrained ImageNet": ("orangecontrib.imagenets.widgets.ow_pretrained_imagenet.OWPretrainedImageNet", IN),
    "Inspect ImageNet": ("orangecontrib.imagenets.widgets.ow_inspect_imagenet.OWInspectImageNet", IN),
    "Misclassified Images": ("orangecontrib.imagenets.widgets.ow_misclassified.OWMisclassified", IN),
    "Load ImageNet": ("orangecontrib.imagenets.widgets.ow_load_imagenet.OWLoadImageNet", IN),
    "Distributions": ("Orange.widgets.visualize.owdistributions.OWDistributions", "Orange3"),
    "Data Table": ("Orange.widgets.data.owtable.OWTable", "Orange3"),
}
# (source widget, sink widget) -> ((source channel name, id), (sink channel name, id))
LINKS = {
    ("Import Images", "Train GAN"): (("Data", "data"), ("Images", "data")),
    ("Import Images", "Train Diffusion"): (("Data", "data"), ("Images", "data")),
    ("Import Images", "Pretrained ImageNet"): (("Data", "data"), ("Images", "data")),
    ("Pretrained ImageNet", "Train ImageNet"): (("Model", "model"), ("Learner", "model")),
    ("Classify Images", "Misclassified Images"): (("Annotated Data", "annotated_data"), ("Annotated Data", "data")),
    ("Misclassified Images", "Image Viewer"): (("Misclassified", "misclassified"), ("Data", "data")),
    ("Train ImageNet", "Inspect ImageNet"): (("Trained Model", "trained_model"), ("Model", "model")),
    ("Import Images", "Inspect ImageNet"): (("Data", "data"), ("Images", "data")),
    ("Import Images", "Image Viewer"): (("Data", "data"), ("Data", "data")),
    ("Import Images", "Data Table"): (("Data", "data"), ("Data", "data")),
    ("Import Images", "Distributions"): (("Data", "data"), ("Data", "data")),
    ("Import Images", "Train ImageNet"): (("Data", "data"), ("Evaluation Data", "data")),
    ("Import Images", "Classify Images"): (("Data", "data"), ("Data", "data")),
    ("Build ImageNet", "Train ImageNet"): (("Model", "model"), ("Learner", "model")),
    ("Train ImageNet", "Classify Images"): (("Trained Model", "trained_model"), ("Model", "model")),
    ("Train ImageNet", "Save ImageNet"): (("Trained Model", "trained_model"), ("Model", "model")),
    ("Classify Images", "Data Table"): (("Annotated Data", "annotated_data"), ("Data", "data")),
    ("Train GAN", "Generate Images"): (("Generator", "generator"), ("Model", "model")),
    ("Train Diffusion", "Generate Images"): (("Diffusion Model", "model"), ("Model", "model")),
    ("Generate Images", "Image Viewer"): (("Images", "images"), ("Data", "data")),
}


def build(nodes, links, annotations=()):
    """nodes: [(key, widget, title, (x, y), settings dict)], links: [(key, key)], annotations: [(text, (x, y, w, h))]."""
    ids = {k: i for i, (k, *_) in enumerate(nodes)}
    kind = {k: w for k, w, *_ in nodes}
    out = ["<?xml version='1.0' encoding='utf-8'?>", '<scheme version="2.0" title="" description="">', "\t<nodes>"]
    for k, w, title, (x, y), _ in nodes:
        q, proj = WIDGETS[w]
        out.append(f'\t\t<node id="{ids[k]}" name="{w}" qualified_name="{q}" project_name="{proj}" '
                   f'version="" title={quoteattr(title)} position="({x}, {y})" />')
    out += ["\t</nodes>", "\t<links>"]
    for i, (a, b) in enumerate(links):
        (sn, sid), (kn, kid) = LINKS[(kind[a], kind[b])]
        out.append(f'\t\t<link id="{i}" source_node_id="{ids[a]}" sink_node_id="{ids[b]}" '
                   f'source_channel="{sn}" sink_channel="{kn}" enabled="true" '
                   f'source_channel_id="{sid}" sink_channel_id="{kid}" />')
    out += ["\t</links>", "\t<annotations>"]
    for i, (text, rect) in enumerate(annotations):
        out.append(f'\t\t<text id="{i}" type="text/markdown" rect="{rect}" '
                   f'font-family="Sans Serif" font-size="16">{escape(text)}</text>')
    out += ["\t</annotations>", "\t<thumbnail />", "\t<node_properties>"]
    for k, _, _, _, settings in nodes:
        if settings:
            blob = base64.b64encode(pickle.dumps(settings, protocol=4)).decode()
            out.append(f'\t\t<properties node_id="{ids[k]}" format="pickle">{blob}</properties>')
    out += ["\t</node_properties>", "</scheme>", ""]
    return "\n".join(out)


def import_images(key, folder, title, xy):
    """Import Images pointed at a folder under datasets/ (absolute path: re-point it on another machine)."""
    from orangewidget.utils.filedialogs import RecentPath

    path = str(BOOK / "datasets" / folder)
    return (key, "Import Images", title, xy,
            {"controlAreaVisible": True, "recent_paths": [RecentPath(path, "", "")], "__version__": 1})


def gan():
    nodes = [
        import_images("imp", "mnist/train", "Import Images: MNIST digits", (100, 150)),
        ("viewreal", "Image Viewer", "Image Viewer: real digits", (325, 20), {}),
        ("gan", "Train GAN", "Train GAN", (325, 150),
         {"controlAreaVisible": True, "img_height": 28, "img_width": 28, "channels_index": 0, "epochs": 15, "batch_size": 64}),
        ("gen", "Generate Images", "Generate Images", (550, 150), {"controlAreaVisible": True, "count": 16}),
        ("viewgen", "Image Viewer", "Image Viewer: generated digits", (775, 150), {}),
    ]
    links = [("imp", "viewreal"), ("imp", "gan"), ("gan", "gen"), ("gen", "viewgen")]
    note = ("Open Train GAN and click Train (about a minute for 15 epochs). Samples appear after every epoch. "
            "The dataset path is absolute: re-point Import Images on another machine.")
    return build(nodes, links, [(note, (550, 260, 300, 120))])


def diffusion():
    nodes = [
        import_images("imp", "mnist/train", "Import Images: MNIST digits", (100, 150)),
        ("diff", "Train Diffusion", "Train Diffusion", (325, 150),
         {"controlAreaVisible": True, "img_height": 28, "img_width": 28, "channels_index": 0, "epochs": 20, "noise_steps": 200,
          "use_classes": True}),
        ("gen", "Generate Images", "Generate Images", (550, 150), {"controlAreaVisible": True, "count": 10, "show_steps": True}),
        ("viewgen", "Image Viewer", "Image Viewer: generated digits", (775, 150), {}),
    ]
    links = [("imp", "diff"), ("diff", "gen"), ("gen", "viewgen")]
    note = ("Open Train Diffusion and click Train (about 4 minutes for 20 epochs; previews every 5 epochs). "
            "In Generate Images pick a Class, raise Guidance to push toward it, and tick Show denoising steps.")
    return build(nodes, links, [(note, (550, 260, 300, 120))])


def cnn():
    import json
    sys.path.insert(0, str(DEV))
    from orangecontrib.imagenets.widgets.ow_imagenet_builder import PREBUILT_MODELS

    nodes = [
        import_images("train", "mnist/train", "Import Images: MNIST train", (100, 150)),
        ("build", "Build ImageNet", "Build ImageNet: SimpleCNN 28x28", (325, 20),
         {"controlAreaVisible": True, "input_height": 28, "input_width": 28, "channels_index": 0,
          "model_config": json.dumps(PREBUILT_MODELS["SimpleCNN"])}),
        ("trainnet", "Train ImageNet", "Train ImageNet", (325, 150),
         {"controlAreaVisible": True, "epochs": 5, "batch_size": 32, "validation_percent": 20}),
        import_images("test", "mnist/test", "Import Images: MNIST test", (100, 300)),
        ("classify", "Classify Images", "Classify Images", (550, 225), {}),
        ("table", "Data Table", "Data Table: predictions", (775, 225), {}),
        ("mis", "Misclassified Images", "Misclassified Images", (775, 350), {"controlAreaVisible": True}),
        ("inspect", "Inspect ImageNet", "Inspect ImageNet", (550, 350), {"controlAreaVisible": True}),
    ]
    links = [("train", "trainnet"), ("build", "trainnet"), ("trainnet", "classify"), ("test", "classify"), ("classify", "table"),
             ("classify", "mis"), ("trainnet", "inspect"), ("train", "inspect")]
    note = ("Train ImageNet starts when both inputs arrive (5 epochs, about 10 seconds). "
            "Classify Images shows the accuracy on the test images; the Data Table lists every Prediction next to its category; "
            "Misclassified Images shows the confusion matrix; Inspect ImageNet shows the filters. The dataset paths are absolute.")
    return build(nodes, links, [(note, (550, 20, 300, 180))])


def explore():
    nodes = [
        import_images("imp", "mnist/train", "Import Images: MNIST train", (100, 150)),
        ("view", "Image Viewer", "Image Viewer", (325, 20), {}),
        ("table", "Data Table", "Data Table", (325, 150), {}),
        ("dist", "Distributions", "Distributions", (325, 280), {}),
    ]
    links = [("imp", "view"), ("imp", "table"), ("imp", "dist")]
    note = ("Chapter 1. Image Viewer shows the pictures, Data Table the columns Import Images made, "
            "Distributions the number of images per category. The dataset path is absolute: re-point Import Images.")
    return build(nodes, links, [(note, (520, 20, 300, 130))])


def tour():
    sets = [("mnist", "MNIST digits"), ("fashion-mnist", "Fashion-MNIST"), ("emnist-letters", "EMNIST letters"),
            ("cifar10", "CIFAR-10")]
    nodes, links = [], []
    for i, (folder, title) in enumerate(sets):
        y = 20 + i * 130
        nodes.append(import_images(f"imp{i}", f"{folder}/train", f"Import Images: {title}", (100, y)))
        nodes.append((f"view{i}", "Image Viewer", f"Image Viewer: {title}", (325, y), {}))
        links.append((f"imp{i}", f"view{i}"))
    note = ("Chapter 2. Build the datasets first with `make datasets`. The dataset paths are absolute: "
            "re-point each Import Images on another machine.")
    return build(nodes, links, [(note, (520, 20, 300, 110))])


def transfer():
    nodes = [
        import_images("train", "cifar10/train", "Import Images: CIFAR-10 train", (100, 150)),
        ("pre", "Pretrained ImageNet", "Pretrained ImageNet", (325, 20),
         {"controlAreaVisible": True, "base_index": 0, "n_classes": 10, "freeze": True}),
        ("trainnet", "Train ImageNet", "Train ImageNet", (325, 150),
         {"controlAreaVisible": True, "epochs": 5, "batch_size": 32, "validation_percent": 20}),
        import_images("test", "cifar10/test", "Import Images: CIFAR-10 test", (100, 300)),
        ("classify", "Classify Images", "Classify Images", (550, 225), {}),
        ("mis", "Misclassified Images", "Misclassified Images", (775, 225), {"controlAreaVisible": True}),
        ("viewer", "Image Viewer", "Image Viewer: mistakes", (1000, 225), {}),
    ]
    links = [("train", "pre"), ("pre", "trainnet"), ("train", "trainnet"), ("trainnet", "classify"),
             ("test", "classify"), ("classify", "mis"), ("mis", "viewer")]
    note = ("Chapter 7. The first use downloads about 2 MB of pretrained weights. Training takes about 35 seconds. "
            "Classify Images shows the test accuracy; click a cell in Misclassified Images to see those pictures. "
            "The dataset paths are absolute.")
    return build(nodes, links, [(note, (550, 20, 300, 140))])


def check(path):
    """Load the .ows with Orange's own reader; print node/link counts and errors."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Orange.canvas import config
    from orangecanvas.registry import WidgetRegistry
    from orangecanvas.scheme import Scheme
    from orangecanvas.scheme.readwrite import scheme_load
    from orangewidget.workflow.discovery import WidgetDiscovery
    reg = WidgetRegistry()
    WidgetDiscovery(reg).run(config.widgets_entry_points())
    scheme, errors = Scheme(), []
    with open(path, "rb") as f:
        scheme_load(scheme, f, registry=reg, error_handler=errors.append)
    print(f"{path}: {len(scheme.nodes)} nodes, {len(scheme.links)} links, errors={errors}")
    return not errors


# name -> (builder, output file under workflows/).  Add new workflows here.
WORKFLOWS = {"mnist-gan": (gan, "mnist-gan.ows"),
             "mnist-diffusion": (diffusion, "mnist-diffusion.ows"),
             "mnist-cnn": (cnn, "mnist-cnn.ows"),
             "mnist-explore": (explore, "mnist-explore.ows"), "datasets-tour": (tour, "datasets-tour.ows"),
             "cifar-transfer": (transfer, "cifar-transfer.ows")}

if __name__ == "__main__":
    # Use the dev checkout's widgets until a release includes them (orangecontrib is a namespace package).
    import orangecontrib
    orangecontrib.__path__ = [str(DEV / "orangecontrib")] + list(orangecontrib.__path__)
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(WORKFLOWS)
    for name in names:
        fn, fname = WORKFLOWS[name]
        dest = BOOK / "workflows" / fname
        dest.write_text(fn())
        print("wrote", dest)
        if "--check" in sys.argv:
            check(dest)
