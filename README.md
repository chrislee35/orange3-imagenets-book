# Training ImageNets with Orange: A Lab-Based Approach

![](imgs/Title%20Image.png)

## Course Overview

This workbook teaches how neural networks learn from images, using Orange Data Mining and the **orange3-imagenets** add-on. You will build convolutional networks layer by layer, train them on well-known datasets such as MNIST, watch them overfit and fix it, and finish by building simplified versions of a **GAN** and a **diffusion model** (the idea behind Stable Diffusion) that generate new images.

### Topics Covered

- **Images as data**: pixels, image tables, preprocessing and augmentation
- **Convolutional networks**: filters, pooling, depth, normalization, residual blocks
- **Training**: loss, accuracy, overfitting, dropout
- **Reusing and inspecting models**: saving, transfer learning from a pretrained network, filters, and confusion matrices
- **Generative models**: GANs and diffusion, in miniature
- **A capstone project** that puts it together

### Datasets

- **MNIST**: handwritten digits
- **Fashion-MNIST**: clothing items
- **EMNIST**: handwritten letters
- **CIFAR-10**: small color photos

## Learning Outcomes

By the end of this workbook, you will be able to:

- Prepare image data for training in Orange.
- Build, train, save, and apply a CNN with the ImageNet widgets.
- Explain what each layer type does and what changes when you add, remove, or resize it.
- Diagnose and reduce overfitting.
- Explain, and run in miniature, how GANs and diffusion models generate images.

## Course Requirements

- **Prerequisites**: Basic familiarity with Orange (widgets, links, data tables).
- **Tools**: Orange 3.39+ with the Orange3-ImageAnalytics add-on and the orange3-imagenets add-on. The widgets *Pretrained ImageNet*, *Inspect ImageNet*, *Misclassified Images*, and the GAN and diffusion widgets are newer than the version in the Add-ons dialog, so install the add-on from its GitHub repository (`pip install git+https://github.com/chrislee35/orange3-imagenets.git`) until a release includes them.
- **Datasets**: `make datasets` builds the image folders and needs a Python environment with Keras (TensorFlow) and torchvision. The first run downloads the datasets (CIFAR-10 about 170 MB, EMNIST about 560 MB).
- **Hardware**: everything runs on a CPU; labs that take more than about five minutes are marked *Optional*. The diffusion labs are the slowest (about 4 minutes for 20 epochs).

## Authors

Chris Lee

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. See [LICENSE.md](LICENSE.md).

---
