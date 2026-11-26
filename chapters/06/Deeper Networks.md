# Chapter 6: Deeper Networks

## Learning Objectives

By the end of this chapter, you will be able to:

- Build deeper networks for color photographs and measure how much depth helps on CIFAR-10.
- Explain what batch normalization does to the numbers inside a network, and measure its effect.
- Explain what a skip connection (residual block) is, and build a network with one.
- Read a training graph that shows a deep network memorizing its data, and decide what to do about it.
- Judge when a design idea pays off, given its cost in training time.

---

## 6.1 A Harder Problem

On MNIST, two convolution layers reach 96% to 97%, and there is little left to win. CIFAR-10 is different. In Chapter 2 SimpleCNN scored only about 50% on it, and a network that guessed would score 10%. Photographs vary in color, background, size, and pose, so the network has to learn much more. This chapter uses CIFAR-10 to see what deeper and better-built networks can do.

All numbers in this chapter use the 5,000-image CIFAR-10 subset (500 training images per class), 20 epochs, and the 1,000 test images. With this little data the scores are modest. The *comparisons* are what we are after.

---

## 6.2 Going Deeper

Chapter 4 showed that stacking convolutions raised MNIST accuracy. We try the same on CIFAR-10, with three networks:

| Network | Layers | Weights |
|---|---|---|
| **1 conv** | Conv 32, Pool, Dense | 72,906 |
| **2 conv** | Conv 32, Pool, Conv 64, Pool, Dense | 42,442 |
| **3 conv** | Conv 32, Pool, Conv 64, Pool, Conv 128, Pool, Dense | 113,738 |

All convolutions are 3 x 3 with relu. In the 3-conv network they use **same** padding, which adds a one-pixel border of zeros so that the map keeps its size (32 x 32 stays 32 x 32), and pooling does the shrinking. Counting the 3-conv weights: (3 x 3 x 3 + 1) x 32 = 896, then 18,496, then (3 x 3 x 64 + 1) x 128 = 73,856, and finally 4 x 4 x 128 = 2,048 numbers into the *Dense* layer, (2,048 + 1) x 10 = 20,490. The sum is 113,738.

Results (two runs each, test accuracy):

| Network | Test accuracy | Training accuracy | Time |
|---|---|---|---|
| 1 conv | 48.9%, 48.7% | 88% | 11 s |
| 2 conv | 54.7%, 52.2% | 83% to 85% | 18 s |
| 3 conv | 54.4%, 52.5% | 94% | 31 s |

Two convolutions help by about four points. A third adds nothing on the test images, but raises the training accuracy to 94%. The network is memorizing, not learning more.

---

## 6.3 Batch Normalization

Inside a deep network, the numbers that flow from layer to layer can drift: one layer's outputs may be tiny and the next layer's huge. Each layer has to adapt to whatever the layer before it happens to produce, and that slows learning down.

**Batch normalization** (BatchNorm) steadies the numbers. For each channel, over the batch of images being processed, it subtracts the mean and divides by the standard deviation, so the numbers have a mean of 0 and a spread of 1. Then it applies a learned scale and shift, so the network can undo the normalization where that helps.

A worked example with four numbers, 2, 4, 6, and 8:

- The mean is (2 + 4 + 6 + 8) / 4 = 5.
- The standard deviation is the square root of ((9 + 1 + 1 + 9) / 4) = 2.24.
- The normalized values are (2 - 5) / 2.24, (4 - 5) / 2.24, and so on: **-1.34, -0.45, 0.45, 1.34**.

Whether the inputs were 2, 4, 6, 8 or 2,000, 4,000, 6,000, 8,000, the layer after BatchNorm sees the same numbers.

You place a *BatchNormalization* layer after each convolution. It has 4 numbers per channel: two are learned, and two are running averages (the mean and variance seen during training) that it uses instead of the batch when you classify one image at a time. Adding BatchNorm to the 3-conv network adds 4 x (32 + 64 + 128) = 896 weights, which is why it has 114,634.

| Network | Test accuracy | Training accuracy | Time |
|---|---|---|---|
| 3 conv | 54.4%, 52.5% | 94% | 31 s |
| 3 conv + BatchNorm | **60.9%, 57.5%** | **100%** | 50 s |

BatchNorm raised the test accuracy by about **five to six points** for 1,000 extra weights, and made training about 60% slower. It is the cheapest large gain in this book so far.

---

## 6.4 Skip Connections

The deeper a network, the harder it is to train. The signal that tells each layer how to change has to travel backwards through every layer above it, and it can fade away on the way. In very deep plain networks (twenty layers or more) the deeper network can even do *worse* than a shallower one, and not because of overfitting: its training accuracy is also worse.

The idea that fixed this is the **skip connection**. Each block does not replace its input with something new. It computes a small *change* and adds it to the input:

> **output = input + change(input)**

If the best thing a block can do is nothing, it only needs to make the change zero. The input also reaches the output unharmed, and the training signal has a direct path back through the addition. A block with a skip connection is a **residual block**, and a network built from them is a **ResNet** (residual network). ResNets made networks of 50 to 150 layers trainable, and nearly every modern image network uses the idea.

*Build ImageNet* offers a **ResidualBlock** layer. Inside it are two 3 x 3 convolutions, each followed by BatchNorm, and the addition of the input at the end. A *Sequential* network cannot branch, so the branch lives inside the layer. You only choose the number of **filters**, and when that number differs from the input's depth, the skip passes through a small 1 x 1 convolution so that the two can be added.

The **MiniResNet** preset is built this way:

| Layer | Output shape |
|---|---|
| Rescaling | 32 x 32 x 3 |
| Conv2D 32, same | 32 x 32 x 32 |
| ResidualBlock 32 | 32 x 32 x 32 |
| MaxPooling2D | 16 x 16 x 32 |
| ResidualBlock 64 | 16 x 16 x 64 |
| MaxPooling2D | 8 x 8 x 64 |
| Flatten | 4,096 |
| Dense 10 | 10 |

It has 118,666 weights.

| Network | Weights | Test accuracy | Training accuracy | Time |
|---|---|---|---|---|
| 3 conv + BatchNorm | 114,634 | 60.9%, 57.5% | 100% | 50 s |
| MiniResNet | 118,666 | 58.6%, 57.5% | 100% | 150 s |

**The skip connections did not help here.** MiniResNet is no better than the plain network that has BatchNorm, and it takes three times as long. This is an honest result, and an instructive one. Residual blocks pay off when a network is so deep that plain stacking fails: this network has 6 convolutions in all. Do not expect a trick designed for 50 layers to help a 6-layer network, and always measure.

> 💡 A first version of the MiniResNet preset ended in *GlobalAveragePooling2D* instead of *Flatten* and scored only 49% to 51%. The head of a network matters as much as its body. Averaging each map to one number threw away where things were in the picture, and with only 5,000 images the network had too little data to learn a replacement.

---

## 6.5 What Is Limiting Us Now?

The 3-conv network with BatchNorm and MiniResNet both reach 100% training accuracy and about 58% to 61% test accuracy. That is a gap of 40 points, the signature of overfitting from Chapter 5: the networks have enough weights to memorize 4,000 images, and the images do not vary enough for them to generalize. Making the network *better built* did not close the gap. The remaining remedies are the ones from Chapter 5, more and more varied data, and a new one, in Chapter 7: start from a network that has already learned to see.

---

## 6.6 The Widgets

| Widget | What it does | Key settings |
|---|---|---|
| *Build ImageNet* | Builds the network | **Input Images** (32 x 32, 3 channels for CIFAR-10); layers *Conv2D*, *BatchNormalization*, *MaxPooling2D*, *ResidualBlock*, *Flatten*, *Dense*; **Clear**; the **MiniResNet** preset |
| *Train ImageNet* | Trains it | **Epochs**, **Validation split (%)** |
| *Classify Images* | Reports the test accuracy | |
| *Misclassified Images* | Confusion matrix of the mistakes | click a cell to see those images |

In the *Conv2D* box, the **padding** field takes `valid` (no padding, the default) or `same`.

> ⚠️ **Time.** The networks of this chapter take 10 seconds to 2.5 minutes to train on a laptop CPU. The *Orange window pauses while training*. Labs 6B and 6C are the longest; run them while you read.

---

## Lab Exercise 6A: Depth on CIFAR-10

**Goal:** Measure what extra convolution layers do for photographs.

*Time: about 1 minute per network.*

**Before you run it:** SimpleCNN scored about 50%. What accuracy do you expect from three convolution layers?

**Steps:**

1. Import `datasets/cifar10/train` and `datasets/cifar10/test` (build them first with `make datasets`; see Chapter 1).
2. In *Build ImageNet*, set **Height** 32, **Width** 32, **Channels** *3 (color)*. Press **Clear**.
3. Build *1 conv* by pressing, in this order, **Conv2D** (filters 32), **MaxPooling2D**, **Flatten**. The last *Dense* layer already has 10 units. Check **Weights: 72,906**.
4. Connect it to *Train ImageNet* with the train images, set **Epochs** to **20**, and **Validation split (%)** to **20**. Classify the test images and note the accuracy.
5. Build *3 conv*: **Clear**, then **Conv2D** (set **padding** to `same`), **MaxPooling2D**, **Conv2D** (filters **64**, padding `same`), **MaxPooling2D**, **Conv2D** (filters **128**, padding `same`), **MaxPooling2D**, **Flatten**. Check **Weights: 113,738**. Train and classify.

**Expected results:** about 49% for *1 conv* and about 53% to 54% for *3 conv*; the training accuracy of *3 conv* is about 94%.

**Questions:**

1. Compare the training and validation graphs of the two networks. Which one overfits more?
2. How much did you gain, and how much did the weights grow? Is it worth it?

---

## Lab Exercise 6B: Batch Normalization

**Goal:** Add *BatchNormalization* to the 3-conv network and measure the gain.

*Time: about 1 minute per run.*

**Steps:**

1. Rebuild the *3 conv* network of 6A with **Clear**, pressing the buttons in this order: **Conv2D** (padding `same`), **BatchNormalization**, **MaxPooling2D**, **Conv2D** (filters 64, `same`), **BatchNormalization**, **MaxPooling2D**, **Conv2D** (filters 128, `same`), **BatchNormalization**, **MaxPooling2D**, **Flatten**. Check **Weights: 114,634**.
2. Train 20 epochs and classify the test images. Repeat once.

**Before you run it:** 896 extra weights will not make the network any more flexible. What do you expect?

**Expected results:** about **58% to 61%**, five to six points better than without BatchNorm, with a training accuracy of 100%.

**Questions:**

1. If BatchNorm adds hardly any weights, why does it help? *(Hint: look at how fast the training graph rises.)*
2. The training accuracy reached 100%. Is this good news?
3. Find the epoch with the lowest validation loss, as you did in Chapter 5. After it, does anything improve?

---

## Lab Exercise 6C: A Residual Network

**Goal:** Build a network with skip connections and compare it with the plain one. (Optional: about 2.5 minutes per run.)

**Steps:**

1. In *Build ImageNet*, choose the **MiniResNet** preset. Check **Weights: 118,666**.
2. Train 20 epochs and classify the test images.
3. Compare accuracy and time with the *3 conv + BatchNorm* network of 6B.

**Expected results:** about **58% to 59%**, no better than 6B, and about three times slower (about 150 seconds).

**Questions:**

1. Did skip connections help? Why might they matter more for a network with 50 layers?
2. Delete one *ResidualBlock* and replace it with a *Conv2D*. What happens to the accuracy and the time?
3. Use *Misclassified Images* on the result. Which pairs of classes are confused most? Are they the pairs you would confuse?

---

## Comprehension Questions

1. What does BatchNorm do to the numbers 10, 20, 30, 40? *(Mean 25, standard deviation 11.2.)*
2. Why does BatchNorm keep running averages, and when are they used?
3. What is the output of a residual block, in one line? What happens if the block's convolutions output all zeros?
4. In 6.2 the third convolution layer raised the training accuracy and not the test accuracy. What does that mean?
5. Why did MiniResNet not beat the plain network with BatchNorm, and what would you try to make it do so?
6. Both the 3-conv + BatchNorm network and MiniResNet reached 100% training accuracy. Name two remedies from Chapter 5 that you would try next.

---

## Key Terms

- **Same / valid padding**: padding a convolution with zeros so the map keeps its size, or not padding it.
- **Batch normalization**: rescaling each channel's numbers to mean 0 and spread 1 inside the network.
- **Skip connection**: adding a block's input to its output.
- **Residual block / ResNet**: a block with a skip connection, and a network built from such blocks.
- **Vanishing signal**: the training signal fading as it travels through many layers.
- **Head**: the last layers of a network, which turn features into class scores.

---

## Notes

- **Workflow for this chapter:** edits of `workflows/mnist-cnn.ows`, pointed at `datasets/cifar10/{train,test}` with the input set to 32 x 32 x 3.
- **Setup:** CIFAR-10 subset (500 training and 100 test images per class), Adam, batch size 32, 20 epochs, 20% stratified validation, test accuracy on the 1,000 test images. Two runs of each network.
- **Numbers quoted (test accuracy, two runs; time for one run):**
  - 1 conv 48.9%, 48.7% (11 s); 2 conv 54.7%, 52.2% (18 s); 3 conv 54.4%, 52.5% (31 s).
  - 3 conv + BatchNorm 60.9%, 57.5% (50 s); MiniResNet 58.6%, 57.5% (about 150 s).
  - The earlier MiniResNet with *GlobalAveragePooling2D*: 51.0%, 49.3%.
- **Variance:** two runs differ by up to three points here, so the 6B gain is larger than the noise but 6C against 6B is not a difference.
