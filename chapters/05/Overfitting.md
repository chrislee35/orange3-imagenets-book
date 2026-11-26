# Chapter 5: Overfitting and Regularization

## Learning Objectives

By the end of this chapter, you will be able to:

- Recognize overfitting in a training graph, and explain it in terms of memorizing versus learning.
- Use a validation split to find the epoch at which a network stops improving on new images.
- Measure how the amount of training data changes accuracy.
- Add dropout and augmentation to a network and measure whether each helps.
- Choose a fix for overfitting, and say why one remedy can work when another does not.

---

## 5.1 Memorizing Is Not Learning

A student who memorizes the answers to last year's exam can score 100% on last year's exam and fail this year's. A network can do the same. A network with 54,410 weights, shown 800 images, has far more adjustable numbers than images, so it can store the answers: "this exact pattern of pixels is a 3". Its **training accuracy** reaches 100%, but on a new picture of a 3 it is much less sure. That is **overfitting**.

You met the first sign in Chapter 3: the training accuracy rose above the validation accuracy. Here we make the problem large on purpose, by training for a long time on a small dataset.

The cure always comes from one of three directions:

1. **More data**, so there is less room to memorize.
2. **A less flexible network**: fewer weights, or layers that randomly drop information during training (dropout).
3. **Varied data**: augmentation, which presents each picture in many versions so the exact pixels can no longer be memorized.

---

## 5.2 Seeing It in the Graph

We train SimpleCNN for 30 epochs on only 100 images per class (1,000 images, of which 800 are used for training and 200 held out for validation). Here is what happens:

| Epoch | 1 | 6 | 11 | 16 | 21 | 26 |
|---|---|---|---|---|---|---|
| Training accuracy | 48% | 92.5% | 97.9% | 99.6% | **100%** | **100%** |
| Validation accuracy | 81% | 88.5% | 88.5% | 90.5% | 91.5% | 91.5% |

The training accuracy goes up to 100% and stays there. The validation accuracy stops improving at about 91%. The more telling curve is the **validation loss**:

| Epoch | 1 | 4 | 7 | 10 | 13 | 16 | 19 | 22 | 25 | 28 |
|---|---|---|---|---|---|---|---|---|---|---|
| Validation loss | 1.28 | 0.39 | 0.33 | 0.35 | 0.38 | 0.37 | 0.39 | 0.40 | 0.41 | 0.43 |

The loss is lowest around **epoch 7** and then slowly **rises**, while training continues to push the training loss toward zero. After epoch 7 the network is getting more confident, and more wrong in a few cases, on images it has not seen. A common rule is to stop training at the epoch where the validation loss is lowest (**early stopping**). With the graph in *Train ImageNet*, you can find that epoch by eye.

> 💡 Validation accuracy is a blunt tool: it changed little between epochs 7 and 26, yet the loss shows the network's confidence on unseen images getting worse. When the two disagree, trust the loss for *when* to stop and the test accuracy for *how good* the final model is.

---

## 5.3 Remedy 1: More Data

The most reliable cure is more data. We trained the same SimpleCNN for 20 epochs on different numbers of images per class and measured test accuracy (two samples of images for each size):

| Training images per class | Total | Test accuracy |
|---|---|---|
| 10 | 100 | 77.7% (80.1, 75.3) |
| 25 | 250 | 85.3% (84.3, 86.2) |
| 50 | 500 | 87.3% (86.6, 87.9) |
| 100 | 1,000 | 90.4% (90.1, 90.7) |
| 250 | 2,500 | 92.9% (92.5, 93.3) |
| 500 | 5,000 | 95.5% (95.6, 95.5) |

Every time the data doubles, the accuracy goes up, but by less and less: from 10 to 25 images per class we gain 7.6 points, from 250 to 500 only 2.6. The training accuracy was 100% every time. The network always memorized its training set, so what mattered was how many *different* examples it had to memorize.

The cost of more data is cost in collecting and labeling it. When you cannot get more, you make more out of what you have, which is Remedy 3.

---

## 5.4 Remedy 2: Dropout

**Dropout** is a layer with no weights that has a **rate**, for example 0.5. During training it sets a random half of the numbers passing through it to zero, with a different random half on every training step. The network cannot count on any single feature being present, so it has to spread what it knows across many. During classification the dropout layer does nothing.

In *Build ImageNet* the **Add Layers** list has *Dropout* (default rate **0.5**). Put it before the last *Dense* layer, after *Flatten*. It adds no weights.

Does it work? On 50 images per class and 20 epochs, three runs each:

| Network | Test accuracy per run | Average |
|---|---|---|
| plain | 86.6%, 88.3%, 88.5% | 87.8% |
| with dropout 0.5 | 87.0%, 87.7%, 88.7% | 87.8% |

**It made no difference.** This is a useful result to remember: dropout is a standard remedy, but it is not a universal one. The network here is small, with one convolution layer and one *Dense* layer, and dropout helps most in networks with big hidden *Dense* layers that have many weights to spare. Never assume a remedy works; measure it.

---

## 5.5 Remedy 3: Augmentation

Chapter 2 introduced *Augment Images*. Here it is as a remedy. For each of the 500 training images (50 per class) we made 2 extra copies, randomly zoomed (up to 20%) and rotated (up to 30 degrees), giving 1,500 images. The same experiment, three runs each:

| Network | Test accuracy per run | Average |
|---|---|---|
| plain (500 images) | 86.6%, 88.3%, 88.5% | 87.8% |
| plain, augmented (1,500 images) | 90.3%, 92.1%, 92.2% | **91.5%** |
| dropout, augmented | 92.5%, 92.3%, 93.2% | **92.7%** |

Augmentation raised the average by almost four points, as much as doubling the real data in the table above, and it did so for free. Dropout added a little on top (about one point), where alone it had added nothing. A network that cannot memorize exact pixels has more use for dropout.

---

## 5.6 The Widgets

| Widget | What it does | Key settings |
|---|---|---|
| *Data Sampling* | Takes a random part of the images | **Fixed proportion of data**, **Stratify sample** |
| *Train ImageNet* | Trains and plots training and validation curves | **Epochs**, **Validation split (%)** |
| *Build ImageNet* | Adds a *Dropout* layer | **rate** |
| *Augment Images* | Makes varied copies | **Zoom**, **Rotation**, **Augmentations per Image**, **Include the original images** |
| *Classify Images* | Reports the test accuracy | |

> ⚠️ Training makes its validation split after augmentation, so copies of the same picture can land on both sides. The validation numbers then look better than they should. Use the validation graph to see the *trend* only, and judge the final model by the test accuracy on the untouched `test` folder.

---

## Lab Exercise 5A: Watch It Overfit

**Goal:** Train long enough on little data to see the validation loss turn upward.

*Time: about 10 seconds.*

**Steps:**

1. From `workflows/mnist-cnn.ows`, add *Data Sampling* between *Import Images* (train) and *Train ImageNet*. Choose **Fixed proportion of data**, **20%**, and tick **Stratify sample**. That is 100 images per class.
2. In *Train ImageNet*, set **Epochs** to **30** and **Validation split (%)** to **20**.
3. Watch the graph. When training ends, find the epoch where the dashed red validation loss is lowest.
4. Classify the test images and write down the accuracy.

**Expected results:** the training accuracy reaches 100% by about epoch 21; the validation loss is lowest between epochs 5 and 15, and rises afterwards; the test accuracy is about **89%**.

**Questions:**

1. At which epoch would you stop? What would the test accuracy of the network stopped there be? (Try it: set **Epochs** to that number.)
2. The validation accuracy plateaus at about 91%, the test accuracy is 89%. Why are they not the same? *(Hint: the validation set has only 200 images, and was used to look at the curves.)*

---

## Lab Exercise 5B: How Much Data Do You Need?

**Goal:** Measure the gain from more data.

*Time: about 5 minutes.*

**Steps:** In *Data Sampling*, set **Fixed proportion of data** to **2%**, **5%**, **10%**, **20%**, **50%** and **100%** (10, 25, 50, 100, 250, and 500 images per class). For each, train 20 epochs with **Validation split (%)** at 0 and record the test accuracy. Plot accuracy against the number of images (a log scale on the images axis is more readable).

**Before you run it:** predict the accuracy at 10 images per class.

**Expected results:** see the table in section 5.3, within a few points. At 2% the numbers vary a lot between samples.

**Questions:**

1. Does the curve flatten? What would you expect from 1,000 images per class?
2. Pick a target accuracy of 90%. About how many images per class did you need?
3. Each doubling of data bought less. When is it better to stop collecting?

---

## Lab Exercise 5C: Dropout

**Goal:** Test whether dropout helps a small network.

*Time: 5 minutes.*

**Steps:**

1. Use **10%** in *Data Sampling* (50 images per class), no validation split, 20 epochs. Train *SimpleCNN* three times and record the test accuracy. This is the *plain* result.
2. In *Build ImageNet*, press **Dropout** in **Add Layers** (it goes between *Flatten* and the last *Dense*). Leave **rate** at 0.5. Train three times and record the test accuracy.
3. Try rates 0.25 and 0.75, three runs each.

**Expected result:** no clear difference at 0.5 (87.8% for both).

**Questions:**

1. Does dropout help here? How do you know that the differences are not just noise?
2. Why might dropout matter more in a network with a hidden *Dense* layer of 512 units?
3. Dropout is switched off when the network classifies. Why?

---

## Lab Exercise 5D: Augmentation Against Overfitting

**Goal:** Measure the gain from augmentation, alone and with dropout.

*Time: about 10 minutes.*

**Steps:**

1. Take the 10% sample from 5C, add *Augment Images* (**Zoom** and **Rotation** on, **Horizontal Flip** off, **Augmentations per Image** 2, **Include the original images** on), choose a save folder, and generate. You have 1,500 images.
2. Train SimpleCNN on the augmented table (no validation split, 20 epochs), three runs, and record the test accuracy.
3. Add the *Dropout* layer and repeat.

**Expected results:** plain about 88%; augmented about **91.5%**; augmented with dropout about **92.5%**.

**Questions:**

1. Why did augmentation help where dropout alone did not?
2. What changes would make a network overfit even **more** on this data? (More epochs? More weights?) Try one.
3. What would be a bad augmentation for letters? For photos of text?

---

## Comprehension Questions

1. A network reaches 99% training accuracy and 80% validation accuracy. What is the problem? Name three ways to address it.
2. The validation loss was lowest at epoch 7 and the network trained until epoch 30. What was wasted, and what was harmed?
3. In 5.3, why is the gain from 250 to 500 images per class smaller than from 25 to 50?
4. Explain dropout in two sentences. When is it active?
5. Dropout made no difference alone and one point with augmentation. Offer an explanation.
6. Why is it risky to choose the number of epochs by watching the validation graph and then quote the validation accuracy as your result?

---

## Key Terms

- **Overfitting**: doing much better on training images than on new ones, by memorizing.
- **Early stopping**: ending training when the validation loss is lowest.
- **Regularization**: any technique that fights overfitting (more data, dropout, augmentation).
- **Dropout**: randomly zeroing a share of the numbers in a layer, during training only.
- **Rate**: the share zeroed by a dropout layer.
- **Learning curve**: accuracy as a function of the amount of training data.

---

## Notes

- **Workflow for this chapter:** `workflows/mnist-cnn.ows`, with *Data Sampling* and *Augment Images* inserted as described.
- **Setup:** MNIST subsets drawn from the 500-per-class training folder; SimpleCNN, Adam, batch size 32; accuracy on the 1,000 test images. Each size or remedy in the tables is a different random sample of images.
- **Numbers quoted:**
  - 5.2: 100 images per class, 30 epochs, 20% validation (800 training): final test accuracy 89.2%; validation loss minimum at epoch 7.
  - 5.3: 20 epochs, no validation: 10/class 80.1% and 75.3%; 25/class 84.3% and 86.2%; 50/class 86.6% and 87.9%; 100/class 90.1% and 90.7%; 250/class 92.5% and 93.3%; 500/class 95.6% and 95.5%.
  - 5.4 and 5.5 (50 per class, 20 epochs, three samples): plain 86.6, 88.3, 88.5; dropout 87.0, 87.7, 88.7; augmented 90.3, 92.1, 92.2; augmented with dropout 92.5, 92.3, 93.2.
- **Variance:** with this little data, runs differ by two points or more. Only compare averages of several runs.
