# Chapter 10: Capstone Project

## Learning Objectives

By the end of this chapter, you will be able to:

- Plan and run a small image-learning project on your own, from a question to a written conclusion.
- Choose a baseline and improve it with one change at a time, predicting each result before you see it.
- Compare models fairly, with the same data, repeated runs, and a test set used once.
- Explain a model's mistakes with a confusion matrix and the mistaken images.
- Report what worked, what did not, and how sure you are.

---

## 10.1 What This Chapter Is

The earlier chapters told you which widgets to connect and what to expect. This one does not. You pick a problem, design the experiments, and decide what the results mean. Nothing in it is new; it all comes from Chapters 1 to 9. The work is in deciding what to try and in being honest about the numbers.

Choose **one** of the projects in section 10.3, and spend about two to four hours on it. A reference example, with real numbers, follows in 10.4 so you can see what a good result looks like. Do not copy it: use it to check the standard.

---

## 10.2 Method: Six Rules

These rules are the lessons of the nine chapters. A project that follows them is good whatever its accuracy.

1. **Start with a baseline.** Train the simplest network that could work (*SimpleCNN*, five epochs) before anything else. Every later result is a comparison with it. Without a baseline, "85%" means nothing.
2. **Change one thing at a time**, and **predict** the result first. Write the prediction down before you train. If you change three things at once you cannot say which one mattered.
3. **Keep the test set for the end.** Use the validation split (and *Train ImageNet*'s curves) while you decide. Evaluate on the `test` folder only for the models you report. Never augment it, and never tune on it.
4. **Repeat before you believe.** Runs differ by a point or two, more with small data. Train each important model at least twice, and call a difference real only if it is clearly larger than the spread between runs (Chapters 4 and 5).
5. **Look at the mistakes.** An accuracy number tells you how much is wrong, not why. Use *Misclassified Images*, click the biggest red cells, and look at the pictures.
6. **Count the cost.** Note the training time and the number of weights beside every accuracy. A model that is a half point better and five times slower is not necessarily better.

---

## 10.3 Choose a Project

### Project A: Classify a dataset

Pick one of **Fashion-MNIST**, **EMNIST letters**, or **CIFAR-10** (not MNIST, which is too easy). Build a classifier, and try at least **four** variations from this list, one at a time.

| Idea | Chapter | Widget settings |
|---|---|---|
| Depth: add a convolution layer | 4, 6 | *Build ImageNet*: *Conv2D*, *MaxPooling2D* |
| Width: more or fewer filters, larger kernel | 3 | *Conv2D* **filters**, **kernel_size** |
| BatchNorm | 6 | *BatchNormalization* after each convolution |
| Dropout | 5 | *Dropout* before the last *Dense* |
| Augmentation | 2, 5 | *Augment Images*; no flips for letters |
| More data | 5 | the `--per-class` option of `make datasets` |
| Smaller images | 2 | *Preprocess Images* |
| Transfer learning | 7 | *Pretrained ImageNet*, frozen |
| Learning rate, epochs | 3, 7 | *Train ImageNet* |

### Project B: Your own images

Photograph or collect two to five kinds of object (leaves, coins, mugs, the faces of your pets, fruit at different ripeness), at least **60 pictures per class**, in a folder per class (`mydata/ripe/`, `mydata/unripe/`). Use pictures you took or may redistribute; do not put anyone's face in the data without their permission. Keep 20% of every class apart in a separate `test` folder at the very start, and do not look at it again until the end. Reduce the pictures with *Preprocess Images* (for example to 96 x 96). Do the same experiments as Project A. With so little data, expect transfer learning (Chapter 7) to be the baseline to beat. Lab 7F is a worked example with 42 pictures per class: a network from scratch scored about 58% and a frozen pretrained one 82% to 88%. If you use pictures from other people, keep a file that records the author and licence of each, as `datasets/animals/ATTRIBUTION.csv` does.

### Project C: Generate something

Pick a dataset (for example, Fashion-MNIST for clothes, or one class of CIFAR-10). Train a *GAN* **and** a *Diffusion* model on it, with the same images. Produce 64 samples from each, and write a short comparison: which looks better, which varies more, how long each took, and where each fails. To measure how good the samples are, classify them with a classifier you trained in Project A and say how confident it is on real images and on generated ones (Chapter 9's method). Be honest: at this size, neither model will be impressive.

---

## 10.4 A Reference Example

Here is what Project A looks like for **Fashion-MNIST** and **EMNIST letters** (500 training images per class: 5,000 and 13,000 images; accuracy on the 100 test images per class; single runs).

### Fashion-MNIST

| Model | Weights | Epochs | Test accuracy | Time |
|---|---|---|---|---|
| SimpleCNN (baseline) | 54,410 | 5 | 84.1% | 3 s |
| 2 conv | 34,826 | 10 | 85.0% | 8 s |
| 2 conv + BatchNorm | 35,210 | 10 | 83.7% | 14 s |
| 2 conv + BatchNorm + Dropout 0.5 | 35,210 | 10 | 83.5% | 14 s |
| Pretrained MobileNet, frozen | 423,018 (12,810 trained) | 5 | **86.0%** | 22 s |

### EMNIST letters

| Model | Weights | Epochs | Test accuracy | Time |
|---|---|---|---|---|
| SimpleCNN (baseline) | 140,954 | 5 | 82.9% | 7 s |
| 2 conv | 60,442 | 10 | 85.3% | 19 s |
| 2 conv + BatchNorm | 60,826 | 10 | 84.1% | 33 s |
| 2 conv + BatchNorm + Dropout 0.5 | 60,826 | 10 | **87.4%** | 33 s |
| Pretrained MobileNet, frozen | 443,514 (33,306 trained) | 5 | 82.8% | 53 s |

### What a good report says about these

- **There is no winner across both datasets.** Transfer learning is best on Fashion-MNIST and no better than the baseline on letters, where the pictures (thin white strokes on black) are far from the photographs the network was pretrained on. BatchNorm helped CIFAR-10 in Chapter 6 and did not help here. Dropout gave the best score on letters (about 3 points over the same network without it) and made no difference on Fashion-MNIST. This is why we test an idea on the data in front of us, instead of trusting a rule.
- **Most differences are small.** On Fashion-MNIST all five models lie between 83.5% and 86.0%. With run-to-run variation of a point or two, only the pretrained model is arguably better than the baseline, and even that needs a second run. A report that says "dropout lowered the accuracy by 0.2 points" is claiming something the data cannot support.
- **The mistakes make sense.** In Fashion-MNIST the five most common confusions of the 2-conv model with BatchNorm (out of 1,000 test images) are *pullover* taken for *shirt* (19), *shirt* for *tshirt* (18), *tshirt* for *shirt* (16), *coat* for *shirt* (16), and *tshirt* for *dress* (10). These are garments that people also find similar. In EMNIST letters the worst by far is *i* taken for *l* (44 of 100), then *a* for *h* (14), *l* for *i* (13), *g* for *q* (13), and *b* for *h* (13). The letter *i* without its dot and an *l* are almost the same stroke, and capital and lowercase are merged. No change to the network will fix that; only more informative data would.
- **The cost matters.** On Fashion-MNIST the 2-conv model scores about a point above the baseline in 8 seconds; BatchNorm doubles the time and does not help.

---

## 10.5 The Deliverables

Hand in (or save, if you work alone) the following:

1. **The workflows**, as `.ows` files, one per model you report, named so that each can be matched to a row in your table.
2. **A report of one or two pages**, with:
   - **Question and data.** What are you classifying or generating, how many images, and from where (and the license, for your own data).
   - **Baseline.** The network, its settings, and its accuracy, with the training and validation curves.
   - **Experiments.** A table with one row per model: what you changed, your prediction, the weights, the time, and the test accuracy of two runs.
   - **Error analysis.** A confusion matrix (or the part of it that matters), three or four pictures of mistakes, and your explanation.
   - **Conclusion.** What worked, what did not, what you would try next, and how sure you are. State what the numbers can and cannot show.
3. **For Project C:** a figure of 64 samples from each generator, and your comparison.

---

## 10.6 How It Is Judged

| Criterion | Points |
|---|---|
| A baseline, and every experiment compared with it | 15 |
| One change at a time, with a written prediction | 15 |
| Fair evaluation: the test set used only at the end, and each key model run at least twice | 20 |
| Error analysis with a confusion matrix and pictures | 20 |
| Honest conclusions: claims no stronger than the data, with the spread of the runs stated | 20 |
| Clarity of the workflows and the report | 10 |

Accuracy itself is **not** a criterion. A project with a modest accuracy, run carefully and reported honestly, scores better than a high accuracy that cannot be repeated.

---

## 10.7 Common Pitfalls

- **Judging by the training accuracy.** It nearly always goes up. Look at validation and test.
- **Tuning on the test set.** If you choose the model that scores best on the test images, and report that score, it is optimistic. Decide with the validation split, and report the test once.
- **Augmenting before the split.** Copies of one picture on both sides leak the answer (Chapter 2). Judge augmentation on the held-out test folder.
- **One run.** A difference of one point is within the noise.
- **A mismatched input size.** Build ImageNet starts at 224 x 224 x 3. Set it to the data. The last *Dense* layer must have as many units as there are classes.
- **Forgetting the cost.** Report the time.
- **Reading a generated image as proof.** One good sample says nothing about the variety; look at many.

---

## Lab Exercise 10A: Project A, Step by Step

**Goal:** Carry out Project A with the method of 10.2.

*Time: 2 to 4 hours.*

**Steps:**

1. Build the data with `make datasets` (use `python utils/export_datasets.py fashion-mnist` for just one). Import the `train` and `test` folders.
2. Build the baseline: *SimpleCNN*, **Input Images** set to the dataset's size (28 x 28 x 1, or 32 x 32 x 3), the last *Dense* layer set to the number of classes, 5 epochs, **Validation split (%)** 20. Record the accuracy, the weights, and the time. Run it twice.
3. Choose four ideas from 10.3. For each, **write your prediction** (better or worse, by how much, and why), build it, run it twice, and record the results in your table.
4. Choose the best model *by validation accuracy and cost*, not by the test. Train it once more, classify the `test` folder with *Classify Images*, and open *Misclassified Images*.
5. Click the three largest off-diagonal cells, and look at the pictures. Write what you see.
6. Write the report of 10.5.

**Questions to answer in your report:**

1. Which of your predictions were wrong? What did you misjudge?
2. Which of your changes made a difference larger than the spread between runs?
3. What is the most common mistake, and could more training or a better network fix it? Why or why not?

---

## Lab Exercise 10B: Project C, Step by Step

**Goal:** Compare a GAN and a diffusion model on the same data.

*Time: about 2 hours, most of it waiting for training.*

**Steps:**

1. Import one dataset (for instance `datasets/fashion-mnist/train`).
2. Train *Train GAN* for 30 epochs (about a minute) and *Train Diffusion* for 30 epochs with **Condition on the class** ticked (about 6 minutes). Save both models.
3. Generate 64 images from each. For the diffusion model, generate 6 images per class with guidance 2.
4. Train a classifier for the same data (Project A). Classify the generated images from each model, and record the share of the diffusion images that match the requested class, and the classifier's mean confidence on each set.
5. Compare the GAN and the diffusion samples in a short report: the quality, the variety, the time, and the failure of each.

**Questions to answer in your report:**

1. Which model produced more variety? How can you tell?
2. Does the classifier find the samples recognizable? How does that compare with real images?
3. What would you change to improve the weaker one: more epochs, more filters, or more data? Try one, and report the result.

---

## Comprehension Questions

1. Why is a baseline necessary before improving a model?
2. You change the number of filters, add dropout, and augment the data in one run, and the accuracy rises by 3 points. What have you learned?
3. Why must the test set be left until the end?
4. Two models score 86.0% and 85.0%. What would you do before saying that the first is better?
5. Give an example from this book where an idea that helped on one dataset did not help on another.
6. Which of the six rules in 10.2 did you find hardest to follow, and why?

---

## Key Terms

- **Baseline**: the simplest reasonable model, used as the point of comparison.
- **Ablation / controlled experiment**: changing one thing at a time to learn its effect.
- **Held-out test set**: images used once, at the end, to report the final accuracy.
- **Error analysis**: studying the mistakes to understand what the model gets wrong and why.
- **Reproducibility**: another person being able to get the same results from your workflows and report.

---

## Notes

- **Workflows:** you make your own. The starting points are `workflows/mnist-cnn.ows` (Project A), `workflows/cifar-transfer.ows` (transfer learning), and `workflows/mnist-gan.ows` with `workflows/mnist-diffusion.ows` (Project C).
- **Setup of the reference example:** 500 training and 100 test images per class (Fashion-MNIST 5,000 and 1,000; EMNIST letters 13,000 and 2,600); Adam, batch size 32; 20% stratified validation; the frozen MobileNet was the *small* 96 x 96 one, with the gray images repeated in three colors; test accuracy from a single run each.
- **Numbers quoted:**
  - Fashion-MNIST: 84.1%, 85.0%, 83.7%, 83.5%, 86.0%.
  - EMNIST letters: 82.9%, 85.3%, 84.1%, 87.4%, 82.8%.
  - Confusions: the most common for the 2-conv model with BatchNorm (Fashion-MNIST) and for the same network on letters.
- **Variance:** every difference in these tables is within a few points, and none was repeated. They are a standard to compare your work with, and not results to cite.
- **Licenses and ethics:** Fashion-MNIST (MIT), EMNIST (NIST), and CIFAR-10 (Krizhevsky, 2009) are used as described in Chapter 1. For your own pictures, collect only what you may use, and ask permission before including people.
