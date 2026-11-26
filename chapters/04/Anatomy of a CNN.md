# Chapter 4: Anatomy of a CNN

## Learning Objectives

By the end of this chapter, you will be able to:

- Track the shape of the data and the number of weights through a stack of layers, and predict both before building.
- Explain what stacking convolution layers does and why a deeper network can see larger patterns.
- Build a deeper network in *Build ImageNet* and measure the gain.
- Compare the classic LeNet with SimpleCNN, and say what differs.
- Judge from an experiment whether a design choice (activation, pooling, an extra *Dense* layer) really matters.

---

## 4.1 Shapes and Weights, Layer by Layer

In Chapter 3 you counted the weights of one network. This chapter turns that count into a skill, because you will need it to design a network without trial and error. Two rules cover every layer in this book.

**Shape.** A convolution without padding trims 2 pixels from each side with a 3 x 3 filter (4 pixels in all), and 4 with a 5 x 5 filter. Its depth becomes the number of filters. Pooling by 2 halves the height and width, rounding down.

**Weights.** A *Conv2D* layer has (kernel height x kernel width x input depth + 1) x filters weights. A *Dense* layer has (inputs + 1) x units. Pooling, *Flatten*, and *Rescaling* have none.

Here is a deeper network, which we will call *TwoConv*, worked out by hand:

| Layer | Output shape | Weights | Count |
|---|---|---|---|
| Input (Rescaling) | 28 x 28 x 1 | 0 | |
| Conv2D, 32 filters, 3 x 3 | 26 x 26 x 32 | 320 | (3 x 3 x 1 + 1) x 32 |
| MaxPooling2D 2 | 13 x 13 x 32 | 0 | |
| Conv2D, 64 filters, 3 x 3 | 11 x 11 x 64 | 18,496 | (3 x 3 x 32 + 1) x 64 |
| MaxPooling2D 2 | 5 x 5 x 64 | 0 | 11 / 2 = 5.5, rounded down |
| Flatten | 1,600 | 0 | 5 x 5 x 64 |
| Dense, 10 units | 10 | 16,010 | (1,600 + 1) x 10 |
| **Total** | | **34,826** | |

Notice that TwoConv has **fewer** weights than SimpleCNN (34,826 against 54,410) even though it has an extra layer. SimpleCNN's single *Dense* layer receives 5,408 numbers. TwoConv's pools them down to 1,600 first. Depth does not have to mean size.

---

## 4.2 Why Stack Convolutions?

One 3 x 3 filter sees a 3 x 3 patch of the image. That is enough for a short edge or a corner, and not enough for a loop or a crossing. A second convolution layer looks at the *feature maps* of the first, so each of its 3 x 3 windows covers a block of feature-map positions, and each of those covered a patch of the image.

Count how much of the original image one number in each layer depends on (its **receptive field**):

| After this layer | Receptive field in the image |
|---|---|
| first Conv2D (3 x 3) | 3 x 3 pixels |
| first MaxPooling2D | 4 x 4 pixels |
| second Conv2D (3 x 3) | 8 x 8 pixels |

An 8 x 8 window is large enough to hold a whole loop of an 8 or a 6. This is the usual picture of a deep CNN: **early layers find edges, middle layers combine them into parts (curves, corners, loops), and late layers combine parts into whole objects.** Each layer works on the output of the one before, so the patterns get bigger and more abstract.

The price of depth is training time (more layers to compute) and, as Chapter 6 shows, difficulty of training when the network gets very deep.

---

## 4.3 LeNet: A Network from 1998

*LeNet* is one of the first successful CNNs. It was built to read handwritten digits on bank cheques. It is a preset in *Build ImageNet*, and it differs from SimpleCNN in four ways:

| | SimpleCNN | LeNet |
|---|---|---|
| Convolutions | 1 (32 filters, 3 x 3) | 2 (6 filters then 16, both 5 x 5) |
| Pooling | max | average |
| Activation | relu | tanh |
| Hidden *Dense* layers | none | 120 units, then 84 |
| Weights (28 x 28 x 1 input) | 54,410 | 44,426 |

**Average pooling** keeps the mean of each block instead of the maximum. **tanh** squashes a number into the range -1 to 1, and was the usual choice before relu, which is simpler and trains faster in deep networks.

---

## 4.4 Do the Details Matter?

Designing a network involves many small choices: relu or tanh, max or average pooling, one more layer or not. Beginners often assume each matters a great deal. The honest answer is that the *structure* (convolution, depth) matters most, and most single details change the accuracy by less than the run-to-run noise. The labs below let you see that for yourself.

> 💡 To compare two designs fairly, train both with the same data, the same number of epochs, and the same test images, and repeat each at least twice. A difference smaller than the gap between two runs of the *same* network (about a point) is not a finding.

---

## 4.5 The Widgets

| Widget | What it does | Key settings |
|---|---|---|
| *Build ImageNet* | Designs the network | The **Add Layers** buttons (*Conv2D*, *MaxPooling2D*, *AveragePooling2D*, *Flatten*, *Dense*, ...); in each layer box, its numbers; ↑ ↓ ✕ |
| *Train ImageNet* | Trains it | **Epochs**, **Validation split (%)** |
| *Classify Images* | Reports the test accuracy | |
| *Inspect ImageNet* | Shows the filters and feature maps of any convolution layer | **Layer**, **Row** |

New layers are inserted just before the last *Dense* layer, and the weight count under **Input Images** updates after every change. Use it to check your hand count before you train.

---

## Lab Exercise 4A: How Deep?

**Goal:** Add convolution layers one at a time and measure the effect.

*Time: about 10 seconds of training per network.*

**Before you run it:** SimpleCNN scored about 93%. Predict the accuracy with two convolution layers, and with three.

**Steps:**

1. Start from the workflow of Lab 3A (`workflows/mnist-cnn.ows`).
2. In *Build ImageNet* choose **SimpleCNN**. This is the *1 conv* network.
3. For *TwoConv*, press **Clear** and then press, in this order, **Conv2D** (32 filters), **MaxPooling2D**, **Conv2D** (set **filters** to **64**), **MaxPooling2D**, **Flatten**. Check **Weights: 34,826**.
4. For *ThreeConv*, press **Clear** and build **Conv2D** (32), **MaxPooling2D**, **Conv2D** (64), **MaxPooling2D**, **Conv2D** (**128**), **Flatten**, with no pooling after the last convolution. Check **Weights: 104,202**.
5. Train each for **5 epochs** and read the test accuracy from *Classify Images*. Repeat each twice.
6. Connect the trained *2 conv* network to *Inspect ImageNet* (and the training images to its **Images** input). Compare the feature maps of the first and of the second *Conv2D* layer for the same digit.

**Expected results** (two runs each; your numbers will differ by a point or two):

| Network | Weights | Test accuracy (two runs) |
|---|---|---|
| 1 conv (SimpleCNN) | 54,410 | 92.7%, 93.7% |
| 2 conv | 34,826 | 96.5%, 97.0% |
| 3 conv | 104,202 | 96.3%, 97.0% |

**Questions:**

1. How much did the second convolution layer add? About how many of the remaining errors did it remove? *(From 7% wrong to 3.5% wrong is half.)*
2. The third layer tripled the weights and added nothing measurable. Why might MNIST not need a third?
3. 2 conv is both smaller and better than 1 conv. What does that tell you about "bigger network, better results"?
4. In *Inspect ImageNet*, how do the feature maps of the second *Conv2D* layer differ from those of the first? Are they smaller, coarser, harder to read? Relate it to the receptive field of section 4.2.

---

## Lab Exercise 4B: LeNet Against SimpleCNN

**Goal:** Compare a classic design with the one you have been using.

*Time: 10 to 20 seconds per run.*

**Steps:**

1. In *Build ImageNet* choose **LeNet**. Check **Weights: 44,426**.
2. Train for **5** epochs and classify the test images. Then train for **10** epochs and classify again.

**Expected results:**

| Network | Weights | 5 epochs | 10 epochs |
|---|---|---|---|
| SimpleCNN | 54,410 | 93% | 94% to 95% |
| LeNet | 44,426 | 93.1% | 95.0% |

**Questions:**

1. LeNet has two convolutions and SimpleCNN has one, yet they score about the same after 5 epochs, while the 2 conv network of 4A scored 96% or more. What does that tell you about *which* details of the design matter?
2. LeNet's training accuracy after 5 epochs (95%) is lower than SimpleCNN's was (96%). Which of its differences could slow it down? *(Hint: look at the activation and the pooling.)*
3. Why would a design from 1998 still be a fair starting point in 2026?

---

## Lab Exercise 4C: Activation and Pooling

**Goal:** Change one detail at a time, and decide whether it matters.

*Time: 10 minutes.*

**Steps:** Starting from SimpleCNN, make each change, train for 5 epochs, and record the test accuracy.

| Change | How |
|---|---|
| relu to tanh | Set the *Conv2D* **activation** to `tanh` |
| relu to nothing | Set the *Conv2D* **activation** to `linear` |
| max to average pooling | Add *AveragePooling2D* (it lands below *Flatten*; press **↑** once to move it above), then delete *MaxPooling2D* |

**Expected results** (one run each):

| Variant | Test accuracy |
|---|---|
| relu, max pooling (SimpleCNN) | 92.7% |
| tanh | 92.0% |
| linear (no activation) | 91.9% |
| average pooling | 91.5% |

**Questions:**

1. Each variant is within about one point of the baseline, which is about the size of the run-to-run noise. Can you conclude from these numbers that relu is better than tanh?
2. With no activation, the network is still not purely linear. Which layer still makes it nonlinear? *(Hint: what does max pooling do to a number?)*
3. Here the choice barely matters, but in deeper networks relu is much preferred. Why might a squashing function like tanh cause trouble when there are many layers? *(Chapter 6 returns to this.)*

---

## Lab Exercise 4D: A Hidden Dense Layer (Optional)

**Goal:** Test whether adding weights, rather than structure, helps.

*Time: 5 minutes.*

**Steps:** Take *TwoConv* from 4A and add a *Dense* layer with **64** units and **relu** activation before the last *Dense* layer (the **Add Layers** *Dense* button does this). Check **Weights: 121,930**. Train for 5 epochs and record the test accuracy.

**Expected result:** about **96%** (this run: 96.1%), no better than the 34,826-weight TwoConv.

**Questions:**

1. The new layer adds 87,000 weights. Where do they come from? *(1,600 x 64 + 64, plus 64 x 10 + 10, less the 16,010 of the old output layer.)*
2. When would extra weights help, and when do they only add the risk of overfitting?

---

## Comprehension Questions

1. A *Conv2D* layer with 16 filters of size 5 x 5 receives an input of shape 12 x 12 x 6. Give its output shape (no padding) and its number of weights.
2. A 24 x 24 x 8 map goes through *MaxPooling2D* 2 and then *Flatten*. How many numbers come out?
3. What is a receptive field, and why does it grow with depth?
4. In Lab 4A the third convolution layer did not help. Name two possible reasons.
5. Why is it wrong to conclude that tanh is worse than relu from a single run that differs by 0.7 points?
6. Which choice mattered most in this chapter: activation, pooling type, or the number of convolutions?

---

## Key Terms

- **Receptive field**: the part of the input image that one number in a layer depends on.
- **Feature hierarchy**: edges in early layers, parts in middle layers, objects in late layers.
- **LeNet**: an early CNN for digits, with two convolutions, average pooling, and tanh.
- **Activation function**: the nonlinearity after a layer (relu, tanh, linear).
- **Average pooling**: shrinking by keeping the mean of each block.
- **Hidden layer**: any layer between the input and the output.

---

## Notes

- **Workflow for this chapter:** `workflows/mnist-cnn.ows`; each lab is an edit of the network in its *Build ImageNet*.
- **Setup:** MNIST subset (500 training images per class), Adam, batch size 32, 5 epochs, 20% stratified validation, test accuracy on the 1,000 test images.
- **Numbers quoted:**
  - Depth, two runs each: 1 conv 92.7%, 93.7%; 2 conv 96.5%, 97.0%; 3 conv 96.3%, 97.0%.
  - LeNet: 93.1% at 5 epochs and 95.0% at 10.
  - Activation and pooling (one run): relu 92.7%, tanh 92.0%, linear 91.9%, average pooling 91.5%.
  - 2 conv plus *Dense* 64: 96.1%.
- **Variance:** a point or two between runs, so none of the differences in 4C is established.
