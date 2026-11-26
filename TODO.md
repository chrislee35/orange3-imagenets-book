# TODO

## Done

- Add-on widgets: Build / Train / Classify ImageNet, Preprocess, Augment, Train GAN, Train Diffusion, Generate Images, Save / Load ImageNet, Pretrained ImageNet, Inspect ImageNet, Misclassified Images
- Chapters 1-10 drafted; every number comes from a script run or a widget run (see each chapter's Notes)
- Workflows: mnist-explore, datasets-tour, mnist-cnn, cifar-transfer, mnist-gan, mnist-diffusion

## Before a release

- [ ] Title image (`imgs/Title Image.png`; README links to it)
- [ ] Click through every lab in the Orange canvas (most were verified by script or by running the widgets headless)
- [x] orange3-imagenets 0.0.6 is on PyPI with all the widgets; all six workflows load against the released add-on (checked with `scheme_load`)
- [ ] Open each workflow in a clean Orange install (a fresh environment, not the development one) and click through the labs
- [ ] Real icons for the add-on widgets (GAN, diffusion, generate, inspect, misclassified, pretrained are placeholder copies)
- [ ] Re-check run-to-run numbers on another machine; check licensing text for MNIST, EMNIST, CIFAR-10
- [ ] Optional: a cat / dog / duck image set with a clear license (not used by the chapters now)
- [ ] Optional: Classify Images drops unreadable images, which would misalign the Prediction column
