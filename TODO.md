# TODO

## Add-on widgets needed (orange3-imagenets)

- [ ] Dataset loaders (or a documented recipe) for MNIST, Fashion-MNIST, EMNIST, CIFAR-10 as Orange image tables
- [ ] Build ImageNet: configurable input shape, grayscale (1 channel) support, Conv2DTranspose/UpSampling layers (needed for GAN generator and diffusion denoiser)
- [ ] Train GAN (generator + discriminator, live sample grid per epoch)
- [ ] Train Diffusion (noise schedule, noise-prediction net, optional class conditioning)
- [ ] Generate Images (GAN: latent vectors; diffusion: denoising steps, show intermediate steps)
- [ ] Check Train ImageNet handles >10 classes (EMNIST) and a validation split

## Book

- [ ] Write chapters 1-10 (stubs exist)
- [ ] Title image
- [ ] Generate workflows with utils/make_ows.py and run every lab
- [ ] Record real numbers (accuracy, epochs, timings) for each lab
