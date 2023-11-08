# Wasserstein GAN with Gradient Penalty (WGAN-GP)

This repository contains the implementation of a Wasserstein GAN with Gradient Penalty (WGAN-GP). It was trained on the Handwritten Letters Dataset (https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/data).

This repository contains a Jupyter notebook which implements the entire pipeline for training and evaluating a Wasserstein GAN with Gradient Penalty (WGAN-GP). The notebook includes data preprocessing, model definition, training loop, and visualization of the generated images.

## Features
- Implementation of WGAN-GP for improved training stability.
- Use of Wasserstein distance for loss calculations, which provides meaningful training curves.
- Gradient penalty term to enforce the Lipschitz constraint, leading to better convergence properties.

## Model Architecture
The model consists of two main components: the Generator (G) and the Critic (D), which are trained concurrently through adversarial training where the generator attempts to produce data that are indistinguishable from real data, while the critic aims to distinguish between real and generated data.

### Generator (G)
- Input: Random noise vector `z`.
- Architecture: 
- **Label Embedding Layer**: Maps class labels to a dense vector.
- **Transposed Convolutional Blocks**: Upsample the noise vector to generate images.
  - **LeakyReLU Activation**: Applied in all but the final block.
  - **Batch Normalization**: Stabilizes training by normalizing the output of each convolutional layer.
  - **Tanh Activation**: Ensures the final image pixel values are in the range `[-1, 1]`.
    
- Output: Synthetic data that mimics the distribution of the real data.

### Critic (D)
- Input: Real data or generated data.
- Architecture: 
- **Label Embedding Layer**: Converts class labels into a flat tensor that can be concatenated with the image tensor.
- **Convolutional Layers**: Extract features from the combined image and label tensor.
  - **LeakyReLU Activation**: Ensures efficient backpropagation of gradients with a negative slope of 0.2.
  - **Batch Normalization**: Normalizes the output of each layer to improve training dynamics.
- **Dropout**: Mitigates overfitting by randomly zeroing out a portion of the features.
- **Linear Layer**: Produces the final score indicating the authenticity of the input image.

- Output: A score representing the critic's assessment of the data's authenticity.

The key innovation in WGAN-GP is the gradient penalty, which stabilizes training by enforcing a soft version of the Lipschitz constraint on the critic. Instead of clipping the weights of the critic, we penalize the norm of the gradient of the critic with respect to its input.

## References
- Original WGAN paper: "Wasserstein GAN" by Martin Arjovsky, Soumith Chintala, and Léon Bottou.
- WGAN-GP paper: "Improved Training of Wasserstein GANs" by Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville.
