# AttGAN: Facial Attribute Editing by Only Changing What You Want
This project is an implementation of [AttGAN: Facial Attribute Editing by Only Changing What You Want](https://arxiv.org/abs/1711.10678), a deep learning model for facial attribute editing. 


## Introduction
Facial attribute editing aims to edit attributes of interest (e.g. hair color, accessories, mouth) on a face image while preserving the identity of the person. It has applications in various industries, like entertainment and media generation (e.g. deepfakes, face manipulation in films or photo shooting), but also plastic surgery or security. 
With the lack of suitable labeled datasets, current approaches rely on generative models like Generative Adversarial Networks (GANs) and encoder-decoder architectures aiming at decoding latent representation of a given face conditioned on the desired attributes. Some of them attempt to build attribute-independent latent representations. However, these techniques often suffer from loss of facial details, unintended attribute changes due to correlations in the dataset, and the need for multiple models to handle different attribute edits.

[AttGAN](https://arxiv.org/abs/1711.10678) (Attribute Generative Adversarial Network) introduces an improved framework, leveraging an encoder-decoder structure combined with attribute classification constraints, reconstruction learning and adversarial learning at training. It results in more realistic facial transformations, better retention of details and a more flexible model that can handle multiple attributes simultaneously with a single implementation.

## Model Architecture
AttGAN is built upon an encoder-decoder architecture and integrates three key learning objectives:
	1.	Attribute Classification Constraint: Ensures the generated images have the desired attributes by training a classifier to verify attribute modifications.
	2.	Reconstruction Learning: Preserves all attribute-excluding details, ensuring that only the specified attributes change.
	3.	Adversarial Learning: Uses a Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate visually realistic face images.



 Unlike previous approaches that impose constraints on the latent space, AttGAN applies an attribute classification constraint on the generated image, ensuring that only the desired attributes are modified while keeping the identity and other characteristics intact.

Our implementation reproduces the AttGAN model as described in the original paper, enabling the generation of realistic and high-quality face modifications. We trained and tested the model on facial images to evaluate its effectiveness.


1) telecharge le dataset avec download_dataset
2) entraine le modèle avec train.py, ca sauvegardera le modèle quelque part. C'est la qu'il faut changer les hyperparamètres
3) Le tester en inférence avec test.py
