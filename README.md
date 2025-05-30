# Final Project Deep Learning Course (ENS MVA / CentraleSupélec) - AttGAN: Facial Attribute Editing by Only Changing What You Want
## Authors: Antoine Bohin and Logan Renaud

This project by Antoine Bohin and Logan Renaud is an implementation of [AttGAN: Facial Attribute Editing by Only Changing What You Want](https://arxiv.org/abs/1711.10678), a deep learning model for facial attribute editing. 

## Introduction
Facial attribute editing aims to edit attributes of interest (e.g. hair color, accessories, mouth) on a face image while preserving the identity of the person. It has applications in various industries, like entertainment and media generation (e.g. deepfakes, face manipulation in films or photo shooting), but also plastic surgery or security. 
With the lack of suitable labeled datasets, current approaches rely on generative models like Generative Adversarial Networks (GANs) and encoder-decoder architectures aiming at decoding latent representation of a given face conditioned on the desired attributes. Some of them attempt to build attribute-independent latent representations. However, these techniques often suffer from loss of facial details, unintended attribute changes due to correlations in the dataset, and the need for multiple models to handle different attribute edits.

[AttGAN](https://arxiv.org/abs/1711.10678) (Attribute Generative Adversarial Network) introduces an improved framework, leveraging an encoder-decoder structure combined with attribute classification constraints, reconstruction learning and adversarial learning at training. It results in more realistic facial transformations, better retention of details and a more flexible model that can handle multiple attributes simultaneously with a single implementation.

Our custom implementation gave the following results on some test images:

![](images/results_single_/single_3.jpg)
From left to right: Input, Reconstruction, Bald, Black Hair, Blond Hair, Brown Hair, Bushy Eyebrows, Eyeglasses, Male, Mouth Slightly Open, Mustache, Beard, Pale Skin, Smiling, Young


## Model Architecture

### Principles

AttGAN is built upon an encoder-decoder architecture: 
![](images/model_overview.png)

The model is combined with additional components used during training to learn how to generate realistic and precise edits: 
1) Attribute Classification Constraint: A pre-trained Attribute Classifier is used to check whether the attributes in the output image match the desired ones.
2) Reconstruction Learning: Aims at preserving the attribute-excluding details. If an image is passed without modification, it should be perfectly reconstructed.
3) Adversarial Learning: Employed for visually realistic generation. A Wasserstein GAN with Gradient Penalty (WGAN-GP) ensures that the generated images look realistic and indistinguishable from real pictures.

### Implementation

![](images/architecture.png)

Our AttGAN implementation presents the following default architecture, consisting of :
- 5 convolutional layers for the encoder.
- 5 transposed convolutional layers	for the decoder to reconstruct the modified image.
- A discriminator based on a CNN with a dual output: one branch for real/fake classification, another branch for multi-label attribute prediction.

The model was only trained on a subset of 13 attributes of interest (out of 27 for the whole CelebA dataset). Attribute injection is performed by concatenating the attribute vector with the encoded feature map before decoding. The attributes, represented as a binary vector (e.g., “smiling” = 1, “beard” = 0), are first reshaped and broadcasted to match the spatial dimensions of the feature map. This injection can occur once at the bottleneck or at multiple decoder layers, by playing on the "inject_layers" parameter.

The model also presents shortcut connections, inspired by U-Net to help preserve fine details during image reconstruction. They link encoder and decoder layers, allowing high-level features from the input image to be skipped over the latent space and reused in the decoder. The number of skipping connections can be modified with the "shortcut_layers" parameters.

### Intensity Control
We explored the model’s capability to generalize beyond binary attribute values and use continuous intensity levels for attribute editing. While AttGAN is originally trained with discrete 0/1 labels, we observed that it can naturally handle gradual attribute modifications during testing, without requiring any architectural changes.
Some results for the attribute "Smiling":

![](images/results_intensity_/intensity_smiling_2.jpg)

## Requirements

Type the following command to install the required modules:
```bash
pip install -r requirements.txt
```

If you want to retrain the model, you'll also need to download the CelebA Dataset. You can modify the path for the dataset in the `download_data.py` file, and then execute the command:
```bash
python3 download_data.py
```
Images will be placed in `./data/img_align_celeba/*.jpg`, and attribute labels in `./data/list_attr_celeba.csv`

## Model Training

A model can be retrained from scratch on the CelebA Dataset using the following command:
```bash
python3 train.py --experiment_name your_training --gpu
```
Although we decided to reuse the same default hyperparameters as the paper, they can be changed by specifying them in the previous command (see the python file for the exact syntax).

Note: A good GPU is recommended. We trained the model using CentraleSupélec Metz DCE, and training time varied from ∼15 min/epoch for a 24 GB RAM GPU to ∼25 min/epoch for a 11 GB RAM GPU. As the usage of CentraleSupélec's GPUs is constrained, we only trained the model for ∼50 epochs but we recommend going further in training for a better reconstruction of the original images. If you want to submit a long training on the DCE you can use `sbatch train_job.sh`

## Model Testing

### Pre-trained model loading
We pre-trained a model during a training of 45 epochs (c.20h). The weights of the model can be downloaded using this [link to the pre-trained model](https://drive.google.com/drive/folders/1rQsXRjN6ZPMtKgkicsPUNXE_8ZtO0UyM?usp=share_link)
Download the folder "full_training" and put it in the `output` folder. The weights of the model will then be in the file `output/full_training/checkpoint/weights.44.pth` and the settings of the training in `output/full_training/setting.txt`.


### Single attribute editing

The following commands will test the single attribute editing for every attribute the model was trained on. If you are using our model/default parameters, the attributes are (left to right) : 
'Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Smiling', 'Young'.

If you want to use test images from CelebA Dataset, you can type :
 ```bash
python3 test.py --experiment_name full_training --gpu
```

If you want to use custom images (need to be cropped as square), add it in the `./data/custom` folder with a file `list_attr_custom.txt` describing the original attributes of the image (see examples):
 ```bash
python3 test.py --custom_img --experiment_name full_training --gpu
```

Here are some results we obtained :

![](images/results_single_/single_1.jpg)
![](images/results_single_/single_4.jpg)
From left to right: Input, Reconstruction, Bald, Black Hair, Blond Hair, Brown Hair, Bushy Eyebrows, Eyeglasses, Male, Mouth Slightly Open, Mustache, Beard, Pale Skin, Smiling, Young

### Attribute intensity editing

We have also implemented intensity control for attribute editing, allowing for smooth transitions between different attribute intensities. This feature enables fine-grained control over attributes such as smiling intensity, hair color strength, or eyeglasses visibility, rather than applying a binary transformation.

You can test this functionality using the provided script. It generates a series of images where the selected attribute is gradually adjusted from a minimum intensity (--test_int_min) to a maximum intensity (--test_int_max) across multiple steps (--n_slide):
 ```bash
python3 test_intensity.py --experiment_name test_full_training --test_att Male --gpu
```

Some results for the attribute "No Beard":

![](images/results_intensity_/intensity_beard_1.jpg)

Some results for the attribute "Young":

![](images/results_intensity_/intensity_young_1.jpg)

