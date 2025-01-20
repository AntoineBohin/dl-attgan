#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script complet d'inférence AttGAN :
  - Charge l'encoder, le decoder depuis des .pth
  - Charge une image
  - Force l'attribut "Blond_Hair" à 1
  - Génére et enregistre la nouvelle image
Usage :
  python inference_attgan.py --input chemin_vers_image.jpg --output resultat.jpg
Ou :
  python inference_attgan.py chemin_vers_image.jpg resultat.jpg
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

##############################################################################
# Définition des modèles (mêmes dimensions que durant l'entraînement)
##############################################################################

class Encoder(nn.Module):
    def __init__(self, input_dim=3):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        skip1 = self.relu(self.conv1(x))    # 64 x 64 x 64
        skip2 = self.relu(self.conv2(skip1))  # 128 x 32 x 32
        skip3 = self.relu(self.conv3(skip2))  # 256 x 16 x 16
        skip4 = self.relu(self.conv4(skip3))  # 512 x 8 x 8
        latent = self.relu(self.conv5(skip4)) # 1024 x 4 x 4
        return latent, [skip1, skip2, skip3, skip4]


class Decoder(nn.Module):
    def __init__(self, output_dim=3, attr_dim=13):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(1024 + attr_dim, 512, 4, 2, 1)  
        self.deconv2 = nn.ConvTranspose2d(1024, 256, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(128, output_dim, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, latent, attrs, skips):
        """
        latent: (N, 1024, 4, 4)
        attrs : (N, attr_dim)
        skips : liste [skip1, skip2, skip3, skip4]
        """
        # Expand (N, attr_dim) -> (N, attr_dim, 4, 4)
        attrs_4d = attrs.view(attrs.size(0), -1, 1, 1).expand(-1, -1, 4, 4)
        x = torch.cat([latent, attrs_4d], dim=1)

        x = self.deconv1(x)
        x = torch.cat([x, skips[3]], dim=1)

        x = self.deconv2(x)
        x = torch.cat([x, skips[2]], dim=1)

        x = self.deconv3(x)
        x = torch.cat([x, skips[1]], dim=1)

        x = self.deconv4(x)
        x = torch.cat([x, skips[0]], dim=1)

        x = self.deconv5(x)
        return self.tanh(x)

##############################################################################
# Paramètres "fixes" correspondant à ceux utilisés à l'entraînement
##############################################################################

# On suppose que la liste d'attributs est la même que pendant l'entraînement.
chosen_attrs = [
    "5_o_Clock_Shadow",  # idx 0
    "Bald",              # idx 1
    "Bangs",             # idx 2
    "Black_Hair",        # idx 3
    "Blond_Hair",        # idx 4  <-- On va forcer celui-ci à 1
    "Brown_Hair",        # idx 5
    "Male",              # idx 6
    "Mouth_Slightly_Open", # idx 7
    "No_Beard",          # idx 8
    "Pale_Skin",         # idx 9
    "Smiling",           # idx 10
    "Wearing_Earrings",  # idx 11
    "Young",             # idx 12
]
attr_dim = len(chosen_attrs)

# Index où se trouve "Blond_Hair"
IDX_BLOND = 4

# Dimensions
IMAGE_SIZE = 128

# Dossier où se trouvent les poids
CHECKPOINT_DIR = "./saved_models_smoothing"  # À adapter si besoin

##############################################################################
# Fonction d'inférence
##############################################################################

def make_person_blonde(input_path, output_path, device="cpu"):
    """
    Charge l'image 'input_path', force l'attribut Blond_Hair = 1,
    encode + decode, puis sauvegarde l'image générée dans 'output_path'.
    """
    # 1) Charger l'encoder/decoder
    encoder = Encoder(input_dim=3).to(device)
    decoder = Decoder(output_dim=3, attr_dim=attr_dim).to(device)

    encoder_ckpt = os.path.join(CHECKPOINT_DIR, "encoder.pth")
    decoder_ckpt = os.path.join(CHECKPOINT_DIR, "decoder.pth")

    assert os.path.exists(encoder_ckpt), f"Fichier introuvable : {encoder_ckpt}"
    assert os.path.exists(decoder_ckpt), f"Fichier introuvable : {decoder_ckpt}"

    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))

    # Mode évaluation
    encoder.eval()
    decoder.eval()

    # 2) Préparer les transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # 3) Charger l'image
    image = Image.open(input_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # shape (1,3,128,128)

    # 4) À l'inférence, on n'a pas forcément les attributs réels,
    #    on peut partir d'un vecteur "réels" si vous les connaissez,
    #    ou tout simplement un vecteur initial 0.
    #    Pour la démo, on part d'un vecteur d'attributs supposé 0:
    attrs = torch.zeros((1, attr_dim), device=device)
    # Si vous avez les attributs réels, vous pourriez les charger ici
    # Ex:
    # real_attrs = [...]
    # attrs[0] = torch.tensor(real_attrs)

    # 5) Forcer Blond_Hair à 1
    attrs[0, IDX_BLOND] = 1.0

    # 6) Encode -> Decode
    with torch.no_grad():
        latent, skips = encoder(image_tensor)        # (1,1024,4,4), skip connections
        output = decoder(latent, attrs, skips)       # (1,3,128,128)

    # 7) Dénormaliser [-1,1] -> [0,1]
    output = (output * 0.5) + 0.5

    # 8) Sauvegarder
    vutils.save_image(output, output_path)
    print(f"[OK] Image générée sauvegardée dans {output_path}")

##############################################################################
# Script principal
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="?", default=None,
                        help="Chemin vers l'image d'entrée")
    parser.add_argument("output", type=str, nargs="?", default=None,
                        help="Chemin du fichier de sortie (image générée)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu ou cuda (ex. cuda:0)")
    return parser.parse_args()

def main():
    sys.argv = [
        "mon_script.py",                # "faux" nom du script
        "image.png",         # input
        "zzzz.png",         # output
        "--device", "cuda:0"            # device
    ]
    args = parse_arguments()

    # Appel de la fonction d'inférence
    make_person_blonde(args.input, args.output, device=args.device)

if __name__ == "__main__":
    main()
