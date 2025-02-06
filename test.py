 #!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

##############################################################################
# Définition des modules (Encoder, Decoder, Discriminateur/Classifier)
##############################################################################


class Encoder(nn.Module):
    def __init__(self, input_dim=3):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, 4, 2, 1)    # 128 -> 64
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)           # 64 -> 32
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)          # 32 -> 16
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)          # 16 -> 8
        self.bn4   = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)         # 8 -> 4
        self.bn5   = nn.BatchNorm2d(1024)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        out1 = self.lrelu(self.bn1(self.conv1(x)))      # (N, 64, 64, 64)
        out2 = self.lrelu(self.bn2(self.conv2(out1)))     # (N, 128, 32, 32)
        out3 = self.lrelu(self.bn3(self.conv3(out2)))     # (N, 256, 16, 16)
        out4 = self.lrelu(self.bn4(self.conv4(out3)))     # (N, 512, 8, 8)
        latent = self.lrelu(self.bn5(self.conv5(out4)))   # (N, 1024, 4, 4)
        return latent, [out1, out2, out3, out4]

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, output_dim=3, attr_dim=5):
        super(Decoder, self).__init__()
        # La couche 1 reçoit le tenseur latent concaténé avec les attributs tilés à 4x4
        self.deconv1 = nn.ConvTranspose2d(1024 + attr_dim, 512, kernel_size=4, stride=2, padding=1)   # sort : (N, 512, 8, 8)
        
        # Pour la couche 2, on injecte :
        # - la sortie de la couche précédente (512 canaux)
        # - un tile des attributs à la résolution 8x8 (attr_dim canaux)
        # - le skip correspondant (skips[3] de 512 canaux)
        # Soit en tout : 512 + attr_dim + 512 = 1024 + attr_dim canaux
        self.deconv2 = nn.ConvTranspose2d(1024 + attr_dim, 256, kernel_size=4, stride=2, padding=1)   # sort : (N, 256, 16, 16)
        
        # Pour la couche 3, entrée = 256 (deconv2) + attr_dim (à 16x16) + 256 (skips[2]) = 512 + attr_dim
        self.deconv3 = nn.ConvTranspose2d(512 + attr_dim, 128, kernel_size=4, stride=2, padding=1)    # sort : (N, 128, 32, 32)
        
        # Pour la couche 4, entrée = 128 (deconv3) + attr_dim (à 32x32) + 128 (skips[1]) = 256 + attr_dim
        self.deconv4 = nn.ConvTranspose2d(256 + attr_dim, 64, kernel_size=4, stride=2, padding=1)     # sort : (N, 64, 64, 64)
        
        # Pour la couche 5, entrée = 64 (deconv4) + attr_dim (à 64x64) + 64 (skips[0]) = 128 + attr_dim
        self.deconv5 = nn.ConvTranspose2d(128 + attr_dim, output_dim, kernel_size=4, stride=2, padding=1)  # sort : (N, output_dim, 128, 128)
        
        # BatchNorm sur les sorties intermédiaires
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, latent, attrs, skips):
        """
        latent : (N, 1024, 4, 4)
        attrs  : (N, attr_dim)
        skips  : liste de tenseurs provenant de l’encodeur, avec :
                 skips[0] : (N, 64, 64, 64)
                 skips[1] : (N, 128, 32, 32)
                 skips[2] : (N, 256, 16, 16)
                 skips[3] : (N, 512, 8, 8)
        """
        # Étape 1 : injection initiale
        # On étend le vecteur d'attributs à la taille spatiale 4x4.
        a_tile = attrs.view(attrs.size(0), -1, 1, 1).expand(-1, -1, 4, 4)
        x = torch.cat([latent, a_tile], dim=1)  # (N, 1024 + attr_dim, 4, 4)
        x = self.relu(self.bn1(self.deconv1(x)))  # (N, 512, 8, 8)
        
        # Étape 2 : avant de passer à deconv2, on concatène :
        # - la sortie précédente x (512 canaux)
        # - les attributs tilés à 8x8 (attr_dim canaux)
        # - le skip correspondant (skips[3], 512 canaux)
        a_tile = attrs.view(attrs.size(0), -1, 1, 1).expand(-1, -1, 8, 8)
        x = torch.cat([x, a_tile, skips[3]], dim=1)  # (N, 512 + attr_dim + 512 = 1024 + attr_dim, 8, 8)
        x = self.relu(self.bn2(self.deconv2(x)))       # (N, 256, 16, 16)
        
        # Étape 3 : injection à la résolution 16x16
        a_tile = attrs.view(attrs.size(0), -1, 1, 1).expand(-1, -1, 16, 16)
        x = torch.cat([x, a_tile, skips[2]], dim=1)      # (N, 256 + attr_dim + 256 = 512 + attr_dim, 16, 16)
        x = self.relu(self.bn3(self.deconv3(x)))         # (N, 128, 32, 32)
        
        # Étape 4 : injection à la résolution 32x32
        a_tile = attrs.view(attrs.size(0), -1, 1, 1).expand(-1, -1, 32, 32)
        x = torch.cat([x, a_tile, skips[1]], dim=1)      # (N, 128 + attr_dim + 128 = 256 + attr_dim, 32, 32)
        x = self.relu(self.bn4(self.deconv4(x)))         # (N, 64, 64, 64)
        
        # Étape 5 : injection à la résolution 64x64
        a_tile = attrs.view(attrs.size(0), -1, 1, 1).expand(-1, -1, 64, 64)
        x = torch.cat([x, a_tile, skips[0]], dim=1)      # (N, 64 + attr_dim + 64 = 128 + attr_dim, 64, 64)
        x = self.deconv5(x)                              # (N, output_dim, 128, 128)
        return self.tanh(x)
    
# --- Discriminateur / Classifieur ---
class ClassifierDiscriminator(nn.Module):
    def __init__(self, input_dim=3, attr_dim=5):
        super(ClassifierDiscriminator, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(input_dim, 64, 4, 2, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(512, 1024, 4, 2, 1))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        # Après convolutions, la taille est (1024 x 4 x 4)
        self.classifier = nn.Linear(1024 * 4 * 4, attr_dim)
        self.discriminator = nn.Linear(1024 * 4 * 4, 1)
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.flatten(x)
        attr_preds = self.classifier(x)
        real_fake_score = self.discriminator(x)
        return attr_preds, real_fake_score

##############################################################################
# Script principal
##############################################################################
def main():
    input = "rousse.png"
    output = "./rousse16.png"
    modify = "Blond_Hair=1"

    # Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Liste des attributs tels qu'utilisés à l'entraînement (l'ordre doit être respecté)
    chosen_attrs = [
    "Blond_Hair",
    "Bald",
    "Male",
    "No_Beard",
    "Black_Hair",
]
    attr_dim = len(chosen_attrs)

    # Chargement des modèles pré-entraînés
    encoder = Encoder(input_dim=3).to(device)
    decoder = Decoder(output_dim=3, attr_dim=attr_dim).to(device)
    encoder_path = os.path.join("saved_model_4O_recommendation_13", "encoder.pth")
    decoder_path = os.path.join("saved_model_4O_recommendation_13", "decoder.pth")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()

    # Optionnel : charger le modèle de classification pour récupérer les attributs originaux
    use_classifier = True
    try:
        classifier = ClassifierDiscriminator(input_dim=3, attr_dim=attr_dim).to(device)
        classifier_path = os.path.join("saved_model_4O_recommendation_13", "classifier_discriminator.pth")
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
    except Exception as e:
        print("Erreur lors du chargement du classifier :", e)
        print("Utilisation d'un vecteur d'attributs par défaut.")
        use_classifier = False

    # Transformation de l'image d'entrée
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Charger l'image d'entrée
    image = Image.open(input).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Récupération (ou définition) du vecteur d'attributs
    if use_classifier:
        with torch.no_grad():
            preds, _ = classifier(input_tensor)
            print(preds)
            preds = torch.sigmoid(preds)
            # Binarisation avec seuil 0.5
            attr_vector = (preds > 0.5).float()
    else:
        # Vecteur par défaut (tous à 0)
        attr_vector = torch.zeros((1, attr_dim), device=device)

    # Analyse de la chaîne de modifications (ex: "Bald=1,Smiling=0")
    modifications = {}
    if modify:
        for pair in modify.split(","):
            if "=" in pair:
                attr, val = pair.split("=")
                attr = attr.strip()
                try:
                    val = float(val.strip())
                except ValueError:
                    print(f"Valeur non valide pour {attr}, utilisation de 0")
                    val = 0.0
                modifications[attr] = val
    print(modifications)

    # Appliquer les modifications au vecteur d'attributs
    for attr, val in modifications.items():
        if attr in chosen_attrs:
            idx = chosen_attrs.index(attr)
            attr_vector[0, idx] = val
        else:
            print(f"Attribut '{attr}' inconnu. Doit être dans : {chosen_attrs}")
    print(attr_vector)

    # Génération de l'image modifiée
    with torch.no_grad():
        latent, skips = encoder(input_tensor)
        output_tensor = decoder(latent, attr_vector, skips)

    # Denormalisation (les sorties sont dans [-1, 1])
    output_tensor = (output_tensor.clamp(-1, 1) + 1) / 2.0
    logits, _ = classifier(output_tensor)
    probs = torch.sigmoid(logits)
    attr_vector_modified = (probs > 0.5).float()
    print(f"attrs_vector modifier{attr_vector_modified}")
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
    output_image.save(output)
    
    # Transformation de l'image d'entrée
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Charger l'image d'entrée
    image = Image.open(input).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Récupérer le vecteur d'attribut original en utilisant le classifieur (si disponible)
    if use_classifier:
        with torch.no_grad():
            logits, _ = classifier(input_tensor)
            probs = torch.sigmoid(logits)
            original_attr_vector = (probs > 0.5).float()
        print("Vecteur d'attribut original :", original_attr_vector)
    else:
        original_attr_vector = torch.zeros((1, attr_dim), device=device)



if __name__ == "__main__":
    main()
