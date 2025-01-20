import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

##############################################################################
# 1) Dataset personnalisé (lecture CSV + transformation)
##############################################################################

class CelebADataset(Dataset):
    """
    Dataset qui lit les images et leurs attributs (CSV).
    On suppose :
      - img_dir contient toutes les images de CelebA (ex. "img_align_celeba/")
      - attr_path est le chemin vers "list_attr_celeba.csv"
      - chosen_attrs est la liste ordonnée des attributs qu'on veut extraire
    """
    def __init__(self, img_dir, attr_path, chosen_attrs, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.chosen_attrs = chosen_attrs

        # Lecture du CSV
        df = pd.read_csv(attr_path)

        # Conversion -1 -> 0 pour les attributs
        for c in df.columns[1:]:
            df[c] = (df[c] == 1).astype(int)

        # Filtrage sur les colonnes utiles
        self.df_all = df[["image_id"] + chosen_attrs]

    def __len__(self):
        return len(self.df_all)

    def __getitem__(self, idx):
        row = self.df_all.iloc[idx]
        img_name = row["image_id"]
        img_path = os.path.join(self.img_dir, img_name)

        # Charger l'image
        image = Image.open(img_path).convert("RGB")

        # Transformations (resize, normalisation, etc.)
        if self.transform:
            image = self.transform(image)

        # Attributs shape (len(chosen_attrs),)
        attrs = row[self.chosen_attrs].values.astype(np.float32)
        attrs = torch.from_numpy(attrs)

        return image, attrs


##############################################################################
# 2) Paramètres
##############################################################################

root_dir = "/usr/users/siapartnerscomsportif/renaud_log/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2" # A adapter
img_dir = os.path.join(root_dir, "img_align_celeba/img_align_celeba")
attr_path = os.path.join(root_dir, "list_attr_celeba.csv")

chosen_attrs = [
    "5_o_Clock_Shadow",
    "Bald",
    "Bangs",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Male",
    "Mouth_Slightly_Open",
    "No_Beard",
    "Pale_Skin",
    "Smiling",
    "Wearing_Earrings",
    "Young",
]
attr_dim = len(chosen_attrs)

image_size = 128
batch_size = 64
epochs = 10

# Changements de coefficients
lambda_rec = 5   # Au lieu de 10, pour laisser plus de place à la partie adversaire
lambda_cls = 1

# Appliquer un LR plus bas pour le disc, plus haut pour le gen
lr_disc = 1e-4
lr_gen = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# 3) Transform & DataLoader
##############################################################################
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

dataset = CelebADataset(img_dir, attr_path, chosen_attrs, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


##############################################################################
# 4) Définition des modules (Encoder, Decoder, Disc/Classifier)
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
        skip1 = self.relu(self.conv1(x))   # 64 x 64 x 64
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
        # Expand attrs (N, attr_dim) -> (N, attr_dim, 4, 4)
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


class ClassifierDiscriminator(nn.Module):
    def __init__(self, input_dim=3, attr_dim=13):
        super(ClassifierDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(1024 * 4 * 4, attr_dim)
        self.discriminator = nn.Linear(1024 * 4 * 4, 1)

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.conv1(x))
        x = nn.LeakyReLU(0.2)(self.conv2(x))
        x = nn.LeakyReLU(0.2)(self.conv3(x))
        x = nn.LeakyReLU(0.2)(self.conv4(x))
        x = nn.LeakyReLU(0.2)(self.conv5(x))
        x = self.flatten(x)

        attr_preds = self.classifier(x)
        real_fake_score = self.discriminator(x)
        return attr_preds, real_fake_score


##############################################################################
# 5) Initialisation modèles & Optimiseurs
##############################################################################

encoder = Encoder(input_dim=3).to(device)
decoder = Decoder(output_dim=3, attr_dim=attr_dim).to(device)
classifier_discriminator = ClassifierDiscriminator(input_dim=3, attr_dim=attr_dim).to(device)

# Notez qu'on utilise un LR différent pour le disc et le gen
opt_enc = optim.Adam(encoder.parameters(), lr=lr_gen, betas=(0.5, 0.999))
opt_dec = optim.Adam(decoder.parameters(), lr=lr_gen, betas=(0.5, 0.999))
opt_cls_disc = optim.Adam(classifier_discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

criterion_adv = nn.BCEWithLogitsLoss()
criterion_cls = nn.BCEWithLogitsLoss()
criterion_rec = nn.L1Loss()

##############################################################################
# 6) Boucle d'entraînement avec label smoothing
##############################################################################

for epoch in range(epochs):
    for i, (images, real_attrs) in enumerate(loader):
        if i>1000:
            break
        images = images.to(device)         # (N, 3, 128, 128)
        real_attrs = real_attrs.to(device) # (N, attr_dim)
        batch_size = images.size(0)

        # -------------------------------------------------
        # (A1) Entraînement Discriminateur + Classifieur
        # -------------------------------------------------
        with torch.no_grad():
            latent, skips = encoder(images)
            fake_images = decoder(latent, real_attrs, skips)

        # Disc/Cls sur vrai
        real_attr_preds_d, real_score_d = classifier_discriminator(images)
        # Disc/Cls sur faux
        fake_attr_preds_d, fake_score_d = classifier_discriminator(fake_images.detach())

        # Label smoothing => ex. "vrai" = 0.9, "faux" = 0.1
        real_labels = 0.9 * torch.ones(batch_size, 1, device=device)
        fake_labels = 0.1 * torch.ones(batch_size, 1, device=device)

        # Pertes adversaires
        loss_real = criterion_adv(real_score_d, real_labels)
        loss_fake = criterion_adv(fake_score_d, fake_labels)
        loss_disc = loss_real + loss_fake

        opt_cls_disc.zero_grad()
        loss_disc.backward()
        opt_cls_disc.step()

        # -------------------------------------------------
        # (A2) Classifieur sur images réelles
        # -------------------------------------------------
        real_attr_preds_cls, _ = classifier_discriminator(images)
        loss_cls_real = criterion_cls(real_attr_preds_cls, real_attrs)

        opt_cls_disc.zero_grad()
        loss_cls_real.backward()
        opt_cls_disc.step()

        # -------------------------------------------------
        # (A3) Générateur (Encoder + Decoder)
        # -------------------------------------------------
        latent, skips = encoder(images)
        fake_images = decoder(latent, real_attrs, skips)

        fake_attr_preds_gen, fake_score_gen = classifier_discriminator(fake_images)

        # On veut que le disc sorte "vrai" => 0.9
        loss_adv_g = criterion_adv(fake_score_gen, real_labels)

        # Reconstruction
        loss_rec = criterion_rec(images, fake_images)

        # Classification
        loss_cls_g = criterion_cls(fake_attr_preds_gen, real_attrs)

        loss_gen = loss_adv_g + lambda_rec * loss_rec + lambda_cls * loss_cls_g

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        loss_gen.backward()
        opt_enc.step()
        opt_dec.step()

        # Logs
        if (i+1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Step {i+1}/{len(loader)} "
                  f"LossD: {loss_disc.item():.4f} | "
                  f"LossCls: {loss_cls_real.item():.4f} | "
                  f"LossG: {loss_gen.item():.4f}")

    print(f"End of epoch {epoch+1}/{epochs} - "
          f"LossD: {loss_disc.item():.4f} | "
          f"LossCls: {loss_cls_real.item():.4f} | "
          f"LossG: {loss_gen.item():.4f}")

##############################################################################
# 7) Sauvegarde
##############################################################################

save_dir = "./saved_models_smoothing"
os.makedirs(save_dir, exist_ok=True)

torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
torch.save(decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
torch.save(classifier_discriminator.state_dict(), os.path.join(save_dir, "classifier_discriminator.pth"))

torch.save(opt_enc.state_dict(), os.path.join(save_dir, "opt_enc.pth"))
torch.save(opt_dec.state_dict(), os.path.join(save_dir, "opt_dec.pth"))
torch.save(opt_cls_disc.state_dict(), os.path.join(save_dir, "opt_cls_disc.pth"))

print("Fin de l'entraînement, modèles sauvegardés avec succès !")
