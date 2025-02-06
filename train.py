import os
import torch
print(torch.__version__)
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
    Lit les images et leurs attributs depuis un CSV.
    On suppose :
      - img_dir contient les images (ex. "img_align_celeba/")
      - attr_path est le chemin vers "list_attr_celeba.csv"
      - chosen_attrs est la liste ordonnée des attributs souhaités.
    """
    def __init__(self, img_dir, attr_path, chosen_attrs, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.chosen_attrs = chosen_attrs

        # Lecture du CSV
        df = pd.read_csv(attr_path)
        # Conversion : -1 devient 0
        for c in df.columns[1:]:
            df[c] = (df[c] == 1).astype(int)
        # Garder uniquement les colonnes utiles
        self.df_all = df[["image_id"] + chosen_attrs]

    def __len__(self):
        return len(self.df_all)

    def __getitem__(self, idx):
        row = self.df_all.iloc[idx]
        img_name = row["image_id"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        attrs = row[self.chosen_attrs].values.astype(np.float32)
        attrs = torch.from_numpy(attrs)
        return image, attrs

##############################################################################
# 2) Paramètres
##############################################################################
# À adapter à votre environnement
root_dir = "/usr/users/siapartnerscomsportif/renaud_log/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2"  # A adapter à votre environnement
img_dir = os.path.join(root_dir, "img_align_celeba/img_align_celeba")
attr_path = os.path.join(root_dir, "list_attr_celeba.csv")

# On utilise 5 attributs
chosen_attrs = [
    "Blond_Hair",
    "Bald",
    "Male",
    "No_Beard",
    "Black_Hair",
]
attr_dim = len(chosen_attrs)  # Doit être égal à 5

image_size = 128
batch_size = 64
epochs = 2

# Coefficients de perte selon le papier
lambda_rec = 100  # lambda_1
lambda_clsg = 10.0   # lambda_2
lambda_clsc = 1.0    # lambda_3

# Taux d'apprentissage
lr_disc = 1e-4
lr_gen  = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# 3) Transformations & DataLoader
##############################################################################
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
dataset = CelebADataset(img_dir, attr_path, chosen_attrs, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

##############################################################################
# 4) Définition des modules (Encoder, Decoder, Discriminateur/Classifier)
##############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- Encoder ---
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
# 5) Initialisation des modèles & Optimiseurs
##############################################################################
encoder = Encoder(input_dim=3).to(device)
decoder = Decoder(output_dim=3, attr_dim=attr_dim).to(device)
classifier_discriminator = ClassifierDiscriminator(input_dim=3, attr_dim=attr_dim).to(device)

encoder.apply(weights_init)
decoder.apply(weights_init)
classifier_discriminator.apply(weights_init)

opt_enc = optim.Adam(encoder.parameters(), lr=lr_gen, betas=(0.5, 0.999))
opt_dec = optim.Adam(decoder.parameters(), lr=lr_gen, betas=(0.5, 0.999))
opt_disc = optim.Adam(classifier_discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

# Critères de perte
criterion_adv = nn.BCEWithLogitsLoss()
criterion_cls = nn.BCEWithLogitsLoss()  # Pour la classification
criterion_rec = nn.L1Loss()             # Pour la reconstruction

##############################################################################
# 6) Boucle d'entraînement
##############################################################################
for epoch in range(epochs):
    for i, (images, real_attrs) in enumerate(loader):
        images = images.to(device)         # (N, 3, 128, 128)
        real_attrs = real_attrs.to(device)   # (N, 5)
        cur_batch = images.size(0)

        # --- Création du vecteur d'attributs manipulé ---
        # Pour chaque image, on choisit un nombre k aléatoire (entre 1 et 5)
        # et on flip les k premiers indices d'une permutation aléatoire
        manipulated_attrs = real_attrs.clone()
        for idx in range(cur_batch):
            k = torch.randint(1, attr_dim + 1, (1,)).item()  # nombre d'attributs à changer
            perm = torch.randperm(attr_dim, device=device)
            flip_idx = perm[:k]
            manipulated_attrs[idx, flip_idx] = 1 - manipulated_attrs[idx, flip_idx]


        # ------------- Entraînement du Discriminateur -------------
        # On ne considère ici que la branche d'édition pour l'adversarial.
        with torch.no_grad():
            latent, skips = encoder(images)
            fake_recon_images = decoder(latent, real_attrs, skips)   # branche reconstruction (pour L_rec)
            fake_edit_images  = decoder(latent, manipulated_attrs, skips)  # branche édition

        # Prédictions sur images réelles
        real_attr_preds, real_scores = classifier_discriminator(images)
        # Prédictions sur images éditées (fake)
        _, fake_edit_scores = classifier_discriminator(fake_edit_images.detach())

        # Labels (avec label smoothing)
        real_labels = 0.9 * torch.ones(cur_batch, 1, device=device)
        fake_labels = 0.1 * torch.ones(cur_batch, 1, device=device)

        # Perte adversariale pour le discriminateur
        L_adv_D = criterion_adv(real_scores, real_labels) + criterion_adv(fake_edit_scores, fake_labels)
        # Perte de classification sur images réelles
        L_clsc = criterion_cls(real_attr_preds, real_attrs)
        # Selon le papier : L_disc = L_adv^D + lambda3 * L_clsc, avec lambda3 = 1
        loss_disc = L_adv_D + lambda_clsc * L_clsc 

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # ------------- Entraînement du Générateur (Encoder + Decoder) -------------
        latent, skips = encoder(images)
        fake_recon_images = decoder(latent, real_attrs, skips)   # branche reconstruction (pour L_rec)
        fake_edit_images  = decoder(latent, manipulated_attrs, skips)  # branche édition
        # Pour la reconstruction, comparer fake_recon_images à images
        L_rec = criterion_rec(fake_recon_images, images)
        # Pour la classification (édition), comparer les prédictions des fake_edit_images aux attributs manipulés
        fake_edit_attr_preds, fake_edit_scores = classifier_discriminator(fake_edit_images)
        L_clsg = criterion_cls(fake_edit_attr_preds, manipulated_attrs) + 0.01 * torch.norm(fake_edit_attr_preds, p=2)
        # Perte adversariale pour que fake_edit_images soient considérées comme réelles
        L_adv_G = criterion_adv(fake_edit_scores, real_labels)
        # Selon le papier : L_gen = lambda1 * L_rec + lambda2 * L_clsg + L_adv^G
        loss_gen = 100*L_rec + 10*L_clsg + L_adv_G

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        loss_gen.backward()
        opt_enc.step()
        opt_dec.step()

        # Affichage périodique des logs
        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Step {i+1}/{len(loader)} | "
                  f"L_disc: {loss_disc.item():.4f} | L_gen: {loss_gen.item():.4f} | "
                  f"L_rec: {L_rec.item():.4f} | L_clsg: {L_clsg.item():.4f} | "
                  f"L_adv_G: {L_adv_G.item():.4f} | L_clsc: {L_clsc.item():.4f}")

    print(f"Fin de l'epoch {epoch+1}/{epochs} | L_disc: {loss_disc.item():.4f} | L_gen: {loss_gen.item():.4f}")
    print("#####################")
    print(manipulated_attrs, real_attrs)
    print("-------")
    print(fake_edit_attr_preds)
    print("-------")
    print(real_attr_preds)

##############################################################################
# 7) Sauvegarde des modèles
##############################################################################
#save_dir = "./saved_models_last"
#save_dir="./saved_models_3_epochs_change_parameters_fake"
save_dir="./saved_model_4O_recommendation_13"
os.makedirs(save_dir, exist_ok=True)
torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
torch.save(decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
torch.save(classifier_discriminator.state_dict(), os.path.join(save_dir, "classifier_discriminator.pth"))
torch.save(opt_enc.state_dict(), os.path.join(save_dir, "opt_enc.pth"))
torch.save(opt_dec.state_dict(), os.path.join(save_dir, "opt_dec.pth"))
torch.save(opt_disc.state_dict(), os.path.join(save_dir, "opt_disc.pth"))

print("Entraînement terminé, modèles sauvegardés avec succès !")
