import kagglehub

# Download latest version
path = "./zz_AttGAN"
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

print("Path to dataset files:", path)
