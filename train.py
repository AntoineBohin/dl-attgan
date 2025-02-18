import argparse
import datetime
import json
import os
from os.path import join

import torch.utils.data as data
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from src.model import AttGAN
from src.dataset import check_attribute_conflict, CelebA
from src.utils import Progressbar, add_scalar_dict


DEFAULT_ATTRIBUTES = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Smiling', 'Young']

def parse(args=None):
    parser = argparse.ArgumentParser()
    
    ### DATASET CONFIGURATION ###
    parser.add_argument('--attributes', dest='attributes', default=DEFAULT_ATTRIBUTES, nargs='+', help='attributes to learn')
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, choices=['CelebA', 'CelebA-HQ'], default='CelebA')
    parser.add_argument('--dataset_path', dest='dataset_path', type=str, default='/usr/users/siapartnerscomsportif/bohin_ant/att-gan/data/celeba-dataset/versions/2/img_align_celeba/img_align_celeba')
    parser.add_argument('--attribute_labels_path', dest='attribute_labels_path', type=str, default='/usr/users/siapartnerscomsportif/bohin_ant/att-gan/data/celeba-dataset/versions/2/list_attr_celeba.csv')
    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default='data/image_list.txt')
    
    ### MODEL ARCHITECTURE ###
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1, help='Number of shortcut layers for residual connections.')
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=1, help='Number of layers where attributes are injected.')

    # Encoder settings (Matches Generator Class)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64, help='Base number of channels for the encoder.')
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5, help='Number of layers in the encoder.')
    parser.add_argument('--enc_norm_fn', dest='enc_norm_fn', type=str, default='batchnorm', help='Normalization function for the encoder.')
    parser.add_argument('--enc_acti_fn', dest='enc_acti_fn', type=str, default='lrelu', help='Activation function for the encoder.')

    # Decoder settings (Matches Generator Class)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64, help='Base number of channels for the decoder.')
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5, help='Number of layers in the decoder.')
    parser.add_argument('--dec_norm_fn', dest='dec_norm_fn', type=str, default='batchnorm', help='Normalization function for the decoder.')
    parser.add_argument('--dec_acti_fn', dest='dec_acti_fn', type=str, default='relu', help='Activation function for the decoder.')

    # Discriminator settings (Matches Discriminator Class)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64, help='Base number of channels for the discriminator.')
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5, help='Number of layers in the discriminator.')
    parser.add_argument('--dis_norm_fn', dest='dis_norm_fn', type=str, default='instancenorm', help='Normalization function for the discriminator.')
    parser.add_argument('--dis_acti_fn', dest='dis_acti_fn', type=str, default='lrelu', help='Activation function for the discriminator.')

    # Discriminator Fully Connected Layer Settings
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024, help='Number of neurons in the discriminator\'s fully connected layer.')
    parser.add_argument('--dis_fc_norm_fn', dest='dis_fc_norm_fn', type=str, default='none')
    parser.add_argument('--dis_fc_acti_fn', dest='dis_fc_acti_fn', type=str, default='relu')
    
    ### TRAINING CONFIGURATION ###
    parser.add_argument('--lambda_reconstruction', dest='lambda_reconstruction', type=float, default=100.0)
    parser.add_argument('--lambda_attribute_classification', dest='lambda_attribute_classification', type=float, default=10.0)
    parser.add_argument('--lambda_adversarial_classification', dest='lambda_adversarial_classification', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0, help='Weight for the gradient penalty loss in WGAN-GP.')
    
    parser.add_argument('--gan_mode', dest='gan_mode', default='wgan', choices=['wgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=6, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    
    ### ATTRIBUTE SAMPLING ###
    parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')
    
    ### CHECKPOINT, LOGGING, AND VISUALIZATION ###
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    
    return parser.parse_args(args)

args = parse()
print(args)

args.learning_rate_base = args.learning_rate
args.n_attrs = len(args.attributes)
args.betas = (args.beta1, args.beta2)

os.makedirs(join('output', args.experiment_name), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'sample_training'), exist_ok=True)

with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

#Load Dataset
train_dataset = CelebA(args.dataset_path, args.attribute_labels_path, args.img_size, 'train', args.attributes)
valid_dataset = CelebA(args.dataset_path, args.attribute_labels_path, args.img_size, 'valid', args.attributes)

for i in range(5):
    print(f"Image: {train_dataset.image_names[i]}, Labels: {train_dataset.attribute_labels[i]}")

# Data loaders
train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
valid_dataloader = data.DataLoader(valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers, shuffle=False, drop_last=False)
print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

#Initialize AttGAN model
attgan = AttGAN(args)
progressbar = Progressbar()
writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

# Prepare fixed images for evaluation
fixed_real_images, fixed_real_attributes = next(iter(valid_dataloader))
fixed_real_images = fixed_real_images.cuda() if args.gpu else fixed_real_images
fixed_real_attributes = fixed_real_attributes.cuda() if args.gpu else fixed_real_attributes
fixed_real_attributes = fixed_real_attributes.type(torch.float)

# Generate attribute modifications for evaluation
sampled_target_attributes_list = [fixed_real_attributes]
for i in range(args.n_attrs):
    tmp_attributes = fixed_real_attributes.clone()
    tmp_attributes[:, i] = 1 - tmp_attributes[:, i]
    tmp_attributes = check_attribute_conflict(tmp_attributes, args.attributes[i], args.attributes)
    sampled_target_attributes_list.append(tmp_attributes)

it = 0
it_per_epoch = len(train_dataset) // args.batch_size
for epoch in range(args.epochs):
    # Adjust learning rate: Full rate for first 100 epochs, then reduce by factor of 10 every 100 epochs
    learning_rate = args.learning_rate_base / (10 ** (epoch // 100))
    attgan.set_learning_rate(learning_rate)
    writer.add_scalar('LR/learning_rate', learning_rate, it+1)

    for real_images, real_attributes in progressbar(train_dataloader):
        attgan.train()
        
        real_images = real_images.cuda() if args.gpu else real_images
        real_attributes = real_attributes.cuda() if args.gpu else real_attributes

        # Shuffle attributes to generate new target attributes
        shuffled_indices = torch.randperm(len(real_attributes))
        target_attributes = real_attributes[shuffled_indices].contiguous()
        
        real_attributes = real_attributes.type(torch.float)
        target_attributes = target_attributes.type(torch.float)
        
        # Normalize attributes for input to generator
        normalized_real_attributes = (real_attributes * 2 - 1) * args.thres_int

        if args.b_distribution == 'none':
            normalized_target_attributes = (target_attributes * 2 - 1) * args.thres_int
        elif args.b_distribution == 'uniform':
            normalized_target_attributes = (target_attributes * 2 - 1) * \
                                           torch.rand_like(target_attributes) * (2 * args.thres_int)
        elif args.b_distribution == 'truncated_normal':
            normalized_target_attributes = (target_attributes * 2 - 1) * \
                                           (torch.fmod(torch.randn_like(target_attributes), 2) + 2) / 4.0 * (2 * args.thres_int)
        
        # Train discriminator multiple times before updating generator
        if (it+1) % (args.n_d+1) != 0:
            loss_discriminator = attgan.train_discriminator(real_images, real_attributes, normalized_real_attributes, target_attributes, normalized_target_attributes)
            add_scalar_dict(writer, loss_discriminator, it + 1, 'Discriminator')
        else:
            loss_generator = attgan.train_generator(real_images, real_attributes, normalized_real_attributes, target_attributes, normalized_target_attributes)
            add_scalar_dict(writer, loss_generator, it + 1, 'Generator')
            progressbar.say(epoch=epoch, iter=it + 1, d_loss=loss_discriminator['discriminator_loss'], g_loss=loss_generator['generator_loss'])
        
        # Save model checkpoints at specified intervals
        if (it+1) % args.save_interval == 0:
            attgan.save_generator(os.path.join(
                'output', args.experiment_name, 'checkpoint', f'weights.{epoch}.pth'
            ))
            # attgan.save(os.path.join(
            #     'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            # ))
        if (it+1) % args.sample_interval == 0:
            attgan.eval()
            with torch.no_grad():
                generated_samples = [fixed_real_images]
                for i, target_attributes in enumerate(sampled_target_attributes_list):
                    normalized_target_attributes = (target_attributes * 2 - 1) * args.thres_int
                    if i > 0:
                        normalized_target_attributes[..., i - 1] = normalized_target_attributes[..., i - 1] * args.test_int / args.thres_int
                    generated_samples.append(attgan.generator(fixed_real_images, normalized_target_attributes))
                generated_samples = torch.cat(generated_samples, dim=3)
                writer.add_image('sample', vutils.make_grid(generated_samples, nrow=1, normalize=True, value_range=(-1., 1.)), it + 1)
                vutils.save_image(generated_samples, os.path.join(
                        'output', args.experiment_name, 'sample_training',
                        'Epoch_({:d})_({:d}of{:d}).jpg'.format(epoch, it%it_per_epoch+1, it_per_epoch)
                    ), nrow=1, normalize=True, value_range=(-1., 1.))
        it += 1
