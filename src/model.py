import torch
import torch.nn as nn
from network import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from torchsummary import summary
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim


# This architecture is for images of 128x128
# In the original AttGAN, slim.conv2d uses padding 'same'
MAX_DIM = 64 * 16  # 1024

class Generator(nn.Module):
    """
    Generator network for AttGAN.
    This network takes an image and its attributes, encodes it into a latent space,
    then decodes it back into an image with modified attributes.
    """
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=0, img_size=128):
        """
        Initialize the generator network.
        Args:
            enc_dim (int): Base number of channels for the encoder.
            enc_layers (int): Number of encoding layers.
            enc_norm_fn (str): Normalization function for encoder.
            enc_acti_fn (str): Activation function for encoder.
            dec_dim (int): Base number of channels for the decoder.
            dec_layers (int): Number of decoding layers.
            dec_norm_fn (str): Normalization function for decoder.
            dec_acti_fn (str): Activation function for decoder.
            n_attrs (int): Number of attributes for conditional generation.
            shortcut_layers (int): Number of layers where residual connections are used.
            inject_layers (int): Number of layers where attributes are injected.
            img_size (int): Input image size.
        """
        super(Generator, self).__init__()
        # Limit shortcut and injection layers to valid values
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        
        ###### ENCODER ######
        layers = []
        n_in = 3 #3 channels for input
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn)]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers) # Store layers in a module list
        
        ###### DECODER ######
        layers = []
        # Add attribute vector to latent space
        n_in = n_in + n_attrs  
        for i in range(dec_layers):
            if i < dec_layers - 1: # All layers except the last one
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)  # Decrease channels
                layers += [ConvTranspose2dBlock(n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn)]
                n_in = n_out # Update input channels for next layer
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in # Add shortcut connection if applicable
                n_in = n_in + n_attrs if self.inject_layers > i else n_in # Inject attribute vector at this layer if required
            else: # Last layer outputs an RGB image
                layers += [ConvTranspose2dBlock(n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh')]
        self.dec_layers = nn.ModuleList(layers)
    
    def encode(self, x):
        """Encode the input image into a feature map."""
        z = x
        latent = []
        for layer in self.enc_layers:
            z = layer(z)
            latent.append(z)
        return latent
    
    def decode(self, latent, attributes):
        """Decode the latent representation back into an image, applying attribute modifications."""
        # Expand the attribute vector to match the feature map size
        att_tile = attributes.view(attributes.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        # Concatenate attribute vector with the last encoded feature map
        z = torch.cat([latent[-1], att_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Apply shortcut connections 
                z = torch.cat([z, latent[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i: # Inject attribute vector again at specified layers
                att_tile = attributes.view(attributes.size(0), -1, 1, 1) \
                          .repeat(1, 1, self.f_size * 2**(i+1), self.f_size * 2**(i+1))
                z = torch.cat([z, att_tile], dim=1)
        return z
    
    def forward(self, x, a=None, mode='enc-dec'):
        """Forward pass for the generator.
        Output depends on the mode:
                - 'enc-dec': Returns modified image.
                - 'enc': Returns encoded feature maps.
                - 'dec': Decodes an existing encoded representation with new attributes.
        """
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)

class Discriminators(nn.Module):
    """
    Discriminator network for AttGAN.
    This network takes an image and classifies it into real or fake while also predicting its attributes.
    """
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu', n_attrs=13,
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        """
        Initialize the discriminator network.
        Args:
            dim (int): Base number of channels for the discriminator.
            norm_fn (str): Normalization function for convolutional layers.
            acti_fn (str): Activation function for convolutional layers.
            fc_dim (int): Number of neurons in the fully connected layers.
            fc_norm_fn (str): Normalization function for fully connected layers.
            fc_acti_fn (str): Activation function for fully connected layers.
            n_layers (int): Number of convolutional layers.
            img_size (int): Input image size.
        """
        super(Discriminators, self).__init__()
        # Compute the final feature map size after downsampling
        self.f_size = img_size // 2**n_layers 
        
        ##### Convolutional Feature Extractor #####
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn)]
            n_in = n_out
        self.conv = nn.Sequential(*layers)

        ##### FC Layers for Adversarial Discrimination #####
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn), #FC Layer
            LinearBlock(fc_dim, 1, 'none', 'none') # Single output for real/fake classification
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn), #FC Hidden Layer
            LinearBlock(fc_dim, n_attrs, 'none', 'none') # Output layer predicting the attributes
        )
    
    def forward(self, x):
        """Forward pass for the discriminator."""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h)


# multilabel_soft_margin_loss = sigmoid + binary_cross_entropy

class AttGAN():
    def __init__(self, args):
        self.gan_mode = args.gan_mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_reconstruction = args.lambda_reconstruction
        self.lambda_attribute_classification = args.lambda_attribute_classification
        self.lambda_adversarial_classification = args.lambda_adversarial_classification
        self.lambda_gp = args.lambda_gp
        
        # Initialize the GENERATOR
        self.generator = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm_fn, args.enc_acti_fn,
            args.dec_dim, args.dec_layers, args.dec_norm_fn, args.dec_acti_fn,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.generator.train()  # Set model to training mode
        if self.gpu: self.generator.cuda()
        summary(self.generator, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        # Initialize the DISCRIMINATOR
        self.discriminator = Discriminators(
            args.dis_dim, args.dis_norm_fn, args.dis_acti_fn, args.n_attrs,
            args.dis_fc_dim, args.dis_fc_norm_fn, args.dis_fc_acti_fn, args.dis_layers, args.img_size
        )
        self.discriminator.train()
        if self.gpu: self.discriminator.cuda()
        summary(self.discriminator, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        
        # Initialize optimizers
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=args.learning_rate, betas=args.betas)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.learning_rate, betas=args.betas)
    
    def set_learning_rate(self, learning_rate):
        """Update the learning rate for both Generator and Discriminator."""
        for param_group in self.generator_optimizer.param_groups:
            param_group['lr'] = learning_rate
        for param_group in self.discriminator_optimizer.param_groups:
            param_group['lr'] = learning_rate
    
    def train_generator(self, real_images, real_attributes, real_attributes_normalized, target_attributes, target_attributes_normalized):
        # Freeze Discriminator parameters during Generator training
        for param in self.discriminator.parameters():
            param.requires_grad = False
        
        # Encode real images into latent space
        latent_codes = self.generator(real_images, mode='enc')
        # Generate fake images with modified attributes
        generated_images = self.generator(latent_codes, target_attributes_normalized, mode='dec')
        reconstructed_images = self.generator(latent_codes, real_attributes_normalized, mode='dec')

        discriminator_fake_output, discriminator_fake_classification = self.discriminator(generated_images)
        
        # Adversarial loss for WGAN
        adversarial_loss = -discriminator_fake_output.mean()
        attribute_classification_loss = F.binary_cross_entropy_with_logits(discriminator_fake_classification, target_attributes)
        reconstruction_loss = F.l1_loss(reconstructed_images, real_images)
        generator_loss = adversarial_loss + self.lambda_attribute_classification * attribute_classification_loss + self.lambda_reconstruction * reconstruction_loss
        
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()
        
        return {
            'generator_loss': generator_loss.item(),
            'adversarial_loss': adversarial_loss.item(),
            'attribute_classification_loss': attribute_classification_loss.item(),
            'reconstruction_loss': reconstruction_loss.item()
        }
    
    def train_discriminator(self, real_images, real_attributes, real_attributes_normalized, target_attributes, target_attributes_normalized):
        # Unfreeze Discriminator parameters
        for param in self.discriminator.parameters():
            param.requires_grad = True
        
        generated_images = self.generator(real_images, target_attributes_normalized).detach()

        # Evaluate real and fake images using Discriminator
        discriminator_real_output, discriminator_real_classification = self.discriminator(real_images)
        discriminator_fake_output, _ = self.discriminator(generated_images)
        
        # Gradient Penalty function (for WGAN-GP)
        def compute_gradient_penalty(critic, real_samples, fake_samples=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                return a + alpha * (b - a)
            
            interpolated_samples = interpolate(real_samples, fake_samples).requires_grad_(True)
            critic_output = critic(interpolated_samples)[0]
            if isinstance(critic_output, tuple):
                critic_output = critic_output[0]
            gradients = autograd.grad(
                outputs=critic_output, inputs=interpolated_samples,
                grad_outputs=torch.ones_like(critic_output),
                create_graph=True, retain_graph=True, only_inputs=True)[0]
            return ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1.0) ** 2).mean()
        
        wasserstein_distance = discriminator_real_output.mean() - discriminator_fake_output.mean()
        discriminator_loss = -wasserstein_distance
        gradient_penalty = compute_gradient_penalty(self.discriminator, real_images, generated_images)

        classification_loss = F.binary_cross_entropy_with_logits(discriminator_real_classification, real_attributes)
        total_discriminator_loss = discriminator_loss + self.lambda_gp * gradient_penalty + self.lambda_adversarial_classification * classification_loss
        
        # Optimize Discriminator
        self.discriminator_optimizer.zero_grad()
        total_discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'total_discriminator_loss': total_discriminator_loss.item(),
            'gradient_penalty': gradient_penalty.item(),
            'classification_loss': classification_loss.item()
        }
    
    def train(self):
        """
        Sets both the Generator and Discriminator to training mode.
        """
        self.generator.train()
        self.discriminator.train()
    
    def eval(self):
        """
        Sets both the Generator and Discriminator to evaluation mode.
        """
        self.generator.eval()
        self.discriminator.eval()
    
    def save(self, file_path):
        """
        Saves the entire model state, including both Generator and Discriminator, as well as their optimizers.
        """
        model_states = {
            'generator': self.generator.state_dict(),  # Save Generator parameters
            'discriminator': self.discriminator.state_dict(),  # Save Discriminator parameters
            'generator_optimizer': self.generator_optimizer.state_dict(),  # Save Generator optimizer state
            'discriminator_optimizer': self.discriminator_optimizer.state_dict()  # Save Discriminator optimizer state
        }
        torch.save(model_states, file_path)  # Save to disk
    
    def load(self, file_path):
        """
        Loads a saved model state, including the Generator, Discriminator, 
        and their respective optimizers.

        Args:
            file_path (str): The path to the saved model file.
        """
        model_states = torch.load(file_path, map_location=lambda storage, loc: storage)  # Load state dictionary from file
        if 'generator' in model_states:
            self.generator.load_state_dict(model_states['generator'])  # Load Generator parameters
        if 'discriminator' in model_states:
            self.discriminator.load_state_dict(model_states['discriminator'])  # Load Discriminator parameters
        if 'generator_optimizer' in model_states:
            self.generator_optimizer.load_state_dict(model_states['generator_optimizer'])  # Load Generator optimizer state
        if 'discriminator_optimizer' in model_states:
            self.discriminator_optimizer.load_state_dict(model_states['discriminator_optimizer'])  # Load Discriminator optimizer state
    
    def save_generator(self, file_path):
        """
        Saves only the Generator's state dictionary. This is useful for cases where we only care about using the Generator for inference (e.g., image editing).
        """
        generator_state = {
            'generator': self.generator.state_dict()
        }
        torch.save(generator_state, file_path)  # Save to disk

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm_fn', dest='enc_norm_fn', type=str, default='batchnorm')
    parser.add_argument('--dec_norm_fn', dest='dec_norm_fn', type=str, default='batchnorm')
    parser.add_argument('--dis_norm_fn', dest='dis_norm_fn', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm_fn', dest='dis_fc_norm_fn', type=str, default='none')
    parser.add_argument('--enc_acti_fn', dest='enc_acti_fn', type=str, default='lrelu')
    parser.add_argument('--dec_acti_fn', dest='dec_acti_fn', type=str, default='relu')
    parser.add_argument('--dis_acti_fn', dest='dis_acti_fn', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti_fn', dest='dis_fc_acti_fn', type=str, default='relu')
    parser.add_argument('--lambda_reconstruction', dest='lambda_reconstruction', type=float, default=100.0)
    parser.add_argument('--lambda_attribute_classification', dest='lambda_attribute_classification', type=float, default=10.0)
    parser.add_argument('--lambda_adversarial_classification', dest='lambda_adversarial_classification', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gradient_penalty', type=float, default=10.0)
    parser.add_argument('--gan_mode', dest='gan_mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.n_attrs = 13
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)
