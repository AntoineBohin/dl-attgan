import argparse
import json
import os
from os.path import join

import torch
import torch.utils.data as data
import torchvision.utils as vutils

from src.model import AttGAN
from src.dataset import CelebA, CustomDataset, check_attribute_conflict
from src.utils import Progressbar, find_model

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', required=True)
    parser.add_argument('--test_att', dest='test_att', required=True, help='test_att')
    parser.add_argument('--test_int_min', dest='test_int_min', type=float, default=-1.0, help='test_int_min')
    parser.add_argument('--test_int_max', dest='test_int_max', type=float, default=1.0, help='test_int_max')
    parser.add_argument('--n_slide', dest='n_slide', type=int, default=10, help='n_slide')
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    return parser.parse_args(args)

args_ = parse()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.test_att = args_.test_att
args.test_int_min = args_.test_int_min
args.test_int_max = args_.test_int_max
args.n_slide = args_.n_slide
args.num_test = args_.num_test
args.load_epoch = args_.load_epoch
args.custom_img = args_.custom_img
args.custom_data = args_.custom_data
args.custom_attr = args_.custom_attr
args.gpu = args_.gpu
args.multi_gpu = args_.multi_gpu

print(args)

assert args.test_att is not None, 'test_att should be chosen in %s' % (str(args.attributes))

if args.custom_img:
    output_path = join('output', args.experiment_name, 'custom_testing_slide_' + args.test_att)
    test_dataset = CustomDataset(args.custom_data, args.custom_attr, args.img_size, args.attributes)
else:
    output_path = join('output', args.experiment_name, 'sample_testing_slide_' + args.test_att)
    if args.dataset_name == 'CelebA':
        test_dataset = CelebA(args.dataset_path, args.attribute_labels_path, args.img_size, 'test', args.attributes)
os.makedirs(output_path, exist_ok=True)
test_dataloader = data.DataLoader(
    test_dataset, batch_size=8, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
if args.num_test is None:
    print('Testing images:', len(test_dataset))
else:
    print('Testing images:', min(len(test_dataset), args.num_test))


attgan = AttGAN(args)
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
progressbar = Progressbar()

attgan.eval()
for index, (real_images, real_attributes) in enumerate(test_dataloader):
    if args.num_test is not None and index == args.num_test:
        break
    
    real_images = real_images.cuda() if args.gpu else real_images
    real_attributes = real_attributes.cuda() if args.gpu else real_attributes
    real_attributes = real_attributes.type(torch.float)
    modified_attributes = real_attributes.clone()

    with torch.no_grad():
        generated_samples = [real_images]
        for i in range(args.n_slide):
            test_int = (args.test_int_max - args.test_int_min) / (args.n_slide - 1) * i + args.test_int_min
            modified_attributes_scaled = (modified_attributes * 2 - 1) * args.thres_int
            modified_attributes_scaled[..., args.attributes.index(args.test_att)] = test_int
            generated_samples.append(attgan.generator(real_images, modified_attributes_scaled))
        generated_samples = torch.cat(generated_samples, dim=3)
        if args.custom_img:
            out_file = test_dataset.images[index]
        else:
            out_file = '{:06d}.jpg'.format(index + 182638)
        vutils.save_image(
            generated_samples, join(output_path, out_file),
            nrow=1, normalize=True, value_range=(-1., 1.)
        )
        print('{:s} done!'.format(out_file))
