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
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true', help='Use custom images')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/custom/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    return parser.parse_args(args)

args_ = parse()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.test_int = args_.test_int
args.num_test = args_.num_test
args.gpu = args_.gpu
args.load_epoch = args_.load_epoch
args.multi_gpu = args_.multi_gpu
args.custom_img = args_.custom_img
args.custom_data = args_.custom_data
args.custom_attr = args_.custom_attr
args.n_attrs = len(args.attributes)
args.betas = (args.beta1, args.beta2)

print(args)


if args.custom_img:
    output_path = join('output', args.experiment_name, 'custom_testing')
    test_dataset = CustomDataset(args.custom_data, args.custom_attr, args.img_size, args.attributes)
else:
    output_path = join('output', args.experiment_name, 'sample_testing')
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
for index, (real_image, real_attributes) in enumerate(test_dataloader):
    if args.num_test is not None and index == args.num_test:
        break
    
    real_image = real_image.cuda() if args.gpu else real_image
    real_attributes = real_attributes.cuda() if args.gpu else real_attributes
    real_attributes = real_attributes.type(torch.float)
    
    modified_attributes_list = [real_attributes]
    for i in range(args.n_attrs):
        temp_attributes = real_attributes.clone()
        temp_attributes[:, i] = 1 - temp_attributes[:, i]
        temp_attributes = check_attribute_conflict(temp_attributes, args.attributes[i], args.attributes)
        modified_attributes_list.append(temp_attributes)

    with torch.no_grad():
        generated_samples = [real_image]
        for i, modified_attributes in enumerate(modified_attributes_list):
            modified_attributes_scaled = (modified_attributes * 2 - 1) * args.thres_int
            if i > 0:
                modified_attributes_scaled[..., i - 1] = modified_attributes_scaled[..., i - 1] * args.test_int / args.thres_int
            generated_samples.append(attgan.generator(real_image, modified_attributes_scaled))
        generated_samples = torch.cat(generated_samples, dim=3)
        if args.custom_img:
            out_file = test_dataset.image_names[index]
        else:
            out_file = '{:06d}.jpg'.format(index + 182638)
        vutils.save_image(
            generated_samples, join(output_path, out_file),
            nrow=1, normalize=True, value_range=(-1., 1.)
        )
        print('{:s} done!'.format(out_file))
