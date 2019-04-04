import torch
import os
import pickle
from loaders.cache_loader import CacheLoader
from torch.utils.data import DataLoader
from train.train_options import TrainOptions as Opt
from networks.baselines import UnetBaselineD, UnetDiscriminatorD
from utils.image_utils import imshow, save_image
from torchvision.utils import make_grid
from time import gmtime, strftime


def init_folders(*folders):
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)


def load_globals(nets_path, globals_dict, override=True):
    save_set = {
                'image_domain', 'wm_tag', 'images_root', 'wm_root', 'image_and_wm_suffix', 'watermark_size',
                'image_size', 'patch_size', 'perturbate', 'opacity_var', 'use_rgb', 'weight', 'shared_depth',
                'num_blocks', 'residual', 'batch_size', 'transfer_data', 'concat', 'use_wm_decoder', 'rotate_wm',
                'scale_wm', 'crop_wm', 'batch_wm', 'font', 'noise', 'text_border', 'additive', 'blur'
                }
    to_save = False
    params_file = '%s/train_params.pkl' % nets_path
    __opt = Opt()
    if os.path.isfile(params_file):
        print('loading options from %s/' % nets_path)
        with open(params_file, 'rb') as f:
            save_globals_dict = pickle.load(f)
    else:
        save_globals_dict = {}
    for item in save_set:
        if item not in save_globals_dict and item in globals_dict:
            to_save = True
            save_globals_dict[item] = globals_dict[item]
        if item in save_globals_dict:
            setattr(__opt, item, save_globals_dict[item])
            print('%s: %s' % (item, str(save_globals_dict[item])))
    if to_save and override:
        with open(params_file, 'wb') as f:
            pickle.dump(save_globals_dict, f, pickle.HIGHEST_PROTOCOL)
    return __opt


def init_loaders(opt, cache_root=''):

    train_dataset = CacheLoader(cache_root, train=True, patch_size=opt.patch_size)
    test_dataset = CacheLoader(cache_root, train=False, patch_size=None)
    # else:
    #     train_dataset = MultiWatermarkLoader(opt.images_root, opt.wm_root, train=True, image_size=opt.image_size,
    #                                          watermark_size=opt.watermark_size, patch_size=opt.patch_size,
    #                                          weight=opt.weight, noise=opt.noise, perturbate=opt.perturbate,
    #                                          opacity_var=opt.opacity_var, rgb=opt.use_rgb, scale_wm=opt.scale_wm,
    #                                          rotate_wm=opt.rotate_wm, crop_wm=opt.crop_wm, batch_wm=opt.batch_wm,
    #                                          font=opt.font, border=opt.text_border, additive=opt.additive)
    #     test_dataset = MultiWatermarkLoader(opt.images_root, opt.wm_root, train=False, image_size=opt.image_size,
    #                                         patch_size=None, watermark_size=opt.watermark_size, weight=opt.weight,
    #                                         noise=opt.noise, perturbate=opt.perturbate, opacity_var=opt.opacity_var,
    #                                         rgb=opt.use_rgb, scale_wm=opt.scale_wm, rotate_wm=opt.rotate_wm,
    #                                         crop_wm=opt.crop_wm, batch_wm=opt.batch_wm, font=opt.font,
    #                                         border=opt.text_border, additive=opt.additive)
    _train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    if opt.patch_size:
        batch_scale = int(opt.image_size / opt.patch_size)
        batch_scale **= 2
    else:
        batch_scale = 1
    _test_data_loader = DataLoader(test_dataset, batch_size=max(2, opt.batch_size // batch_scale),
                                   shuffle=True, num_workers=1)
    return _train_data_loader, _test_data_loader


def init_nets(opt, net_path, device, tag=''):
    if opt.noise:
        out_channels_mask = 3
    else:
        out_channels_mask = 1
    net_baseline = UnetBaselineD(shared_depth=opt.shared_depth, use_wm_decoder=opt.use_wm_decoder, concat=opt.concat,
                                 blocks=opt.num_blocks,out_channels_mask=out_channels_mask, residual=opt.residual,
                                 transfer_data=opt.transfer_data)
    if tag != '':
        tag = '_' + str(tag)
    cur_path = '%s/net_baseline%s.pth' % (net_path, tag)
    if os.path.isfile(cur_path) and eval('net_baseline') is not None:
        print('loading baseline from %s/' % net_path)
        net_baseline.load_state_dict(torch.load(cur_path, map_location=torch.device('cpu')))
    net_baseline = net_baseline.to(device)
    return net_baseline


def save_test_images(net, loader, image_name, device):
    net.eval()
    synthesized, images, wm_mask, _, wm_area = next(iter(loader))
    wm_mask = wm_mask.to(device)
    synthesized = synthesized.to(device)
    output = net(synthesized)
    guess_images, guess_mask = output[0], output[1]
    expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
    expanded_real_mask = wm_mask.repeat(1, 3, 1, 1)
    reconstructed_pixels = guess_images * expanded_guess_mask
    reconstructed_images = synthesized * (1 - expanded_guess_mask) + reconstructed_pixels
    transformed_guess_mask = expanded_guess_mask * 2 - 1
    expanded_real_mask = expanded_real_mask * 2 - 1
    if len(output) == 3:
        guess_wm = output[2]
        reconstructed_wm = (guess_wm - 1) * expanded_guess_mask + 1
        images_un = (torch.cat((synthesized, reconstructed_images, reconstructed_wm, transformed_guess_mask), 0))
    else:
        images_un = (torch.cat((synthesized, reconstructed_images, transformed_guess_mask, expanded_real_mask), 0))
    images_un = torch.clamp(images_un.data, min=-1, max=1)
    images_un = make_grid(images_un, nrow=synthesized.shape[0], padding=5, pad_value=1)
    save_image(images_un, image_name)
    net.train()
    return images_un


def save_test_auto(net, loader, image_name, device):
    net.eval()
    synthesized, _, _, _, _, _, _ = next(iter(loader))
    synthesized = synthesized.to(device)
    output = net(synthesized)
    guess_images = output[0]
    images_un = (torch.cat((synthesized, guess_images), 0))
    images_un = torch.clamp(images_un.data, min=-1, max=1)
    images_un = make_grid(images_un, nrow=synthesized.shape[0], padding=5, pad_value=1)
    save_image(images_un, image_name)
    net.train()
    return images_un


def show_test(net, loader, opt, images_path, device, auto=False):
    current_time = strftime("%m-%d_%H-%M", gmtime())
    image_name = '%s/%s_%s_test_%s.%s' % (images_path, opt.wm_tag, opt.image_domain, current_time, opt.image_and_wm_suffix)
    if auto:
        images = save_test_auto(net, loader, image_name, device)
    else:
        images = save_test_images(net, loader, image_name, device)
    imshow(images)
