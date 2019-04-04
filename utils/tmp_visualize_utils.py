import torch
import os.path
from utils.image_utils import crop_image
from utils.train_utils import load_globals, init_loaders, init_folders, init_nets
from loaders.motif_dataset import MotifDS
from PIL import Image
import numpy as np


# network names
root_path = '..'
# train_tag = 'btl_mega_all'
# train_tag = 'btl_mega_blur_all'
train_tag = 'btl_final3_all'
# train_tag = 'btl_trans_texts_shapes_all'
# globals
device = torch.device('cuda:2')

net_path = '%s/checkpoints/%s' % (root_path, train_tag)
# net_path = '/mnt/data/amir/water/checkpoints/%s' % train_tag

resources_root = '%s/data/test_images/trialswm' % root_path
# resources_root = '%s/data/to_jpg/white_texts' % root_path
target_root = resources_root

tag = ''
# resources_root = '/mnt/data/amir/water/cache/emojis_pert_op_all/colF'
# target_root = '/mnt/data/amir/water/cache/emojis_pert_op_all/eval_sanity'



def load_image(image_path, include_tensor=False):
    numpy_image = None
    tensor_image = None
    if os.path.isfile(image_path):
        to_save = False
        row_image = Image.open(image_path)
        w, h = row_image.size
        if h > 512:
            to_save = True
            h = int((512. * h) / w)
            row_image = row_image.resize((512, h), Image.BICUBIC)
        w, h = row_image.size
        if w % 16 != 0 or h % 16 != 0:
            to_save = True
            row_image = row_image.crop((0, 0, (w // 16) * 16, (h // 16) * 16))
        if to_save:
            row_image.save(image_path)
        numpy_image = np.array(row_image)
        if len(numpy_image.shape) != 3:
            numpy_image = np.repeat(np.expand_dims(numpy_image, 2), 3, axis=2)
        if numpy_image.shape[2] != 3:
            numpy_image = numpy_image[:, :, :3]
        if include_tensor:
            tensor_image = MotifDS.trans(MotifDS.flip(numpy_image)[0])[0]
            tensor_image = torch.unsqueeze(torch.from_numpy(tensor_image), 0).to(device)
        numpy_image = np.expand_dims(numpy_image / 255, 0)
    return numpy_image, tensor_image


def transform_to_numpy_image(tensor_image):
    image = tensor_image.cpu().detach().numpy()
    image = np.transpose(image, (0, 2, 3, 1))
    if image.shape[3] != 3:
        image = np.repeat(image, 3, axis=3)
    else:
        image = (image / 2 + 0.5)
    return image


def collect_synthesized():
    paths = []
    for root, _, files in os.walk(resources_root):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if (file_extension == '.png' or file_extension == '.jpg' or file_extension == '.jpeg') and \
                   ('real' not in file_name and 'reconstructed' not in file_name and 'grid' not in file_name):
                    paths.append(os.path.join(root, file))
    return paths


def save_numpy_image(images, suffix, prefix='', start_count=0):
    images = (images * 255).astype(np.uint8)  # unnormalize
    for image_index in range(images.shape[0]):
        if prefix == '':
            image_path = '%s/%d_%s.png' % (resources_root, image_index + start_count, suffix)
        else:
            image_path = '%s/%s_%s.png' % (target_root, prefix, suffix)
        image = Image.fromarray(images[image_index])
        image.save(image_path)


def run_net(opt):
    net = init_nets(opt, net_path, device, tag).eval()
    synthesized_paths = collect_synthesized()

    image_suffixes = ['reconstructed_image', 'reconstructed_mask']
    for path in synthesized_paths:
        prefix, _ = os.path.splitext(os.path.split(path)[-1])
        prefix = prefix.split('_')[0]
        sy_np, sy_ts = load_image(path, True)
        real_mask, _ = load_image('%s/%s_real_mask.png' % (resources_root, prefix))
        real_box, _ = load_image('%s/%s_real_box.png' % (resources_root, prefix))
        results = list(net(sy_ts))
        for idx, result in enumerate(results):
            results[idx] = transform_to_numpy_image(result)
        reconstructed_mask = results[1]
        reconstructed_watermark = None
        if len(results) == 3:
            reconstructed_raw_watermark = results[2]
            reconstructed_watermark = (reconstructed_raw_watermark - 1) * reconstructed_mask + 1

        reconstructed_image = reconstructed_mask * results[0] + (1 - reconstructed_mask) * sy_np
        for idx, image in enumerate([reconstructed_image, reconstructed_mask]):
            if image is not None and idx < len(image_suffixes):
                save_numpy_image(image, '%s_%s' % (image_suffixes[idx], train_tag), prefix=prefix)
    print('done')


if __name__ == '__main__':
    _opt = load_globals(net_path, {}, override=False)
    init_folders(target_root)
    run_net(_opt)
