import torch
import os.path
from utils.train_utils import load_globals, init_folders, init_nets
from loaders.motif_dataset import MotifDS
from PIL import Image
import numpy as np


# network names
root_path = '..'
train_tag = 'vm_demo_text_remover'
load_tag = ''

device = torch.device('cuda:0')
net_path = '%s/checkpoints/%s' % (root_path, train_tag)
resources_root = 'test images folder'
target_root = '%s/data/tmp' % root_path


def load_image(image_path, _device, include_tensor=False):
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
            tensor_image = torch.unsqueeze(torch.from_numpy(tensor_image), 0).to(_device)
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


def collect_synthesized(_source):
    paths = []
    for root, _, files in os.walk(_source):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if (file_extension == '.png' or file_extension == '.jpg' or file_extension == '.jpeg') and \
                   ('real' not in file_name and 'reconstructed' not in file_name and 'grid' not in file_name):
                    paths.append(os.path.join(root, file))
    return paths


def save_numpy_image(images, suffix, _target_root, _resources_root, prefix='', start_count=0):
    images = (images * 255).astype(np.uint8)  # unnormalize
    for image_index in range(images.shape[0]):
        if prefix == '':
            image_path = '%s/%d_%s.png' % (_resources_root, image_index + start_count, suffix)
        else:
            image_path = '%s/%s_%s.png' % (_target_root, prefix, suffix)
        image = Image.fromarray(images[image_index])
        image.save(image_path)


def run_net(opt, _device, _net_path, _source, _target, _train_tag, _tag=''):
    net = init_nets(opt, _net_path, _device, _tag).eval()
    synthesized_paths = collect_synthesized(_source)
    image_suffixes = ['reconstructed_image', 'reconstructed_motif']
    for path in synthesized_paths:
        prefix, _ = os.path.splitext(os.path.split(path)[-1])
        prefix = prefix.split('_')[0]
        sy_np, sy_ts = load_image(path, _device, True)
        results = list(net(sy_ts))
        for idx, result in enumerate(results):
            results[idx] = transform_to_numpy_image(result)
        reconstructed_mask = results[1]
        reconstructed_motif = None
        if len(results) == 3:
            reconstructed_raw_motif = results[2]
            reconstructed_motif = (reconstructed_raw_motif - 1) * reconstructed_mask + 1
        reconstructed_image = reconstructed_mask * results[0] + (1 - reconstructed_mask) * sy_np
        for idx, image in enumerate([reconstructed_image, reconstructed_motif]):
            if image is not None and idx < len(image_suffixes):
                save_numpy_image(image, '%s_%s' % (image_suffixes[idx], _train_tag), _target, _source,
                                 prefix=prefix)
    print('done')


if __name__ == '__main__':
    _opt = load_globals(net_path, {}, override=False)
    init_folders(target_root)
    run_net(_opt, device, net_path, resources_root, target_root, train_tag, load_tag)
