from utils.train_utils import *
from torch import nn

# paths
root_path = '..'
train_tag = 'demo_emojis'


# datasets paths
cache_root = ['data folder a', 'data folder b', '...']

# dataset configurations
patch_size = 128
image_size = 512

# network
nets_path = '%s/checkpoints/%s' % (root_path, train_tag)
images_path = '%s/images' % nets_path

num_blocks = (3, 3, 3, 3, 3)
shared_depth = 2
use_vm_decoder = True


# train configurations
gamma1 = 2   # L1 image
gamma2 = 1   # L1 visual motif
epochs = 200
batch_size = 32
print_frequency = 100
save_frequency = 10
device = torch.device('cuda:0')


def l1_relative(reconstructed, real, batch, area):
    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / area
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1


def train(net, train_loader, test_loader):
    bce = nn.BCELoss()
    net.set_optimizers()
    losses = []
    print('Training Begins')
    for epoch in range(epochs):
        real_epoch = epoch + 1
        for i, data in enumerate(train_loader, 0):
            synthesized, images, vm_mask, motifs, vm_area = data
            synthesized, images, = synthesized.to(device), images.to(device)
            vm_mask, vm_area = vm_mask.to(device), vm_area.to(device)
            results = net(synthesized)
            guess_images, guess_mask = results[0], results[1]
            expanded_vm_mask = vm_mask.repeat(1, 3, 1, 1)
            reconstructed_pixels = guess_images * expanded_vm_mask
            real_pixels = images * expanded_vm_mask
            batch_cur_size = vm_mask.shape[0]
            net.zero_grad_all()
            loss_l1_images = l1_relative(reconstructed_pixels, real_pixels, batch_cur_size, vm_area)
            loss_mask = bce(guess_mask, vm_mask)
            loss_l1_vm = 0
            if len(results) == 3:
                guess_vm = results[2]
                reconstructed_motifs = guess_vm * expanded_vm_mask
                real_vm = motifs.to(device) * expanded_vm_mask
                loss_l1_vm = l1_relative(reconstructed_motifs, real_vm, batch_cur_size, vm_area)
            loss = gamma1 * loss_l1_images + gamma2 * loss_l1_vm + loss_mask
            loss.backward()
            net.step_all()
            losses.append(loss.item())
            # print
            if (i + 1) % print_frequency == 0:
                print('%s [%d, %3d] , baseline loss: %.2f' % (train_tag, real_epoch, batch_size * (i + 1), sum(losses) / len(losses)))
                losses = []
        # savings
        if real_epoch % save_frequency == 0:
            print("checkpointing...")
            image_name = '%s/%s_%d.png' % (images_path, train_tag, real_epoch)
            _ = save_test_images(net, test_loader, image_name, device)
            torch.save(net.state_dict(), '%s/net_baseline.pth' % nets_path)
            torch.save(net.state_dict(), '%s/net_baseline_%d.pth' % (nets_path, real_epoch))

    print('Training Done:)')


def run():
    init_folders(nets_path, images_path)
    opt = load_globals(nets_path, globals(), override=True)
    train_loader, test_loader = init_loaders(opt, cache_root=cache_root)
    base_net = init_nets(opt, nets_path, device)
    train(base_net, train_loader, test_loader)


if __name__ == '__main__':
    run()
