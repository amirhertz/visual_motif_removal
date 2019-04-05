import torch
import torch.nn as nn
from networks.unet_deeper import UnetEncoderD, UnetDecoderD


class UnetBaselineD(nn.Module):

    def __init__(self, in_channels=3, depth=5, shared_depth=0, use_vm_decoder=False, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True, batch_norm=True,
                 transpose=True, concat=True, transfer_data=True):
        super(UnetBaselineD, self).__init__()
        self.transfer_data = transfer_data
        self.shared = shared_depth
        self.optimizer_encoder,  self.optimizer_image, self.optimizer_vm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks)
        if not transfer_data:
            concat = False
        self.encoder = UnetEncoderD(in_channels=in_channels, depth=depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, batch_norm=batch_norm)
        self.image_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                          out_channels=out_channels_image, depth=depth - shared_depth,
                                          blocks=blocks[1], residual=residual, batch_norm=batch_norm,
                                          transpose=transpose, concat=concat)
        self.mask_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - 1),
                                         out_channels=out_channels_mask, depth=depth,
                                         blocks=blocks[2], residual=residual, batch_norm=batch_norm,
                                         transpose=transpose, concat=concat)
        self.vm_decoder = None
        if use_vm_decoder:
            self.vm_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                           out_channels=out_channels_image, depth=depth - shared_depth,
                                           blocks=blocks[3], residual=residual, batch_norm=batch_norm,
                                           transpose=transpose, concat=concat)
        self.shared_decoder = None
        self._forward = self.unshared_forward
        if self.shared != 0:
            self._forward = self.shared_forward
            self.shared_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - 1),
                                               out_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=shared_depth, blocks=blocks[4], residual=residual,
                                               batch_norm=batch_norm, transpose=transpose, concat=concat,
                                               is_final=False)

    def set_optimizers(self):
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.optimizer_image = torch.optim.Adam(self.image_decoder.parameters(), lr=0.001)
        self.optimizer_mask = torch.optim.Adam(self.mask_decoder.parameters(), lr=0.001)
        if self.vm_decoder is not None:
            self.optimizer_vm = torch.optim.Adam(self.vm_decoder.parameters(), lr=0.001)
        if self.shared != 0:
            self.optimizer_shared = torch.optim.Adam(self.shared_decoder.parameters(), lr=0.001)

    def zero_grad_all(self):
        self.optimizer_encoder.zero_grad()
        self.optimizer_image.zero_grad()
        self.optimizer_mask.zero_grad()
        if self.vm_decoder is not None:
            self.optimizer_vm.zero_grad()
        if self.shared != 0:
            self.optimizer_shared.zero_grad()

    def step_all(self):
        self.optimizer_encoder.step()
        self.optimizer_image.step()
        self.optimizer_mask.step()
        if self.vm_decoder is not None:
            self.optimizer_vm.step()
        if self.shared != 0:
            self.optimizer_shared.step()

    def step_optimizer_image(self):
        self.optimizer_image.step()

    def __call__(self, synthesized):
        return self._forward(synthesized)

    def forward(self, synthesized):
        return self._forward(synthesized)

    def unshared_forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if not self.transfer_data:
            before_pool = None
        reconstructed_image = torch.tanh(self.image_decoder(image_code, before_pool))
        reconstructed_mask = torch.sigmoid(self.mask_decoder(image_code, before_pool))
        if self.vm_decoder is not None:
            reconstructed_vm = torch.tanh(self.vm_decoder(image_code, before_pool))
            return reconstructed_image, reconstructed_mask, reconstructed_vm
        return reconstructed_image, reconstructed_mask

    def shared_forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if self.transfer_data:
            shared_before_pool = before_pool[- self.shared - 1:]
            unshared_before_pool = before_pool[: - self.shared]
        else:
            before_pool = None
            shared_before_pool = None
            unshared_before_pool = None
        x = self.shared_decoder(image_code, shared_before_pool)
        reconstructed_image = torch.tanh(self.image_decoder(x, unshared_before_pool))
        reconstructed_mask = torch.sigmoid(self.mask_decoder(image_code, before_pool))
        if self.vm_decoder is not None:
            reconstructed_vm = torch.tanh(self.vm_decoder(x, unshared_before_pool))
            return reconstructed_image, reconstructed_mask, reconstructed_vm
        return reconstructed_image, reconstructed_mask
