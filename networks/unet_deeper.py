from networks.unet_components import *


class UnetDecoderD(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True):
        super(UnetDecoderD, self).__init__()
        self.conv_final = None
        self.up_convs = []
        outs = in_channels
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat)
            self.up_convs.append(up_conv)
        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpConvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat)
            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

    def forward(self, x, encoder_outs=None):
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            x = up_conv(x, before_pool)
        if self.conv_final is not None:
            x = self.conv_final(x)
        return x


class UnetEncoderD(nn.Module):

    def __init__(self, in_channels=3, depth=5, blocks=1, start_filters=32, residual=True, batch_norm=True):
        super(UnetEncoderD, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConvD(ins, outs, blocks, pooling=pooling, residual=residual, batch_norm=batch_norm)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs


class UnetEncoderDecoderD(nn.Module):
    def __init__(self, in_channels=3, depth=5, blocks_encoder=1, blocks_decoder=1, out_channels=3,
                 start_filters=32, residual=True, batch_norm=True, transpose=True, concat=True, transfer_data=True,
                 activation=f.tanh):
        super(UnetEncoderDecoderD, self).__init__()
        self.transfer_data = transfer_data
        self.__activation = activation
        if not transfer_data:
            concat = False
        self.encoder = UnetEncoderD(in_channels=in_channels, depth=depth, blocks=blocks_encoder,
                                    start_filters=start_filters, residual=residual, batch_norm=batch_norm)
        self.decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - 1),
                                    out_channels=out_channels, depth=depth, blocks=blocks_decoder, residual=residual,
                                    batch_norm=batch_norm, transpose=transpose, concat=concat)

    def __call__(self, synthesized):
        return self.forward(synthesized)

    def forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if self.transfer_data:
            reconstructed = self.decoder(image_code, before_pool)
        else:
            reconstructed = self.decoder(image_code)
        return self.__activation(reconstructed)