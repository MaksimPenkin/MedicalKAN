# """
# @author   Maksim Penkin
# """

from torch import nn
from .encoders import ResidualEncoder
from .decoders import ResidualDecoder
from ..layers import ResBlock, conv3x3


class UNet(nn.Module):

    def __init__(self, in_ch, out_ch=None, filters=None,
                 encoder="ResidualEncoder", encoder_kwargs=None,
                 bottleneck="ResBlock", bottleneck_kwargs=None,
                 decoder="ResidualDecoder", decoder_kwargs=None, **kwargs):
        super(UNet, self).__init__()

        if out_ch is None:
            out_ch = in_ch

        if filters is not None:
            enc_filters = filters
            dec_filters = filters[::-1]
        else:
            enc_filters = encoder_kwargs.pop("filters")
            dec_filters = decoder_kwargs.pop("filters")

        # Embedding.
        self.emb = conv3x3(in_ch, enc_filters[0])

        # Encoder.
        encoder_kwargs = encoder_kwargs or {}
        encoder_kwargs.update(kwargs)
        if encoder == "ResidualEncoder":
            self.encoder = ResidualEncoder(enc_filters, **encoder_kwargs)
        else:
            raise ValueError(f"Unrecognized encoder found: {encoder}.")

        # Bottleneck.
        bottleneck_kwargs = bottleneck_kwargs or {}
        bottleneck_kwargs.update(kwargs)
        if bottleneck == "ResBlock":
            self.bottleneck = ResBlock(enc_filters[-1], **bottleneck_kwargs)
        else:
            raise ValueError(f"Unrecognized bottleneck found: {bottleneck}.")

        # Decoder.
        decoder_kwargs = decoder_kwargs or {}
        decoder_kwargs.update(kwargs)
        if decoder == "ResidualDecoder":
            self.decoder = ResidualDecoder(dec_filters, **decoder_kwargs)
        else:
            raise ValueError(f"Unrecognized decoder found: {decoder}.")

        # Embedding.
        self.restore = conv3x3(dec_filters[-1], out_ch)

    def forward(self, x):
        x = self.emb(x)
        x, feats = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, feats)
        x = self.restore(x)
        return x
