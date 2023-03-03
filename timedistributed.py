import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TimeDistributed(nn.Module):
    """
    TimeDistributed for Pytorch which allows to apply a layer to every temporal slice of an input
    Args:
        Module: a Module instance
    """

    def __init__(self, module, batch_first=False):
        if not isinstance(module, nn.Module):
            raise ValueError(
                "Please initialize `TimeDistributed` with a "
                f"`torch.nn.Module` instance. Received: {module.type}"
            )
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # Input shape is (default)
        #   - Seq_length, BS, *
        # or if batch_first
        #   - BS, Seq_length, *
        # It can also be provided as PackedSequence
        orig_x = x
        if isinstance(x, PackedSequence):
            x, lens_x = pad_packed_sequence(x, batch_first=self.batch_first)

        if self.batch_first:
            # BS, Seq_length, * -> Seq_length, BS, *
            x = x.transpose(0, 1)
        print(x.shape)
        output = torch.stack([self.module(xt) for xt in x], dim=0)
        if self.batch_first:
            # Seq_length, BS, * -> BS, Seq_length, *
            output = output.transpose(0, 1)

        if isinstance(orig_x, PackedSequence):
            output = pack_padded_sequence(output, lens_x, batch_first=self.batch_first)
        return output

