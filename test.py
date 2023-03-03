#!/usr/bin/python3

# Standard imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Local imports
from timedistributed import TimeDistributed



# Testing with fully connected layers
print("\t\t\t============ Testing with fully connected ============ ")
batch_size, seq_length, cin, cout = 8, 10, 64, 128
x = torch.rand([batch_size, seq_length, cin])
model = nn.Sequential(
    nn.Linear(cin, cout),
    nn.ReLU(inplace=True),
    nn.Linear(cout, cout),
    nn.ReLU(inplace=True),
)
s = TimeDistributed(model, batch_first=True)
output = s(x)
expected_shape = [batch_size, seq_length, cout]
assert (
    list(output.shape) == expected_shape
), f"Incorrect output shape, got {output.shape}, expected {expected_shape}"
print("\t\t\t************ Test passed ************\n ")
# Testing with MaxPooling
print("\t\t\t============ Testing MaxPooling2D ============ ")
batch_size, seq_length, cin, height, width = 8, 10, 1, 256, 256
x = torch.rand([batch_size, seq_length, cin, height, width])
model = nn.MaxPool2d(2)
s = TimeDistributed(model, batch_first=True)
output = s(x)
expected_shape = [
    batch_size,
    seq_length,
    cin,
    height // 2,
    width // 2,
]
assert (
    list(output.shape) == expected_shape
), f"Incorrect output shape, got {output.shape}, expected {expected_shape}"
print("\t\t\t************ Test passed ************\n ")
# Testing with Conv2D
print("\t\t\t============ Testing Conv2D ============ ")
x = torch.rand([seq_length, batch_size, cin, height, width])
cout = 32
model = nn.Sequential(
    nn.Conv2d(cin, cout, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
)
s = TimeDistributed(model)
output = s(x)
expected_shape = [seq_length, batch_size, cout, height // 2, width // 2]
assert (
    list(output.shape) == expected_shape
), f"Incorrect output shape, got {output.shape}, expected {expected_shape}"
print("\t\t\t************ Test passed ************\n ")

# Test with PackedSequences
print("\t\t\t============ Testing PackedSequence ============ ")
x = torch.rand([seq_length, batch_size, cin, height, width])
packed_x = pack_padded_sequence(x, [seq_length for _ in range(batch_size)])
model = nn.Sequential(
    nn.Conv2d(cin, cout, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
)
s = TimeDistributed(model, batch_first=False)
output = s(packed_x)

output, _ = pad_packed_sequence(output, batch_first=False)
expected_shape = [seq_length, batch_size, cout, height // 2, width // 2]
assert (
    list(output.shape) == expected_shape
), f"Incorrect output shape, got {output.shape}, expected {expected_shape}"
print("\t\t\t************ Test passed ************\n ")
