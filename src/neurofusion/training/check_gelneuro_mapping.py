"""Check GelNeuro layer sizes against the Speck2f module manual's core budgets."""
from itertools import permutations
import json

from ..models.gelneuro_snn import GelNeuroSNN


def round_address_count(size):
    return 1 << (size - 1).bit_length()


def resource_estimate(model):
    width = 120
    layers = []
    for conv in model.convolutions:
        kernel = conv.kernel_size[0]
        width = (width + 2 * conv.padding[0] - kernel) // conv.stride[0] + 1
        layers.append(dict(
            shape=[conv.out_channels, width, width],
            kernel_entries=conv.in_channels * round_address_count(conv.out_channels)
            * round_address_count(kernel * kernel),
            neuron_entries=conv.out_channels * round_address_count(width) ** 2,
        ))
    # Optical-module guide §8.4. Kernel words are 8 bit; state words are 16 bit.
    budgets = [(16, 64)] * 3 + [(32, 32)] * 2 + [(64, 16)] * 2 + [(16, 16)] * 2
    assignment = next((cores for cores in permutations(range(len(budgets)), len(layers))
                       if all(layer["kernel_entries"] <= budgets[core][0] * 1024
                              and layer["neuron_entries"] <= budgets[core][1] * 1024
                              for core, layer in zip(cores, layers))), None)
    return dict(layers=layers, candidate_cores=assignment,
                scope="Static memory sizing only; SDK mapping and real-chip readout require validation.")


def main():
    result = resource_estimate(GelNeuroSNN())
    print(json.dumps(result, indent=2))
    if result["candidate_cores"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
