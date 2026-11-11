"""Export the trained chip backbone through the official SDK without opening hardware."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from ..models.gelneuro_unified_snn import GelNeuroUnifiedSNN


def main():
    from sinabs.backend.dynapcnn import DynapcnnNetwork
    from sinabs.layers import IAFSqueeze, SumPool2d
    import sinabs
    import samna
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=GelNeuroUnifiedSNN(saved["bin_us"]).eval()
    model.load_state_dict(saved["model"])
    quantized,integer=model.quantized_copy()
    # Use exactly the same integer weights/thresholds evaluated by the trainer.
    sequence=[SumPool2d(4)]
    for original,params in zip(model.convolutions,integer):
        conv=nn.Conv2d(original.in_channels,original.out_channels,original.kernel_size,
                       original.stride,original.padding,bias=False)
        conv.weight.data.copy_(params["weights"].float())
        sequence.extend([conv,IAFSqueeze(batch_size=1,
            spike_threshold=float(params["threshold_high"]),
            min_v_mem=float(params["threshold_low"]))])
    network=DynapcnnNetwork(snn=nn.Sequential(*sequence),input_shape=(2,128,128),
                           dvs_input=True,discretize=False)
    config=network.make_config(device="speck2fmodule",monitor_layers=[-1])
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"speck2f-config.json").write_text(config.to_json())
    # Save effective re-normalized model for offline comparison with exported weights.
    with (args.out/"quantized-model.pt").open("xb") as f:
        torch.save(dict(model=quantized.state_dict(),bin_us=saved["bin_us"],
                        input_gain=1.,integer_layers=integer,
                        native_feature_step_us=saved.get("native_feature_step_us",saved["bin_us"]),
                        host_decoder_calibrated=saved.get("host_decoder_calibrated",False)),f)
    info=dict(checkpoint=str(args.checkpoint.resolve()),sinabs=sinabs.__version__,
              samna=getattr(samna,"__version__","unknown"),
              sdk_valid=True,device="speck2fmodule",memory=model.resource_estimate(),
              sdk_layer_to_core=network._layer2core_map,sdk_memory=network.memory_summary(),
              dvs_pool=[config.dvs_layer.pooling.x,config.dvs_layer.pooling.y],
              last_layer_monitor=True,host_operations=["four causal EMA traces","log1p","linear task heads"],
              scope="Offline configuration validation only. No device opened; no hardware replay, throughput or power claim.")
    (args.out/"mapping.json").write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


if __name__=="__main__":
    main()
