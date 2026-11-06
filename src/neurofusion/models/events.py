"""Host baseline event encoder. Raw column order is x,y,device_us,polarity(0/1)."""
import torch
from torch import nn


def voxelize(packets, start_us, end_us, *, sensor_size=(128, 128), grid_size=32, bins=4, device='cpu'):
    if end_us <= start_us or not packets:
        raise ValueError('A closed nonempty batch and positive device-time window are required')
    width, height = sensor_size
    grid = torch.zeros(len(packets), bins*2, grid_size, grid_size, device=device)
    for index, packet in enumerate(packets):
        packet = torch.as_tensor(packet, device=device)
        if packet.ndim != 2 or packet.shape[1] != 4 or not torch.isfinite(packet).all():
            raise ValueError('Events must be finite Nx4: x,y,device_us,polarity')
        if not len(packet):
            continue
        x, y, t, p = packet.unbind(-1)
        if ((x < 0) | (x >= width) | (y < 0) | (y >= height) | (t < start_us) | (t >= end_us)
            | ((p != 0) & (p != 1)) | (x != x.floor()) | (y != y.floor())).any():
            raise ValueError('Events must lie in sensor bounds and half-open [start_us,end_us)')
        if (t[1:] < t[:-1]).any():
            raise ValueError('Packet timestamps must be ordered')
        channel = ((t-start_us)*bins/(end_us-start_us)).long() + p.long()*bins
        flat = channel*grid_size*grid_size + (y*grid_size/height).long()*grid_size + (x*grid_size/width).long()
        grid[index].view(-1).scatter_add_(0, flat, torch.ones(len(packet), device=device))
    return grid.log1p()


class PacketEncoder(nn.Module):
    def __init__(self, output_dim=64, bins=4):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(bins*2,16,3,2,1), nn.SiLU(), nn.Conv2d(16,32,3,2,1),
                    nn.SiLU(), nn.Conv2d(32,64,3,2,1), nn.SiLU(), nn.AdaptiveAvgPool2d((2,2)),
                    nn.Flatten(), nn.Linear(256,output_dim))

    def forward(self, voxels):
        return self.net(voxels)
