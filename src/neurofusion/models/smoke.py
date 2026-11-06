"""Local structural/gradient checks, not a task-quality or real-time benchmark."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import torch
from .candidates import CandidateConfig, build_action_head
from .encoders import ImageBackbone
from .upstream import project_root


def size_report(model):
    total = sum(p.numel() for p in model.parameters())
    return {'parameters': total, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'fp32_parameter_bytes': total * 4, 'fp16_parameter_bytes': total * 2}


def run(candidate=CandidateConfig(), all_candidates=False, device='cpu', batch_size=2):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install a CUDA PyTorch build on the training machine')
    report = {'synthetic': True, 'scope': 'Pretrained load, finite forward and gradient checks only; no task training or robot commands',
              'device': device, 'batch_size': batch_size, 'torch': torch.__version__,
              'selected': asdict(candidate), 'encoders': {}, 'heads': {}}
    names = ['resnet18', 'dinov2_vits14', 'sparsh_dino_small'] if all_candidates else sorted({candidate.rgb_encoder, candidate.tactile_encoder})
    for name in names:
        model = ImageBackbone(name, output_dim=128, pretrained=candidate.pretrained, frozen=candidate.frozen_encoders).to(device)
        images = torch.rand(batch_size, 6 if name == 'sparsh_dino_small' else 3, 224, 224, device=device)
        result = model(images)
        assert result.shape == (batch_size, 128) and torch.isfinite(result).all()
        result.square().mean().backward()
        assert model.projection.weight.grad is not None
        if candidate.frozen_encoders:
            assert all(p.grad is None for p in model.backbone.parameters())
        report['encoders'][name] = {**size_report(model), 'output_shape': list(result.shape),
                                   'pretrained_loaded': model.pretrained_loaded, 'backbone_frozen': candidate.frozen_encoders}
        del model, result
    names = ['mlp', 'act', 'diffusion_unet', 'diffusion_transformer'] if all_candidates else [candidate.action_head]
    for name in names:
        model = build_action_head(replace(candidate, action_head=name)).to(device)
        condition = torch.randn(batch_size, 400, requires_grad=True, device=device)  # Config(): h256+c128+q16.
        target = torch.rand((batch_size, 5) if name == 'mlp' else (batch_size, candidate.horizon, 5), device=device) * 2 - 1
        loss = model.loss(condition, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert condition.grad is not None and condition.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        model.eval()
        with torch.no_grad():
            prediction = model.sample(condition.detach()) if hasattr(model, 'sample') else model(condition.detach())
        assert torch.isfinite(prediction).all()
        report['heads'][name] = {**size_report(model), 'output_shape': list(prediction.shape),
                                'synthetic_loss': loss.item(), 'condition_gradient_nonzero': True,
                                'pretrained_task_policy': False}
        del model, condition, loss, prediction
    weights = project_root() / 'third_party/weights.lock.json'
    report['downloaded_encoder_weights'] = json.loads(weights.read_text())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/candidates.toml'))
    parser.add_argument('--all', action='store_true', help='Check each candidate once, not task performance')
    parser.add_argument('--device', default='cpu', help='cpu or cuda; CUDA requires a compatible PyTorch build')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run(CandidateConfig.load(args.config), args.all, args.device, args.batch_size)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
    for group in ['encoders', 'heads']:
        for name, item in report[group].items():
            print(f"{name:24} {item['parameters']/1e6:8.3f}M total  {item['trainable_parameters']/1e6:8.3f}M trainable")
    print('Synthetic forward/backward checks passed. No trained policy or robot commands.')


if __name__ == '__main__':
    main()
