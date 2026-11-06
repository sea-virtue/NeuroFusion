"""Training entry for a recorded causal sequence, plus explicitly synthetic smoke data."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tomllib
import torch
from ..models.candidates import CandidateConfig
from ..models.fusion import FusionConfig
from ..models.policy import NeuroFusionPolicy, Observation
from ..models.split_actions import ACTION_LAYOUT


def load_configuration(path):
    with Path(path).open('rb') as stream:
        config = tomllib.load(stream)
    return FusionConfig(**config['fusion']), CandidateConfig(**config['models'])


def smoke_data(candidate, batch=2):
    """No hardware, real contact labels or task-performance claim."""
    torch.manual_seed(11)
    tactile_channels = 6 if candidate.tactile_encoder == 'sparsh_dino_small' else 3
    packets = [torch.tensor([[30,40,1000,1],[32,42,2000,0],[34,43,4000,1]],dtype=torch.float64) for _ in range(batch)]
    observations = [Observation('robot',0,torch.randn(batch,16)),
        Observation('rgb',0,torch.rand(batch,3,224,224),captured_ns=0),
        Observation('frame',0,torch.rand(batch,tactile_channels,224,224),captured_ns=0),
        Observation('arm_executed',1_000_000,torch.zeros(batch,7)),
        Observation('gripper_executed',2_000_000,torch.zeros(batch,1)),
        Observation('event',5_000_000,packets,captured_ns=5_000_000,start_us=0,end_us=5000),
        Observation('control',10_000_000),Observation('control',20_000_000)]
    shape = (batch,8) if candidate.action_head == 'mlp' else (batch,candidate.horizon,8)
    return observations, [torch.rand(shape)*.02-.01 for _ in range(2)]


def recorded_data(path, device):
    """Explicit tensor dictionary format, loaded without arbitrary pickle execution.

    Input images are already RGB float tensors / prepared six-channel Sparsh
    tensors; event payloads remain raw Nx4 arrays. No device-host clock fitting
    is guessed here. All samples in a batch share the replay availability grid.
    """
    data = torch.load(path,map_location=device,weights_only=True)
    observations = [Observation(**item) for item in data['observations']]
    return observations, data['targets']


def move_observations(observations, device):
    def move(value):
        if isinstance(value,torch.Tensor): return value.to(device)
        if isinstance(value,list): return [move(item) for item in value]
        return value
    return [replace(item,value=move(item.value)) for item in observations]


def sequence_loss(model, observations, targets):
    if not observations or not targets:
        raise ValueError('A sequence and action labels are required')
    state = model.initial_state(targets[0].shape[0], observations[0].available_ns)
    state, controls = model.rollout(observations,state)
    if len(controls) != len(targets):
        raise ValueError('One action label per control message required')
    losses = [model.actions.loss(condition,target) for (_,condition),target in zip(controls,targets)]
    return torch.stack(losses).mean(), state


def save_checkpoint(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        torch.save({'fusion':asdict(model.fusion_config),'candidate':asdict(model.candidate),
                    'model':model.state_dict(),'action_layout':ACTION_LAYOUT},stream)


def restore_checkpoint(path, device='cpu'):
    data = torch.load(path,map_location='cpu',weights_only=True)
    candidate = CandidateConfig(**data['candidate'])
    # The checkpoint includes frozen backbones; no second download/load needed.
    model = NeuroFusionPolicy(FusionConfig(**data['fusion']),replace(candidate,pretrained=False))
    model.load_state_dict(data['model'],strict=True)
    model.candidate = candidate
    return model.to(device)


def run(fusion, candidate, *, steps=3, device='cpu', data=None, validation=None, checkpoint=None):
    if steps <= 0:
        raise ValueError('steps must be positive')
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = NeuroFusionPolicy(fusion,candidate).to(device)
    observations, targets = recorded_data(data,device) if data else smoke_data(candidate)
    observations = move_observations(observations,device)
    targets = [target.to(device) for target in targets]
    model.actions.fit_normalizer(torch.cat([target.reshape(-1,8) for target in targets],0))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    losses = []
    first_gradients = {}
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, state = sequence_loss(model,observations,targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite sequence loss')
        loss.backward()
        if step == 0:
            for name,module in [('rgb_projection',model.rgb.projection),('tactile_projection',model.tactile.projection),
                                ('event_encoder',model.events),('fusion',model.fusion),('action_head',model.actions)]:
                first_gradients[name] = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
            assert all(value > 0 for value in first_gradients.values()), first_gradients
            if candidate.frozen_encoders:
                assert all(p.grad is None for p in model.rgb.backbone.parameters())
                assert all(p.grad is None for p in model.tactile.backbone.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        state,controls = model.rollout(observations,model.initial_state(targets[0].shape[0],observations[0].available_ns))
        proposals = [proposal for snapshot,_ in controls for proposal in model.propose(snapshot,physical=True)]
    report = {'synthetic':data is None,'scope':'Sequence fitting and software inference; no real robot commands or task-quality claim',
              'fusion':asdict(fusion),'models':asdict(candidate),'device':device,'losses':losses,
              'first_gradient_sums':first_gradients,'parameters':sum(p.numel() for p in model.parameters()),
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'action_layout':ACTION_LAYOUT,'proposals':[asdict(item) for item in proposals],
              'robot_commands_sent':0,'auxiliary_prediction_implemented':False}
    if validation:
        validation_observations,validation_targets = recorded_data(validation,device)
        with torch.no_grad():
            error,_ = sequence_loss(model,validation_observations,validation_targets)
        report['validation_loss'] = error.item()
    if checkpoint:
        save_checkpoint(checkpoint,model)
        loaded = restore_checkpoint(checkpoint,device).eval()
        with torch.no_grad():
            other,_ = loaded.rollout(observations,loaded.initial_state(targets[0].shape[0],observations[0].available_ns))
        # Compare state deterministically; diffusion sampling has its own RNG.
        assert torch.allclose(other.memory,state.memory,atol=1e-6)
        report['checkpoint_roundtrip'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/pipeline.toml')
    parser.add_argument('--fusion',choices=['plain','gated','attention'])
    parser.add_argument('--head',choices=['mlp','act','diffusion_unet','diffusion_transformer'])
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--data',type=Path,help='Recorded sequence .pt; otherwise explicitly synthetic')
    parser.add_argument('--validation-data',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    args = parser.parse_args()
    fusion,candidate = load_configuration(args.config)
    if args.fusion: fusion = replace(fusion,kind=args.fusion)
    if args.head: candidate = replace(candidate,action_head=args.head)
    report = run(fusion,candidate,steps=args.steps,device=args.device,data=args.data,
                 validation=args.validation_data,checkpoint=args.checkpoint)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
