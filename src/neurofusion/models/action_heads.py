"""Upstream action networks conditioned on the project's shared state.

All action targets here are normalized. No task policy weights are supplied.
ACT uses original CVAE/Transformer definitions with dimensional input adapters.
"""
import copy
from typing import Optional, List
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .upstream import act_definitions, import_upstream


def act_namespace():
    return dict(torch=torch, nn=nn, np=np, Variable=torch.autograd.Variable)


class MLPActionHead(nn.Module):
    def __init__(self, condition_dim, action_dim=5, hidden_dim=256, hidden_depth=2):
        super().__init__()
        namespace = act_definitions('detr/models/detr_vae.py', act_namespace(), {'mlp'})
        self.net = namespace['mlp'](condition_dim, hidden_dim, action_dim, hidden_depth)
        self.action_dim = action_dim

    def forward(self, condition):
        return self.net(condition)

    def loss(self, condition, actions):
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError('MLP actions must be normalized [B,action_dim]')
        return F.smooth_l1_loss(self(condition), actions)


class DiffusionActionHead(nn.Module):
    def __init__(self, kind, condition_dim, action_dim=5, horizon=16, inference_steps=10,
                 down_dims=(64, 128, 256), width=256, layers=4):
        super().__init__()
        if horizon < 4 or horizon % 4:
            raise ValueError('Action horizon must be a positive multiple of four')
        if not 1 <= inference_steps <= 100:
            raise ValueError('inference_steps must be between 1 and 100')
        self.kind, self.horizon, self.action_dim = kind, horizon, action_dim
        self.inference_steps = inference_steps
        if kind == 'diffusion_unet':
            upstream = import_upstream('diffusion_policy', 'diffusion_policy.model.diffusion.conditional_unet1d')
            self.net = upstream.ConditionalUnet1D(input_dim=action_dim, global_cond_dim=condition_dim,
                                                diffusion_step_embed_dim=128, down_dims=list(down_dims),
                                                cond_predict_scale=True)
        elif kind == 'diffusion_transformer':
            upstream = import_upstream('diffusion_policy', 'diffusion_policy.model.diffusion.transformer_for_diffusion')
            self.net = upstream.TransformerForDiffusion(input_dim=action_dim, output_dim=action_dim,
                        horizon=horizon, n_obs_steps=1, cond_dim=condition_dim, n_layer=layers,
                        n_head=4, n_emb=width, p_drop_emb=0., p_drop_attn=0.)
        else:
            raise ValueError(kind)
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        self.scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True, prediction_type='epsilon')

    def forward(self, noisy_actions, timesteps, condition):
        if self.kind == 'diffusion_unet':
            return self.net(noisy_actions, timesteps, global_cond=condition)
        return self.net(noisy_actions, timesteps, cond=condition[:, None])

    def loss(self, condition, actions, generator=None):
        if actions.shape != (condition.shape[0], self.horizon, self.action_dim):
            raise ValueError('Expected normalized action chunks [B,horizon,action_dim]')
        noise = torch.randn(actions.shape, dtype=actions.dtype, device=actions.device, generator=generator)
        steps = torch.randint(100, (actions.shape[0],), device=actions.device, generator=generator)
        noisy = self.scheduler.add_noise(actions, noise, steps)
        return F.mse_loss(self(noisy, steps, condition), noise)

    @torch.no_grad()
    def sample(self, condition, generator=None):
        if self.training:
            raise RuntimeError('Call eval() before sampling actions')
        actions = torch.randn((condition.shape[0], self.horizon, self.action_dim),
                              device=condition.device, dtype=condition.dtype, generator=generator)
        self.scheduler.set_timesteps(self.inference_steps, device=condition.device)
        for step in self.scheduler.timesteps:
            prediction = self(actions, step, condition)
            actions = self.scheduler.step(prediction, step, actions, generator=generator).prev_sample
        return actions


class ConditionBackbone(nn.Module):
    """Present a shared-state token at ACT's original image-feature boundary."""
    def __init__(self, condition_dim, width):
        super().__init__()
        self.num_channels = width
        self.projection = nn.Conv2d(condition_dim, width, 1)

    def forward(self, condition):
        feature = self.projection(condition)
        # Upstream ACT repeats positions across B itself, so this must have B=1.
        return [feature], [torch.zeros_like(feature[:1])]


class ACTActionHead(nn.Module):
    def __init__(self, condition_dim, robot_dim=16, action_dim=5, horizon=16,
                 width=256, layers=2, kl_weight=10.):
        super().__init__()
        namespace = dict(copy=copy, Optional=Optional, List=List, torch=torch, nn=nn, Tensor=Tensor, F=F)
        transform = act_definitions('detr/models/transformer.py', namespace)
        model = act_definitions('detr/models/detr_vae.py', act_namespace(),
                               {'DETRVAE', 'reparametrize', 'get_sinusoid_encoding_table'})
        transformer = transform['Transformer'](d_model=width, nhead=4, num_encoder_layers=layers,
                        num_decoder_layers=layers, dim_feedforward=width*4, dropout=0.1)
        layer = transform['TransformerEncoderLayer'](width, 4, width*4, 0.1)
        encoder = transform['TransformerEncoder'](layer, layers)
        self.net = model['DETRVAE']([ConditionBackbone(condition_dim, width)], transformer, encoder,
                                  state_dim=action_dim, num_queries=horizon, camera_names=['shared_memory'])
        # Four upstream 14D assumptions are boundary layers, not network redesign.
        self.net.input_proj_robot_state = nn.Linear(robot_dim, width)
        self.net.encoder_joint_proj = nn.Linear(robot_dim, width)
        self.net.encoder_action_proj = nn.Linear(action_dim, width)
        self.robot_dim, self.action_dim, self.horizon = robot_dim, action_dim, horizon
        self.kl_weight = kl_weight

    def forward(self, condition, actions=None):
        padding = None if actions is None else torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
        return self.net(condition[:, -self.robot_dim:], condition[:, None, :, None, None], None, actions, padding)

    def loss(self, condition, actions):
        if actions.shape != (condition.shape[0], self.horizon, self.action_dim):
            raise ValueError('ACT needs full normalized chunks [B,horizon,action_dim]; padded batches not yet supported')
        prediction, _, (mu, logvar) = self(condition, actions)
        kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()
        return F.l1_loss(prediction, actions) + self.kl_weight * kl

    @torch.no_grad()
    def sample(self, condition, generator=None):
        if self.training:
            raise RuntimeError('Call eval() before sampling actions')
        return self(condition)[0]
