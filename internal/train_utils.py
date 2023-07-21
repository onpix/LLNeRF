# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training step and model creation functions."""

import collections
import functools
from typing import Any, Callable, Dict, MutableMapping, Optional, Text, Tuple
from flax.training import checkpoints

import ipdb
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
from internal import camera_utils
from internal import configs
from internal import datasets
from internal import image
from internal import math
from internal import models
from internal import ref_utils
from internal import stepfun
from internal import utils
import jax
from jax import random, core
import jax.numpy as jnp
import optax


def tree_sum(tree):
  return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)


def tree_norm_sq(tree):
  # calculate all param's norm value, and sum them together.
  # returned is a scalar value.
  return tree_sum(jax.tree_util.tree_map(lambda x: jnp.sum(x ** 2), tree))


def tree_norm(tree):
  # input: a tree; return: a scalar.
  return jnp.sqrt(tree_norm_sq(tree))


def tree_abs_max(tree):
  return jax.tree_util.tree_reduce(
    lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), tree, initializer=0)


def tree_len(tree):
  return tree_sum(
    jax.tree_util.tree_map(lambda z: jnp.prod(jnp.array(z.shape)), tree))


def summarize_tree(tree, fn, ancestry=(), max_depth=3):
  """Flatten 'tree' while 'fn'-ing values and formatting keys like/this."""
  stats = {}
  for k, v in tree.items():
    name = ancestry + (k,)
    stats['/'.join(name)] = fn(v)
    if hasattr(v, 'items') and len(ancestry) < (max_depth - 1):
      stats.update(summarize_tree(v, fn, ancestry=name, max_depth=max_depth))
  return stats


def rgb2l(rgb):
  # rgb shape: [..., 3]; return shape: [..., 1]
  return (rgb.min(axis=-1) + rgb.max(axis=-1)) / 2


def compute_data_loss(batch, renderings, rays, config):
  """Computes data loss terms for RGB, normal, and depth outputs."""
  data_losses = []
  stats = collections.defaultdict(lambda: [])

  # lossmult can be used to apply a weight to each ray in the batch.
  # For example: masking out rays, applying the Bayer mosaic mask, upweighting
  # rays from lower resolution images and so on.
  lossmult = rays.lossmult  # shape: [4096, 1, 1, 3]
  gt = batch.rgb[..., :3]
  lossmult = jnp.broadcast_to(lossmult, gt.shape)
  if config.disable_multiscale_loss:
    lossmult = jnp.ones_like(lossmult)

  for rendering in renderings:
    resid_sq = (rendering['rgb'] - gt) ** 2
    denom = lossmult.sum()
    stats['mses'].append((lossmult * resid_sq).sum() / denom)
    rgb_render_clip = jnp.minimum(1., rendering['rgb'])
    # jax.debug.breakpoint()

    if config.data_loss_type == 'mse':
      # Mean-squared error (L2) loss.
      data_loss = resid_sq

    elif config.data_loss_type == 'l1':
      data_loss = jnp.abs(rgb_render_clip - gt)

    elif config.data_loss_type == 'rawnerf':
      # Clip raw values against 1 to match sensor overexposure behavior.
      resid_sq_clip = (rgb_render_clip - gt) ** 2
      # Scale by gradient of log tonemapping curve.
      scaling_grad = 1. / (1e-3 + jax.lax.stop_gradient(rgb_render_clip))
      # Reweighted L2 loss.
      data_loss = resid_sq_clip * scaling_grad ** 2

    else:
      assert False

    data_losses.append((lossmult * data_loss).sum() / denom)

    if config.compute_disp_metrics:
      # Using mean to compute disparity, but other distance statistics can
      # be used instead.
      disp = 1 / (1 + rendering['distance_mean'])
      stats['disparity_mses'].append(((disp - batch.disps) ** 2).mean())

    if config.compute_normal_metrics:
      if 'normals' in rendering:
        weights = rendering['acc'] * batch.alphas
        normalized_normals_gt = ref_utils.l2_normalize(batch.normals)
        normalized_normals = ref_utils.l2_normalize(rendering['normals'])
        normal_mae = ref_utils.compute_weighted_mae(weights, normalized_normals,
                                                    normalized_normals_gt)
      else:
        # If normals are not computed, set MAE to NaN.
        normal_mae = jnp.nan
      stats['normal_maes'].append(normal_mae)

  data_losses = jnp.array(data_losses)
  loss = (
      config.data_coarse_loss_mult * jnp.sum(data_losses[:-1]) +
      config.data_loss_mult * data_losses[-1])
  stats = {k: jnp.array(stats[k]) for k in stats}
  return loss, stats


def gray_loss(rgb, config, ref=None):
  assert rgb.shape[-1] == 3

  weight1 = weight2 = 1
  assert ref is not None
  weight2 = ref.var(axis=-1, keepdims=True) + config.gray_variance_bias

  diffs = (rgb - jnp.roll(rgb, 1, axis=-1)) ** 2  # shape: [..., 3]
  if config.gray_loss_clip and config.gray_loss_clip > 0:
    diffs = jnp.minimum(config.gray_loss_clip, diffs)
  return jnp.sqrt(diffs.sum(axis=-1, keepdims=True) / 3 / weight1 / weight2).mean()


def ltv_loss(L_e, L, config, beta=1.5, alpha=2, eps=1e-4):
  # # get the gray scale image of L_enhanced and split it.
  # assert rendering['L_enhanced'].shape[0] % 3 == 0 and config.sample_neighbor_num > 0
  # L_chunks = jnp.array_split(rendering['L'].mean(axis=-1), 3)
  # Le_chunks = jnp.array_split(rendering['L_enhanced'].mean(axis=-1), 3)
  # pix_L, right_pix_L, down_pix_L = L_chunks
  # pix_Le, right_pix_Le, down_pix_Le = Le_chunks

  assert L_e.shape[1] == 3 and config.sample_neighbor_num > 0
  L = jnp.log(L + eps)
  pix_L, right_pix_L, down_pix_L = L[:, 0, ...], L[:, 1, ...], L[:, 2, ...]
  pix_Le, right_pix_Le, down_pix_Le = L_e[:, 0, ...], L_e[:, 1, ...], L_e[:, 2, ...]

  dx_L = pix_L - right_pix_L
  dy_L = pix_L - down_pix_L
  dx_Le = pix_Le - right_pix_Le
  dy_Le = pix_Le - down_pix_Le

  ltv_x = (beta * dx_Le ** 2) / (dx_L ** alpha + eps)
  ltv_y = (beta * dy_Le ** 2) / (dy_L ** alpha + eps)

  return ((ltv_x + ltv_y) / 2).mean()


def spatial_loss(L_e, L, config):
  assert L_e.shape[1] == 3 and config.sample_neighbor_num > 0
  pix_L, right_pix_L, down_pix_L = L[:, 0, ...], L[:, 1, ...], L[:, 2, ...]
  pix_Le, right_pix_Le, down_pix_Le = L_e[:, 0, ...], L_e[:, 1, ...], L_e[:, 2, ...]

  dx_L = pix_L - right_pix_L
  dy_L = pix_L - down_pix_L
  dx_Le = pix_Le - right_pix_Le
  dy_Le = pix_Le - down_pix_Le

  x_loss = (jnp.abs(dx_L) - jnp.abs(dx_Le)) ** 2
  y_loss = (jnp.abs(dy_L) - jnp.abs(dy_Le)) ** 2
  return ((x_loss + y_loss) / 2).mean()


def compute_rgb_enhanced_loss(batch, renderings, rays, ray_history, config):
  """Compute data loss for rgb_enhanced"""
  losses = {}
  rendering = renderings[-1]
  last_history = ray_history[-1]
  rgb_e = rendering['rgb_enhanced']
  R_norm = jax.lax.stop_gradient(rendering['R'] / rendering['R'].max())

  # compute exposure loss
  if config.exposure_loss_mult > 0:
    # low_rgbs = jnp.minimum(rendering['rgb_enhanced'], 0.5)
    if not config.exposure_loss_use_mean:
      losses['exp'] = config.exposure_loss_mult * jnp.mean((rgb_e - config.fixed_exposure) ** 2)
    else:
      losses['exp'] = config.exposure_loss_mult * jnp.mean((rgb_e.mean(axis=-1) - config.fixed_exposure) ** 2)

  # gray-world prior:
  if config.gray_loss_mult > 0:
    losses['gray'] = config.gray_loss_mult * gray_loss(rgb_e, config, ref=R_norm)

  L_sg = jax.lax.stop_gradient(rendering['L'])
  if config.ltv_loss_mult > 0:
    if config.smooth_L_ratio:
      L_e = rendering['L_enhanced'] / (L_sg + 1e-4)
    else:
      L_e = rendering['L_enhanced']
    losses['ltv'] = config.ltv_loss_mult * ltv_loss(L_e, L_sg, config)

  if config.coeff_ltv_ref == 'L_sg':
    coeff_ref_gamma = L_sg
    coeff_ref_alpha = L_sg[..., 0:1]
  elif config.coeff_ltv_ref == 'rgb':
    coeff_ref_gamma = jax.lax.stop_gradient(rendering['rgb'])
    coeff_ref_alpha = coeff_ref_gamma.mean(axis=-1, keepdims=True)
  else:
    raise NotImplementedError()

  if config.gamma_ltv_loss_mult > 0:
    losses['Gltv'] = config.gamma_ltv_loss_mult * ltv_loss(rendering['gamma'], coeff_ref_gamma, config)

  if config.alpha_ltv_loss_mult > 0:
    losses['Altv'] = config.alpha_ltv_loss_mult * ltv_loss(rendering['alpha'], coeff_ref_alpha, config)

  if config.spatial_loss_mult > 0:
    losses['spa'] = config.spatial_loss_mult * spatial_loss(rendering['L_enhanced'], L_sg, config)

  # cos similarity loss (1 - cos / 2), value from 0 ~ 1:
  if config.cos_loss_mult > 0:
    rgb_e = rendering['rgb_enhanced']
    losses['cos'] = config.cos_loss_mult * jnp.mean(
      (1 - (rgb_e * R_norm).sum(axis=-1) /
       (jnp.linalg.norm(rgb_e, axis=-1) * jnp.linalg.norm(R_norm, axis=-1))) / 2
    )

  if config.m_contrast_loss_mult > 0:
    illu = rendering['rgb_enhanced'].mean(axis=-1)
    max_illu = illu.max()
    min_illu = illu.min()
    losses['m_contrast'] = config.m_contrast_loss_mult * (max_illu - min_illu) / (max_illu + min_illu)

  if config.Ce_data_loss_mult > 0:
    rgb_Lg22 = jax.lax.stop_gradient(rendering['rgb_Lg22'])
    R_norm = jax.lax.stop_gradient(rendering['R'] / rendering['R'].max())
    losses['Ce_data_loss'] = config.Ce_data_loss_mult * jnp.mean((rendering['rgb_enhanced'] - R_norm) ** 2)

  return losses


def interlevel_loss(ray_history, config):
  """Computes the interlevel loss defined in mip-NeRF 360."""
  # Stop the gradient from the interlevel loss onto the NeRF MLP.
  last_ray_results = ray_history[-1]
  c = jax.lax.stop_gradient(last_ray_results['sdist'])
  w = jax.lax.stop_gradient(last_ray_results['weights'])
  loss_interlevel = 0.
  for ray_results in ray_history[:-1]:
    cp = ray_results['sdist']
    wp = ray_results['weights']
    loss_interlevel += jnp.mean(stepfun.lossfun_outer(c, w, cp, wp))
  return config.interlevel_loss_mult * loss_interlevel


def distortion_loss(ray_history, config):
  """Computes the distortion loss regularizer defined in mip-NeRF 360."""
  last_ray_results = ray_history[-1]
  c = last_ray_results['sdist']
  w = last_ray_results['weights']
  loss = jnp.mean(stepfun.lossfun_distortion(c, w))
  return config.distortion_loss_mult * loss


# def coeff_norm_loss(ray_history):
#   last_ray_results = ray_history[-1]
#   return (last_ray_results['coeff_gamma'] ** 2).mean()


def orientation_loss(rays, model, ray_history, config):
  """Computes the orientation loss regularizer defined in ref-NeRF."""
  total_loss = 0.
  for i, ray_results in enumerate(ray_history):
    w = ray_results['weights']
    n = ray_results[config.orientation_loss_target]
    if n is None:
      raise ValueError('Normals cannot be None if orientation loss is on.')
    # Negate viewdirs to represent normalized vectors from point to camera.
    v = -1. * rays.viewdirs
    n_dot_v = (n * v[..., None, :]).sum(axis=-1)
    loss = jnp.mean((w * jnp.minimum(0.0, n_dot_v) ** 2).sum(axis=-1))
    if i < model.num_levels - 1:
      total_loss += config.orientation_coarse_loss_mult * loss
    else:
      total_loss += config.orientation_loss_mult * loss
  return total_loss


def predicted_normal_loss(model, ray_history, config):
  """Computes the predicted normal supervision loss defined in ref-NeRF."""
  total_loss = 0.
  for i, ray_results in enumerate(ray_history):
    w = ray_results['weights']
    n = ray_results['normals']
    n_pred = ray_results['normals_pred']
    if n is None or n_pred is None:
      raise ValueError(
        'Predicted normals and gradient normals cannot be None if '
        'predicted normal loss is on.')
    loss = jnp.mean((w * (1.0 - jnp.sum(n * n_pred, axis=-1))).sum(axis=-1))
    if i < model.num_levels - 1:
      total_loss += config.predicted_normal_coarse_loss_mult * loss
    else:
      total_loss += config.predicted_normal_loss_mult * loss
  return total_loss


def clip_gradients(grad, config):
  """Clips gradients of each MLP individually based on norm and max value."""
  # Clip the gradients of each MLP individually.
  # grad_clipped = {'params': {}, **{k: v for k, v in grad.items() if k != 'params'}}
  grad_clipped = {'params': {}}
  for k, g in grad['params'].items():
    # Clip by value.
    if config.grad_max_val > 0:
      g = jax.tree_util.tree_map(
        lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val), g)

    # Then clip by norm.
    if config.grad_max_norm > 0:
      mult = jnp.minimum(
        1, config.grad_max_norm / (jnp.finfo(jnp.float32).eps + tree_norm(g)))
      g = jax.tree_util.tree_map(lambda z: mult * z, g)  # pylint:disable=cell-var-from-loop

    grad_clipped['params'][k] = g
  grad = type(grad)(grad_clipped)
  return grad


def create_train_step(model: models.Model,
                      config: configs.Config,
                      dataset: Optional[datasets.Dataset] = None):
  """Creates the pmap'ed Nerf training function.

  Args:
    model: The linen model.
    config: The configuration.
    dataset: Training dataset.

  Returns:
    pmap'ed training function.
  """
  if dataset is None:
    camtype = camera_utils.ProjectionType.PERSPECTIVE
  else:
    camtype = dataset.camtype

  def train_step(
      rng,
      state,
      batch,
      cameras,
      train_frac,
      img_hw
  ):
    """One optimization step.

    Args:
      rng: jnp.ndarray, random number generator.
      state: TrainState, state of the model/optimizer.
      batch: dict, a mini-batch of data for training.
      cameras: module containing camera poses.
      train_frac: float, the fraction of training that is complete.

    Returns:
      A tuple (new_state, stats, rng) with
        new_state: TrainState, new training state.
        stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        rng: jnp.ndarray, updated random number generator.
    """
    rng, key = random.split(rng)

    def loss_fn(variables):
      rays = batch.rays
      if config.cast_rays_in_train_step:
        rays = camera_utils.cast_ray_batch(
          cameras, rays, img_hw, camtype, xnp=jnp, phase='train',
          sample_neighbor_num=config.sample_neighbor_num)

      # Indicates whether we need to compute output normal or depth maps in 2D.
      compute_extras = (
          config.compute_disp_metrics or config.compute_normal_metrics)

      # import ipdb; ipdb.set_trace()
      # jax.debug.breakpoint()
      # pp rays.origins

      # when using BN:
      # (renderings, ray_history), batch_stats = model.apply(
      renderings, ray_history = model.apply(
        variables,
        key if config.randomized else None,
        rays,
        train_frac=train_frac,
        compute_extras=compute_extras,
        zero_glo=False,
        # is_train=True
        is_train=(train_frac != 1)
        # mutable=['batch_stats']
      )

      losses = {}

      if config.data_loss_mult > 0:
        data_loss, stats = compute_data_loss(batch, renderings, rays, config)
        losses['data'] = data_loss
      else:
        stats = {}

      if not config.disable_enhancement_loss:
        rgb_e_losses = compute_rgb_enhanced_loss(batch, renderings, rays, ray_history, config)
        for k, v in rgb_e_losses.items():
          losses[k] = v

        if config.gamma_norm_loss_mult > 0:
          losses['G_norm'] = config.gamma_norm_loss_mult * (ray_history[-1]['coeff_gamma'] ** 2).mean()

        if config.alpha_norm_loss_mult > 0:
          losses['A_norm'] = config.alpha_norm_loss_mult * (ray_history[-1]['coeff_alpha'] ** 2).mean()

      if config.interlevel_loss_mult > 0:
        losses['interlevel'] = interlevel_loss(ray_history, config)

      if config.distortion_loss_mult > 0:
        losses['distortion'] = distortion_loss(ray_history, config)

      if (config.orientation_coarse_loss_mult > 0 or
          config.orientation_loss_mult > 0):
        losses['orientation'] = orientation_loss(rays, model, ray_history,
                                                 config)

      if (config.predicted_normal_coarse_loss_mult > 0 or
          config.predicted_normal_loss_mult > 0):
        losses['predicted_normals'] = predicted_normal_loss(
          model, ray_history, config)

      stats['weight_l2s'] = summarize_tree(variables['params'], tree_norm_sq)

      if config.weight_decay_mults:
        it = config.weight_decay_mults.items
        losses['weight'] = jnp.sum(
          jnp.array([m * stats['weight_l2s'][k] for k, m in it()]))

      stats['loss'] = jnp.sum(jnp.array(list(losses.values())))
      stats['losses'] = losses

      if 'rgb_enhanced' in renderings[-1]:
        stats['Gpsnr'] = image.mse_to_psnr(
          ((1.5 * renderings[-1]['rgb'] ** 0.45 - 1.5 * batch.rgb[..., :3] ** 0.45) ** 2).mean())
        stats['CePsnr'] = image.mse_to_psnr(
          ((renderings[-1]['rgb_enhanced'] - 1.5 * renderings[-1]['rgb'] ** 0.45) ** 2).mean())
      return stats['loss'], stats

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, stats), grad = loss_grad_fn(state.params)

    pmean = lambda x: jax.lax.pmean(x, axis_name='batch')
    # grad['params'] = pmean(grad['params'])
    grad = pmean(grad)
    stats = pmean(stats)

    stats['grad_norms'] = summarize_tree(grad['params'], tree_norm)
    stats['grad_maxes'] = summarize_tree(grad['params'], tree_abs_max)

    grad = clip_gradients(grad, config)

    grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)

    new_state = state.apply_gradients(grads=grad)

    opt_delta = jax.tree_util.tree_map(lambda x, y: x - y, new_state,
                                       state).params['params']
    stats['opt_update_norms'] = summarize_tree(opt_delta, tree_norm)
    stats['opt_update_maxes'] = summarize_tree(opt_delta, tree_abs_max)

    if 'mses' in stats:
      stats['psnrs'] = image.mse_to_psnr(stats['mses'])
      stats['psnr'] = stats['psnrs'][-1]
    return new_state, stats, rng

  train_pstep = jax.pmap(
    train_step,
    axis_name='batch',
    in_axes=(0, 0, 0, None, None, None),
    donate_argnums=(0, 1))

  return train_pstep


def create_optimizer(
    config: configs.Config,
    variables: FrozenVariableDict) -> Tuple[TrainState, Callable[[int], float]]:
  """Creates optax optimizer for model training."""
  adam_kwargs = {
    'b1': config.adam_beta1,
    'b2': config.adam_beta2,
    'eps': config.adam_eps,
  }
  lr_kwargs = {
    'max_steps': config.max_steps,
    'lr_delay_steps': config.lr_delay_steps,
    'lr_delay_mult': config.lr_delay_mult,
  }

  def get_lr_fn(lr_init, lr_final):
    return functools.partial(
      math.learning_rate_decay,
      lr_init=lr_init,
      lr_final=lr_final,
      **lr_kwargs)

  lr_fn_main = get_lr_fn(config.lr_init, config.lr_final)
  tx = optax.adam(learning_rate=lr_fn_main, **adam_kwargs)

  return TrainState.create(apply_fn=None, params=variables, tx=tx), lr_fn_main


def create_render_fn(model: models.Model):
  """Creates pmap'ed function for full image rendering."""

  def render_eval_fn(variables, train_frac, _, rays):
    return jax.lax.all_gather(
      model.apply(
        variables,
        None,  # Deterministic.
        rays,
        train_frac=train_frac,
        compute_extras=True,
        is_train=False,
        # mutable=['batch_stats']
      ),
      axis_name='batch')

  # pmap over only the data input.
  render_eval_pfn = jax.pmap(
    render_eval_fn,
    in_axes=(None, None, None, 0),
    axis_name='batch',
  )
  return render_eval_pfn


def setup_model(
    config: configs.Config,
    rng: jnp.array,
    dataset: Optional[datasets.Dataset] = None,
) -> Tuple[models.Model, TrainState, Callable[
  [FrozenVariableDict, jnp.array, utils.Rays],
  MutableMapping[Text, Any]], Callable[
             [jnp.array, TrainState, utils.Batch, Optional[Tuple[Any, ...]], float],
             Tuple[TrainState, Dict[Text, Any], jnp.array]], Callable[[int], float]]:
  """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

  dummy_rays = utils.dummy_rays(
    include_exposure_idx=config.rawnerf_mode, include_exposure_values=True)
  model, variables = models.construct_model(rng, dummy_rays, config)

  state, lr_fn = create_optimizer(config, variables)
  render_eval_pfn = create_render_fn(model)
  train_pstep = create_train_step(model, config, dataset=dataset)

  return model, state, render_eval_pfn, train_pstep, lr_fn
