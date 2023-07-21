import concurrent.futures
import functools
import glob
import os
import time

import ipdb
from absl import app
import gin
from internal import configs
from internal import utils
import jax
from jax import random
from render import create_videos


def main(unused_argv):
  config = configs.load_config(save_config=False)
  # dataset = datasets.load_dataset('test', config.data_dir, config)
  key = random.PRNGKey(20200823)
  # _, state, render_eval_pfn, _, _ = train_utils.setup_model(config, key)
  # state = checkpoints.restore_checkpoint(config.checkpoint_dir, state, step=config.render_ckpt_step)
  # step = int(state.step)
  step = config.render_ckpt_step
  assert step is not None
  print(f'Read checkpoint at step {step}.')

  out_name = 'path_renders' if config.render_path else 'test_preds'
  out_name = f'{out_name}_step_{step}'
  base_dir = config.render_dir
  if base_dir is None:
    base_dir = os.path.join(config.checkpoint_dir, 'render')
  out_dir = os.path.join(base_dir, out_name)
  if not utils.isdir(out_dir):
    utils.makedirs(out_dir)

  path_fn = lambda x: os.path.join(out_dir, x)

  # Ensure sufficient zero-padding of image indices in output filenames.
  zpad = max(3, len(str(config.render_path_frames - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  if config.render_save_async:
    async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    async_futures = []

    def save_fn(fn, *args, **kwargs):
      async_futures.append(async_executor.submit(fn, *args, **kwargs))
  else:
    def save_fn(fn, *args, **kwargs):
      fn(*args, **kwargs)

  # Save videos
  num_files = len(glob.glob(path_fn('acc_*.tiff')))
  # if jax.host_id() == 0 and num_files == config.render_path_frames:
  #   print(f'All files found, creating videos (job {config.render_job_id}).')
  create_videos(config, base_dir, out_dir, out_name, config.render_path_frames)


if __name__ == '__main__':
  with gin.config_scope('eval'):  # Use the same scope as eval.py
    app.run(main)
