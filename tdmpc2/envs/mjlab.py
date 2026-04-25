"""Single-env adapter exposing mjlab's vectorized whole-body-tracking env to TDMPC2.

TDMPC2's native trainer expects an unbatched Gym-style env (4-tuple step, single
obs tensor). mjlab is always vectorized, so we instantiate it with num_envs=1 and
squeeze the batch dimension at every boundary.
"""

from __future__ import annotations

import math
import os

import gymnasium as gym
import numpy as np
import torch

from envs.wrappers.timeout import Timeout


_TASK_PREFIX = "mjlab-"


class MjlabSingleEnvWrapper(gym.Env):
	"""Wraps mjlab's ManagerBasedRlEnv (num_envs=1) for TDMPC2's single-env API.

	Exposes the actor observation view only — matches what the deployed policy
	(and PPO/SAC actors) see, including noise corruption.
	"""

	metadata = {"render_modes": []}

	def __init__(self, cfg):
		import mjlab.tasks  # noqa: F401  — registers built-in mjlab tasks
		import src.tasks    # noqa: F401  — registers Unitree-G1-Tracking
		from mjlab.envs import ManagerBasedRlEnv
		from mjlab.tasks.registry import load_env_cfg
		from mjlab.tasks.tracking.mdp import MotionCommandCfg
		from rl.sac_env_wrapper import EmpiricalNormalization

		task_id = cfg.task[len(_TASK_PREFIX):]
		# warp/mjlab require an explicit index ("cuda:0"), not the bare "cuda"
		# alias that torch accepts.
		raw_device = str(cfg.get("device", "cuda:0"))
		if raw_device == "cuda":
			raw_device = "cuda:0"
		device = torch.device(raw_device)

		env_cfg = load_env_cfg(task_id)

		motion_cmd = env_cfg.commands["motion"]
		assert isinstance(motion_cmd, MotionCommandCfg), \
			f"Expected MotionCommandCfg, got {type(motion_cmd)}"
		motion_path = cfg.get("motion_path", None)
		assert motion_path is not None, \
			"cfg.motion_path is required for mjlab tasks (path to retargeted .npz)"
		motion_cmd.motion_file = motion_path
		motion_cmd.sampling_mode = "uniform"

		env_cfg.scene.num_envs = 1
		env_cfg.auto_reset = False
		env_cfg.seed = int(cfg.get("seed", 0))

		self._env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
		self._device = device

		self._actor_obs_dim = self._env.single_observation_space.spaces["actor"].shape[0]
		self._act_dim = self._env.single_action_space.shape[0]

		self._normalize_obs = bool(cfg.get("normalize_obs", True))
		self._rms = (
			EmpiricalNormalization(self._actor_obs_dim, device)
			if self._normalize_obs else None
		)

		# Effective control dt = sim.timestep * decimation. Cap at 1e4 steps so
		# the buffer's per-episode storage stays bounded even if env_cfg sets a
		# huge episode_length_s (some configs use 1e9 for "no time limit").
		dt = float(env_cfg.sim.mujoco.timestep) * int(env_cfg.decimation)
		raw_steps = int(math.ceil(float(env_cfg.episode_length_s) / dt))
		max_steps_cap = int(cfg.get("mjlab_max_episode_steps", 500))
		self._max_episode_steps = min(raw_steps, max_steps_cap)

		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf,
			shape=(self._actor_obs_dim,), dtype=np.float32,
		)
		self.action_space = gym.spaces.Box(
			low=-1.0, high=1.0,
			shape=(self._act_dim,), dtype=np.float32,
		)

		# RMS sidecar (mirrors SAC's `.rms.pt` convention). Saved alongside
		# checkpoints under cfg.work_dir.
		work_dir = cfg.get("work_dir", None)
		self._rms_save_path = (
			os.path.join(str(work_dir), "obs_rms.pt") if work_dir else None
		)

	@property
	def max_episode_steps(self):
		return self._max_episode_steps

	def _process_actor_obs(self, obs_dict, training: bool) -> np.ndarray:
		actor_obs = obs_dict["actor"]  # [1, D] on device
		if self._rms is not None:
			if training:
				self._rms.update(actor_obs)
			actor_obs = self._rms(actor_obs)
		return actor_obs.squeeze(0).detach().cpu().numpy().astype(np.float32)

	def reset(self, **kwargs):
		obs_dict, _ = self._env.reset()
		# Do not update RMS on reset-only obs (biased toward initial poses).
		return self._process_actor_obs(obs_dict, training=False)

	def step(self, action):
		# `action` arrives as numpy float32 of shape (act_dim,) from TensorWrapper.
		action_t = torch.as_tensor(action, dtype=torch.float32, device=self._device)
		action_t = action_t.clamp(-1.0, 1.0).unsqueeze(0)  # [1, act_dim]

		obs_dict, rew, terminated, truncated, _ = self._env.step(action_t)
		obs_np = self._process_actor_obs(obs_dict, training=True)

		term_b = bool(terminated.item())
		trunc_b = bool(truncated.item())
		done = term_b or trunc_b
		reward = float(rew.item())

		info = {
			"success": 0.0,  # WBT has no binary success metric
			"terminated": term_b,
		}
		return obs_np, reward, done, info

	def render(self, *args, **kwargs):
		return None

	def close(self):
		self._env.close()

	def save_obs_rms(self, path: str | None = None):
		"""Persist the obs normalizer state. Call from training callbacks."""
		if self._rms is None:
			return
		path = path or self._rms_save_path
		if not path:
			return
		os.makedirs(os.path.dirname(path), exist_ok=True)
		torch.save(self._rms.state_dict(), path)

	def load_obs_rms(self, path: str):
		if self._rms is None:
			return
		self._rms.load_state_dict(torch.load(path, map_location=self._device))


def make_env(cfg):
	"""TDMPC2 env factory. Routes only `mjlab-*` tasks; otherwise raises ValueError
	so the maker loop in envs/__init__.py falls through to the next backend."""
	if not isinstance(cfg.task, str) or not cfg.task.startswith(_TASK_PREFIX):
		raise ValueError(f"Not an mjlab task: {cfg.task}")

	env = MjlabSingleEnvWrapper(cfg)
	env = Timeout(env, max_episode_steps=env.max_episode_steps)
	return env
