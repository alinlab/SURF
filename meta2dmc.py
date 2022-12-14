# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque, OrderedDict
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv


class ExtendedTimeStep(NamedTuple):
    reward: Any
    true_reward: Any
    discount: Any
    observation: Any
    action: Any
    done: Any
    extra: Any

    def last(self):
        return self.done==True

    def __getitem__(self, attr):
        return getattr(self, attr)

class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                action=action,
                                reward=time_step.reward or 0.0,
                                true_reward=0.0,
                                discount=time_step.discount or 1.0,
                                done=time_step.done or False,
                                extra=time_step.extra or None)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class GymEnvWrapper(dm_env.Environment):
    def __init__(self, env, render_kwargs):
        self._env = env
        self._render_kwargs = render_kwargs
        self._obs_spec = OrderedDict([('pixels', specs.Array(shape=(self._render_kwargs['height'], self._render_kwargs['width'], 3), 
                                        dtype=np.uint8, name='pixels'))])
        self._action_spec = specs.BoundedArray(self._env.action_space.shape,
                                               np.float64,
                                               -1,
                                               1,
                                               None)

    def reset(self):
        _ = self._env.reset()
        obs = self._env.sim.render(
            self._render_kwargs['width'], self._render_kwargs['height'],
            mode=self._render_kwargs['mode'],
            camera_name=self._render_kwargs['camera_name']
        )[::-1, :, ::-1]
        time_step = ExtendedTimeStep(observation=obs,
                                action=None,
                                reward=None,
                                true_reward=None,
                                discount=None,
                                done=None,
                                extra=None)
        return time_step

    def step(self, action):
        action = action.astype(self.action_spec().dtype)
        _, reward, done, extra = self._env.step(action)
        obs = self._env.sim.render(
            self._render_kwargs['width'], self._render_kwargs['height'],
            mode=self._render_kwargs['mode'],
            camera_name=self._render_kwargs['camera_name']
        )[::-1, :, ::-1]
        time_step = ExtendedTimeStep(observation=obs,
                                action=action,
                                reward=reward,
                                true_reward=0.0,
                                discount=1.0,
                                done=done,
                                extra=extra)
        return time_step

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed):
    env_name = name.replace('metaworld_','')
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
    
    env = env_cls()
    
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    
    env = TimeLimit(NormalizedBoxEnv(env), env.max_path_length)

    # add wrappers
    render_kwargs = dict(height=84, width=84, mode='offscreen', camera_name='corner2')
    env = GymEnvWrapper(env, render_kwargs)

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    # env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # stack several frames
    pixels_key = 'pixels'
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
