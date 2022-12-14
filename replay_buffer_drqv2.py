# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)

    def relabel_with_predictor(self, predictor, stack=1):
        batch_size = 100
        stack_index = [list(range(i, i+stack)) for i in range(batch_size)]
        for eps_fn in self._replay_dir.glob("*.npz"):
            episode = load_episode(eps_fn)
            eps_len = episode_len(episode)

            total_iter = int(eps_len/batch_size)            
            if eps_len > batch_size*total_iter:
                total_iter += 1
            
            for index in range(total_iter):
                last_index = (index+1)*batch_size
                if (index+1)*batch_size > eps_len:
                    last_index = eps_len

                ## add +1 for the first dummy transition                
                if index == 0:
                    obses = episode["observation"][index*batch_size:last_index]
                    obses = np.concatenate([np.expand_dims(episode["observation"][0], axis=0) for _ in range(stack-1)]+[obses], axis=0)
                else:
                    obses = episode["observation"][index*batch_size-(stack-1):last_index] ## (B+S-1, C, H, W)
                # frame stacking
                if stack > 1:
                    _, C, H, W = obses.shape
                    temp_batch_size = last_index - index*batch_size
                    if temp_batch_size == batch_size:
                        temp_stack_index = stack_index
                    else:
                        temp_stack_index = [list(range(i, i+stack)) for i in range(temp_batch_size)]
                    obses = np.take(obses, temp_stack_index, axis=0) ## (B, S, C, H, W)
                    obses = obses.reshape(temp_batch_size, stack*C, H, W) ## (B, S*C, H, W)

                actions = episode["action"][index*batch_size+1:last_index+1]
                pred_reward = predictor.r_hat_batch(obses, actions)
                episode["reward"][index*batch_size+1:last_index+1] = pred_reward

            save_episode(episode, eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        frame_stack,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
    ):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._frame_stack = frame_stack
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = np.concatenate(episode["observation"][max(idx - self._frame_stack, 0) : idx], 0)
        next_obs = np.concatenate(episode["observation"][max(idx + self._nstep - self._frame_stack, 0) : idx + self._nstep], 0)
        if idx < self._frame_stack:
            obs = np.concatenate([*[episode["observation"][0]] * (self._frame_stack - idx), obs], 0)
        if idx + self._nstep < self._frame_stack:
            next_obs = np.concatenate([*[episode["observation"][0]] * (self._frame_stack - idx - self._nstep), next_obs], 0)

        action = episode["action"][idx]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount

        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


    def try_fetch_instant(self):
        tmp = self._samples_since_last_fetch
        self._samples_since_last_fetch = self._fetch_every
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch = tmp

    def get_segment(self, eps_idx, idx, length):
        eps_fn = self._episode_fns[eps_idx]
        episode = self._episodes[eps_fn]
        assert (idx >= 1) and (idx + length - 1 <= episode_len(episode))
        obs = np.array(episode["observation"][max(idx - self._frame_stack, 0) : idx + length - 1])
        if idx < self._frame_stack:
            obs = np.concatenate([np.array([*[episode["observation"][0]] * (self._frame_stack - idx)]), obs], 0)

        action = np.array(episode["action"][idx : idx + length])
        reward = np.array(episode["true_reward"][idx : idx + length]) ## must be ground-truth reward

        return obs, action, reward

    def get_segment_batch(self, indices, length=50):
        obses = []
        actions = []
        rewards = []
        for eps_idx, idx in indices:
            obs, action, reward = self.get_segment(eps_idx, idx, length)
            obses.append(obs)
            actions.append(action)
            rewards.append(reward)

        return np.array(obses), np.array(actions), np.array(rewards)

    # def relabel_with_predictor(self, predictor):
    #     print (self._episode_fns)
    #     batch_size = 100
    #     for eps_fn in self._episode_fns:
    #         episode = self._episodes[eps_fn]
    #         eps_len = episode_len(episode)

    #         total_iter = int(eps_len/batch_size)            
    #         if eps_len > batch_size*total_iter:
    #             total_iter += 1
            
    #         for index in range(total_iter):
    #             last_index = (index+1)*batch_size
    #             if (index+1)*batch_size > eps_len:
    #                 last_index = eps_len

    #             ## add +1 for the first dummy transition
    #             obses = episode["observation"][index*batch_size+1:last_index+1]
    #             actions = episode["action"][index*batch_size+1:last_index+1]
    #             pred_reward = predictor.r_hat_batch(obses, actions)
    #             episode["reward"][index*batch_size+1:last_index+1] = pred_reward
            
    #         self._episodes[eps_fn] = episode

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    replay_dir, max_size, batch_size, num_workers, save_snapshot, frame_stack, nstep, discount
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(
        replay_dir,
        max_size_per_worker,
        num_workers,
        frame_stack,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader