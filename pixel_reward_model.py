import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time
# import utils

from agent.drqv2 import Encoder, RandomShiftsAug
from replay_buffer_drqv2 import episode_len
from scipy.stats import norm

device = 'cuda'
# device = 'cpu'

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class RewardPredictor(nn.Module):
    def __init__(
        self, 
        obs_shape, 
        action_shape, 
        feature_dim=50,
        hidden_dim=256, 
        hidden_depth=2, 
        activation='tanh'):
        
        super().__init__()
        self.encoder = Encoder(obs_shape)

        self.reward_model = mlp(
            self.encoder.repr_dim + action_shape[0], hidden_dim, 1, hidden_depth
        )
        if activation == 'tanh':
            self.output_act = nn.Tanh()
        elif activation == 'sig':
            self.output_act = nn.Sigmoid()
        else:
            self.output_act = nn.ReLU()
        
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False, feature=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs)
        if detach_encoder:
            obs = obs.detach()

        obs_action = torch.cat([obs, action], dim=-1)
        if feature:
            return obs_action
        pred_reward = self.output_act(self.reward_model(obs_action))

        return pred_reward
    

def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index
    
def KMeans(x, K=3, Niter=50, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()    # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:,None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average
        
    center_index = D_ij.argmin(dim=0).long().view(-1)
    center = x[center_index]
    
    return  center_index


class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=20, activation='tanh', capacity=2000, 
                 teacher_type=0, teacher_noise=0.0, 
                 teacher_margin=0.0, teacher_thres=0.0, 
                 large_batch=1, label_margin=0.0, stack=1, 
                 img_shift=0,
                 time_shift=0,
                 time_crop=0):
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size # maximum # of episodes for training
        self.activation = activation
        # frame stack
        self.stack = stack
        # augmentation
        self.img_aug = RandomShiftsAug(pad=img_shift)
        self.time_shift = time_shift
        self.time_crop = time_crop

        self.original_size_segment = size_segment
        self.size_segment = size_segment + (stack - 1) + 2 * time_shift
        self.original_stack_index = [list(range(i, i+stack)) for i in range(size_segment)]
        self.stack_index = [list(range(i, i+stack)) for i in range(size_segment + 2 * time_shift)]
        self.stack_index_torch = torch.LongTensor([list(range(i, i+stack)) for i in range(size_segment + 2 * time_shift)]).to(device)

        self.capacity = int(capacity)
        self.buffer_seg1_index = np.empty((self.capacity, 2), dtype=np.uint32)
        self.buffer_seg2_index = np.empty((self.capacity, 2), dtype=np.uint32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        
        self.construct_ensemble()
        # self.inputs = []
        # self.actions = []
        # self.targets = []
        
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 16
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.teacher_type = teacher_type
        self.teacher_noise = teacher_noise
        self.teacher_margin = teacher_margin
        self.teacher_thres = teacher_thres
        self.large_batch = large_batch
        
        file_name = os.getcwd()+'/sampling_log.txt'
        self.f_io = open(file_name, 'a')
        self.round_counter = 0
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin
        self.replay_loader = None
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_margin(self, new_margin):
        self.teacher_margin = new_margin
        
    def construct_ensemble(self):
        obs_shape = list(self.ds)
        obs_shape[0] = obs_shape[0]*self.stack
        for i in range(self.de):
            model = RewardPredictor(
                obs_shape=obs_shape, 
                action_shape=self.da, 
                activation=self.activation).to(device)        
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            params = sum([np.prod(p.size()) for p in model.parameters()])
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
        
    # def add_data(self, obs, act, rew, done):
    #     r_t = rew
    #     r_t = np.array(r_t)
        
    #     flat_target = r_t.reshape(1, 1)
    #     obs = np.expand_dims(obs, axis=0)
    #     act = np.expand_dims(act, axis=0)
    #     init_data = len(self.inputs) == 0
    #     if init_data:
    #         self.inputs.append(obs)
    #         self.actions.append(act)
    #         self.targets.append(flat_target)
    #     elif done:
    #         self.inputs[-1] = np.concatenate([self.inputs[-1], obs])
    #         self.actions[-1] = np.concatenate([self.actions[-1], act])
    #         self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
    #         # FIFO
    #         if len(self.inputs) > self.max_size:
    #             self.inputs = self.inputs[1:]
    #             self.actions = self.actions[1:]
    #             self.targets = self.targets[1:]
    #         self.inputs.append([])
    #         self.actions.append([])
    #         self.targets.append([])
    #         print (self.targets)
    #     else:
    #         if len(self.inputs[-1]) == 0:
    #             self.inputs[-1] = obs
    #             self.targets[-1] = flat_target
    #             self.actions[-1] = act
    #         else:
    #             self.inputs[-1] = np.concatenate([self.inputs[-1], obs])
    #             self.actions[-1] = np.concatenate([self.actions[-1], act])
    #             self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

    def feature_member(self, x, y, member=-1):
        # frame stacking
        if self.stack > 1:
            x = np.take(x, self.original_stack_index, axis=1) ## (B, L, S, C, H, W)
            x = x.reshape(temp_batch_size, self.original_size_segment, self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)

            # y = y[:, self.stack-1:]
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)
        x = x.reshape(-1, *x.shape[2:])
        y = y.reshape(-1, y.shape[-1])
        total_size = x.shape[0]
        batch_size = 10
        
        total_iter = int(total_size/batch_size)            
        if total_size > batch_size*total_iter:
            total_iter += 1
        with torch.no_grad():
            features = []
            for index in range(total_iter):
                last_index = (index+1)*batch_size
                if (index+1)*batch_size > total_size:
                    last_index = total_size
                features.append(self.ensemble[member](
                    x[index*batch_size:last_index], y[index*batch_size:last_index], feature=True))
            features = torch.cat(features, dim=0)
            features = features.reshape(total_size // self.original_size_segment, self.original_size_segment, -1)

        return features
    
    def get_feature(self, x, y):
        features = []
        for member in range(self.de):
            features.append(self.feature_member(x, y, member=member).detach().cpu().numpy())
        features = np.array(features)
        return np.mean(features, axis=0)

    def get_rank_probability(self, x_1, y_1, x_2, y_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, y_1, x_2, y_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, y_1, x_2, y_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, y_1, x_2, y_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, y_1, x_2, y_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        # frame stacking
        if self.stack > 1:
            x_1 = np.take(x_1, self.original_stack_index, axis=1) ## (B, L, S, C, H, W)
            x_1 = x_1.reshape(temp_batch_size, self.original_size_segment, self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)
            x_2 = np.take(x_2, self.original_stack_index, axis=1) ## (B, L, S, C, H, W)
            x_2 = x_2.reshape(temp_batch_size, self.original_size_segment, self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)

            # y_1 = y_1[:, self.stack-1:]
            # y_2 = y_2[:, self.stack-1:]
        x_1 = x_1.reshape(-1, *x_1.shape[2:])
        y_1 = y_1.reshape(-1, y_1.shape[-1])
        x_2 = x_2.reshape(-1, *x_2.shape[2:])
        y_2 = y_2.reshape(-1, y_2.shape[-1])
        total_size = x_1.shape[0]
        batch_size = 10
        
        total_iter = int(total_size/batch_size)            
        if total_size > batch_size*total_iter:
            total_iter += 1
        with torch.no_grad():
            r_hat1, r_hat2 = [], []
            for index in range(total_iter):
                last_index = (index+1)*batch_size
                if (index+1)*batch_size > total_size:
                    last_index = total_size
                r_hat1.append(self.r_hat_member(x_1[index*batch_size:last_index], y_1[index*batch_size:last_index], member=member))
                r_hat2.append(self.r_hat_member(x_2[index*batch_size:last_index], y_2[index*batch_size:last_index], member=member))
            r_hat1 = torch.cat(r_hat1, dim=0)
            r_hat2 = torch.cat(r_hat2, dim=0)
            r_hat1 = r_hat1.reshape(total_size // self.original_size_segment, self.original_size_segment, -1)
            r_hat2 = r_hat2.reshape(total_size // self.original_size_segment, self.original_size_segment, -1)
            r_hat1 = r_hat1.sum(dim=1)
            r_hat2 = r_hat2.sum(dim=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, x_1, y_1, x_2, y_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        # frame stacking
        if self.stack > 1:
            x_1 = np.take(x_1, self.original_stack_index, axis=1) ## (B, L, S, C, H, W)
            x_1 = x_1.reshape(temp_batch_size, self.original_size_segment, self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)
            x_2 = np.take(x_2, self.original_stack_index, axis=1) ## (B, L, S, C, H, W)
            x_2 = x_2.reshape(temp_batch_size, self.original_size_segment, self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)

            # y_1 = y_1[:, self.stack-1:]
            # y_2 = y_2[:, self.stack-1:]
        x_1 = x_1.reshape(-1, *x_1.shape[2:])
        y_1 = y_1.reshape(-1, y_1.shape[-1])
        x_2 = x_2.reshape(-1, *x_2.shape[2:])
        y_2 = y_2.reshape(-1, y_2.shape[-1])
        total_size = x_1.shape[0]
        batch_size = 10
        
        total_iter = int(total_size/batch_size)            
        if total_size > batch_size*total_iter:
            total_iter += 1
        with torch.no_grad():
            r_hat1, r_hat2 = [], []
            for index in range(total_iter):
                last_index = (index+1)*batch_size
                if (index+1)*batch_size > total_size:
                    last_index = total_size
                r_hat1.append(self.r_hat_member(x_1[index*batch_size:last_index], y_1[index*batch_size:last_index], member=member))
                r_hat2.append(self.r_hat_member(x_2[index*batch_size:last_index], y_2[index*batch_size:last_index], member=member))
            r_hat1 = torch.cat(r_hat1, dim=0)
            r_hat2 = torch.cat(r_hat2, dim=0)
            r_hat1 = r_hat1.reshape(total_size // self.original_size_segment, self.original_size_segment, -1)
            r_hat2 = r_hat2.reshape(total_size // self.original_size_segment, self.original_size_segment, -1)
            r_hat1 = r_hat1.sum(dim=1)
            r_hat2 = r_hat2.sum(dim=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, y, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        x_torch = x if type(x) == torch.Tensor else torch.from_numpy(x)
        y_torch = y if type(y) == torch.Tensor else torch.from_numpy(y)
        return self.ensemble[member](
            x_torch.float().to(device), 
            y_torch.float().to(device))

    def r_hat(self, x, y):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, y, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x, y):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, y, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def save_last(self, model_dir):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_last.pt' % (model_dir, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
            
    def load_last(self, model_dir):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_last.pt' % (model_dir, member))
            )
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
            
            idx_1 = self.buffer_seg1_index[epoch*batch_size:last_index]
            idx_2 = self.buffer_seg2_index[epoch*batch_size:last_index]
            s_t_1, a_t_1, _ = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1)) ## (B, L+S-1, C, H, W), (B, L, A)
            s_t_2, a_t_2, _ = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
            if self.time_shift > 0:
                s_t_1 = s_t_1[:, self.time_shift:-self.time_shift]
                a_t_1 = a_t_1[:, self.time_shift:-self.time_shift]
                s_t_2 = s_t_2[:, self.time_shift:-self.time_shift]
                a_t_2 = a_t_2[:, self.time_shift:-self.time_shift]

            # frame stacking
            if self.stack > 1:
                s_t_1 = np.take(s_t_1, self.original_stack_index, axis=1) ## (B, L, S, C, H, W)
                s_t_1 = s_t_1.reshape(temp_batch_size, self.original_size_segment, self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)
                s_t_2 = np.take(s_t_2, self.original_stack_index, axis=1) ## (B, L, S, C, H, W)
                s_t_2 = s_t_2.reshape(temp_batch_size, self.original_size_segment, self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)

                # a_t_1 = a_t_1[:, self.stack-1:]
                # a_t_2 = a_t_2[:, self.stack-1:]
            # get logits
            s_t_1 = s_t_1.reshape(-1, *s_t_1.shape[2:])
            a_t_1 = a_t_1.reshape(-1, a_t_1.shape[-1])
            s_t_2 = s_t_2.reshape(-1, *s_t_2.shape[2:])
            a_t_2 = a_t_2.reshape(-1, a_t_2.shape[-1])

            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            temp_batch_size = labels.size(0)
            total += labels.size(0)

            for member in range(self.de):
                r_hat1 = self.r_hat_member(s_t_1, a_t_1, member=member)
                r_hat2 = self.r_hat_member(s_t_2, a_t_2, member=member)
                r_hat1 = r_hat1.reshape(temp_batch_size, self.original_size_segment, -1)
                r_hat2 = r_hat2.reshape(temp_batch_size, self.original_size_segment, -1)
                
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=20):
        self.replay_loader.dataset.try_fetch_instant()
        max_len = min(len(self.replay_loader.dataset._episode_fns), self.max_size)
        assert max_len > 0
        len_traj = episode_len(self.replay_loader.dataset._episodes[self.replay_loader.dataset._episode_fns[0]])

        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True).reshape(-1,1) + (len(self.replay_loader.dataset._episode_fns) - max_len)
        time_index_1 = np.random.choice(len_traj-self.size_segment + 1, size=mb_size, replace=True).reshape(-1,1) + 1

        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True).reshape(-1,1) + (len(self.replay_loader.dataset._episode_fns) - max_len)
        time_index_2 = np.random.choice(len_traj-self.size_segment + 1, size=mb_size, replace=True).reshape(-1,1) + 1

        return np.concatenate([batch_index_1, time_index_1], axis=1).astype(np.uint32), np.concatenate([batch_index_2, time_index_2], axis=1).astype(np.uint32)
    
    def put_queries(self, idx_1, idx_2, labels):
        total_sample = idx_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1_index[self.buffer_index:self.capacity], idx_1[:maximum_index])
            np.copyto(self.buffer_seg2_index[self.buffer_index:self.capacity], idx_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
        
            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1_index[0:remain], idx_1[maximum_index:])
                np.copyto(self.buffer_seg2_index[0:remain], idx_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1_index[self.buffer_index:next_index], idx_1)
            np.copyto(self.buffer_seg2_index[self.buffer_index:next_index], idx_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
            
    def get_label(self, s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2):
        if self.time_shift > 0:
            sum_r_t_1 = np.sum(r_t_1[:, self.time_shift:-self.time_shift], axis=1)
            sum_r_t_2 = np.sum(r_t_2[:, self.time_shift:-self.time_shift], axis=1)
        else:
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        labels = 1*(sum_r_t_1 < sum_r_t_2)
            
        return s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2, labels
                
    def uniform_sampling(self):
        # get queries
        idx_1, idx_2 =  self.get_queries(
            mb_size=self.mb_size)
        s_t_1, a_t_1, r_t_1 = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1))
        s_t_2, a_t_2, r_t_2 = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
        
        # get labels
        s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2, labels = self.get_label(
            s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(idx_1, idx_2, labels)
        
        return len(labels)
    
    def kcenter_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        # get queries
        idx_1, idx_2 =  self.get_queries(
            mb_size=num_init)
        s_t_1, a_t_1, r_t_1 = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1))
        s_t_2, a_t_2, r_t_2 = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
        
        if self.time_shift > 0:
            sa_t_1 = self.get_feature(s_t_1[:, self.time_shift:-self.time_shift], a_t_1[:, self.time_shift:-self.time_shift])
            sa_t_2 = self.get_feature(s_t_2[:, self.time_shift:-self.time_shift], a_t_2[:, self.time_shift:-self.time_shift])
        else:
            sa_t_1 = self.get_feature(s_t_1, a_t_1)
            sa_t_2 = self.get_feature(s_t_2, a_t_2)

        # get final queries based on kmeans clustering
        total_sa = np.concatenate([sa_t_1.reshape(num_init, -1), 
                                   sa_t_2.reshape(num_init, -1)], axis=1)
        temp_len = total_sa.shape[0]
        feature_dim = total_sa.shape[1]
        X_inputs = total_sa.reshape(-1, feature_dim)
        
        center_index = KMeans(torch.Tensor(X_inputs).to(device), K=self.mb_size)
        center_index = center_index.data.cpu().numpy()
        r_t_1, s_t_1, a_t_1 = r_t_1[center_index], s_t_1[center_index], a_t_1[center_index]
        r_t_2, s_t_2, a_t_2 = r_t_2[center_index], s_t_2[center_index], a_t_2[center_index]

        # get labels
        s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2, labels = self.get_label(
            s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(idx_1[center_index], idx_2[center_index], labels)
        
        return len(labels)
    
    def kcenter_disagree_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        idx_1, idx_2 =  self.get_queries(
            mb_size=num_init)
        s_t_1, a_t_1, r_t_1 = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1))
        s_t_2, a_t_2, r_t_2 = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
        
        # get final queries based on uncertainty
        if self.time_shift > 0:
            _, disagree = self.get_rank_probability(s_t_1[:, self.time_shift:-self.time_shift], a_t_1[:, self.time_shift:-self.time_shift], 
                                                    s_t_2[:, self.time_shift:-self.time_shift], a_t_2[:, self.time_shift:-self.time_shift])
        else:
            _, disagree = self.get_rank_probability(s_t_1, a_t_1, s_t_2, a_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, s_t_1, a_t_1 = r_t_1[top_k_index], s_t_1[top_k_index], a_t_1[top_k_index]
        r_t_2, s_t_2, a_t_2 = r_t_2[top_k_index], s_t_2[top_k_index], a_t_2[top_k_index]
        
        if self.time_shift > 0:
            sa_t_1 = self.get_feature(s_t_1[:, self.time_shift:-self.time_shift], a_t_1[:, self.time_shift:-self.time_shift])
            sa_t_2 = self.get_feature(s_t_2[:, self.time_shift:-self.time_shift], a_t_2[:, self.time_shift:-self.time_shift])
        else:
            sa_t_1 = self.get_feature(s_t_1, a_t_1)
            sa_t_2 = self.get_feature(s_t_2, a_t_2)

        # get final queries based on kmeans clustering
        total_sa = np.concatenate([sa_t_1.reshape(num_init_half, -1), 
                                   sa_t_2.reshape(num_init_half, -1)], axis=1)
        temp_len = total_sa.shape[0]
        feature_dim = total_sa.shape[1]
        X_inputs = total_sa.reshape(-1, feature_dim)
        
        center_index = KMeans(torch.Tensor(X_inputs).to(device), K=self.mb_size)
        center_index = center_index.data.cpu().numpy()
        r_t_1, s_t_1, a_t_1 = r_t_1[center_index], s_t_1[center_index], a_t_1[center_index]
        r_t_2, s_t_2, a_t_2 = r_t_2[center_index], s_t_2[center_index], a_t_2[center_index]

        # get labels
        s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2, labels = self.get_label(
            s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(idx_1[center_index], idx_2[center_index], labels)
        
        return len(labels)
    
    def kcenter_entropy_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        idx_1, idx_2 =  self.get_queries(
            mb_size=num_init)
        s_t_1, a_t_1, r_t_1 = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1))
        s_t_2, a_t_2, r_t_2 = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
        
        # get final queries based on uncertainty
        if self.time_shift > 0:
            entropy, _ = self.get_entropy(s_t_1[:, self.time_shift:-self.time_shift], a_t_1[:, self.time_shift:-self.time_shift], 
                                        s_t_2[:, self.time_shift:-self.time_shift], a_t_2[:, self.time_shift:-self.time_shift])
        else:
            entropy, _ = self.get_entropy(s_t_1, a_t_1, s_t_2, a_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, s_t_1, a_t_1 = r_t_1[top_k_index], s_t_1[top_k_index], a_t_1[top_k_index]
        r_t_2, s_t_2, a_t_2 = r_t_2[top_k_index], s_t_2[top_k_index], a_t_2[top_k_index]
        
        if self.time_shift > 0:
            sa_t_1 = self.get_feature(s_t_1[:, self.time_shift:-self.time_shift], a_t_1[:, self.time_shift:-self.time_shift])
            sa_t_2 = self.get_feature(s_t_2[:, self.time_shift:-self.time_shift], a_t_2[:, self.time_shift:-self.time_shift])
        else:
            sa_t_1 = self.get_feature(s_t_1, a_t_1)
            sa_t_2 = self.get_feature(s_t_2, a_t_2)

        # get final queries based on kmeans clustering
        total_sa = np.concatenate([sa_t_1.reshape(num_init_half, -1), 
                                   sa_t_2.reshape(num_init_half, -1)], axis=1)
        temp_len = total_sa.shape[0]
        feature_dim = total_sa.shape[1]
        X_inputs = total_sa.reshape(-1, feature_dim)
        
        center_index = KMeans(torch.Tensor(X_inputs).to(device), K=self.mb_size)
        center_index = center_index.data.cpu().numpy()
        r_t_1, s_t_1, a_t_1 = r_t_1[center_index], s_t_1[center_index], a_t_1[center_index]
        r_t_2, s_t_2, a_t_2 = r_t_2[center_index], s_t_2[center_index], a_t_2[center_index]

        # get labels
        s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2, labels = self.get_label(
            s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(idx_1[center_index], idx_2[center_index], labels)
        
        return len(labels)
    
    def disagreement_sampling(self):
        
        # get queries
        idx_1, idx_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        s_t_1, a_t_1, r_t_1 = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1))
        s_t_2, a_t_2, r_t_2 = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
        
        # get final queries based on uncertainty
        if self.time_shift > 0:
            _, disagree = self.get_rank_probability(s_t_1[:, self.time_shift:-self.time_shift], a_t_1[:, self.time_shift:-self.time_shift], 
                                                    s_t_2[:, self.time_shift:-self.time_shift], a_t_2[:, self.time_shift:-self.time_shift])
        else:
            _, disagree = self.get_rank_probability(s_t_1, a_t_1, s_t_2, a_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, s_t_1, a_t_1 = r_t_1[top_k_index], s_t_1[top_k_index], a_t_1[top_k_index]
        r_t_2, s_t_2, a_t_2 = r_t_2[top_k_index], s_t_2[top_k_index], a_t_2[top_k_index]
        disagree = disagree[top_k_index]
        
        # logging
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) > self.teacher_margin).reshape(-1)
        for _index in range(len(margin_index)):
            self.f_io.write("{}, {}, {}\n".format(
                self.round_counter, disagree[_index], margin_index[_index]))
        self.round_counter += 1
        
        # get labels
        s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2, labels = self.get_label(
            s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(idx_1[top_k_index], idx_2[top_k_index], labels)
        
        return len(labels)
        
    def entropy_sampling(self):
        
        # get queries
        idx_1, idx_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        s_t_1, a_t_1, r_t_1 = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1))
        s_t_2, a_t_2, r_t_2 = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
        
        # get final queries based on uncertainty
        if self.time_shift > 0:
            entropy, _ = self.get_entropy(s_t_1[:, self.time_shift:-self.time_shift], a_t_1[:, self.time_shift:-self.time_shift], 
                                        s_t_2[:, self.time_shift:-self.time_shift], a_t_2[:, self.time_shift:-self.time_shift])
        else:
            entropy, _ = self.get_entropy(s_t_1, a_t_1, s_t_2, a_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, s_t_1, a_t_1 = r_t_1[top_k_index], s_t_1[top_k_index], a_t_1[top_k_index]
        r_t_2, s_t_2, a_t_2 = r_t_2[top_k_index], s_t_2[top_k_index], a_t_2[top_k_index]
        entropy = entropy[top_k_index]
        
        # logging
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) > self.teacher_margin).reshape(-1)
        for _index in range(len(margin_index)):
            self.f_io.write("{}, {}, {}\n".format(
                self.round_counter, entropy[_index], margin_index[_index]))
        self.round_counter += 1
        
        # get labels
        s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2, labels = self.get_label(
            s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(idx_1[top_k_index], idx_2[top_k_index], labels)
        
        return len(labels)


    def shuffle_dataset(self, max_len):
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        return total_batch_index

    def get_cropping_mask(self, r_hat1, r_hat2):
        B, L, _ = r_hat1.shape
        length = np.random.randint(self.original_size_segment-self.time_crop, self.original_size_segment+self.time_crop+1, size=B)
        start_index_1 = np.random.randint(0, L+1-length)
        start_index_2 = np.random.randint(0, L+1-length)
        mask_1 = np.zeros((B,L,1), dtype=np.float32)
        mask_2 = np.zeros((B,L,1), dtype=np.float32)
        for b in range(B):
            mask_1[b, start_index_1[b]:start_index_1[b]+length[b]]=1
            mask_2[b, start_index_2[b]:start_index_2[b]+length[b]]=1

        return torch.from_numpy(mask_1).to(device), torch.from_numpy(mask_2).to(device)

    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                self.opt.zero_grad()

                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                idx_1 = self.buffer_seg1_index[idxs]
                idx_2 = self.buffer_seg2_index[idxs]
                s_t_1, a_t_1, _ = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1)) ## (B, L+S-1, C, H, W), (B, L, A)
                s_t_2, a_t_2, _ = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device) ## (B)
                temp_batch_size = labels.size(0)
                if member == 0:
                    total += labels.size(0)
                
                s_t_1 = torch.as_tensor(s_t_1, device=device).float()
                s_t_2 = torch.as_tensor(s_t_2, device=device).float()
                # image augmentation
                if self.img_aug.pad > 0:
                    orig_shape = s_t_1.shape
                    s_t_1 = s_t_1.reshape(orig_shape[0], -1, *orig_shape[3:]) ## (B, C*(L+S-1), H, W)
                    s_t_2 = s_t_2.reshape(orig_shape[0], -1, *orig_shape[3:])
                    s_t_1 = self.img_aug(s_t_1)
                    s_t_2 = self.img_aug(s_t_2)
                    s_t_1 = s_t_1.reshape(orig_shape) ## (B, L+S-1, C, H, W)
                    s_t_2 = s_t_2.reshape(orig_shape)

                # frame stacking
                if self.stack > 1:
                    s_t_1 = torch.index_select(s_t_1, dim=1, index=self.stack_index_torch.flatten())
                    s_t_1 = s_t_1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack, self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S, C, H, W)
                    s_t_1 = s_t_1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)
                    s_t_2 = torch.index_select(s_t_2, dim=1, index=self.stack_index_torch.flatten())
                    s_t_2 = s_t_2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack, self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S, C, H, W)
                    s_t_2 = s_t_2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)

                    # a_t_1 = a_t_1[:, self.stack-1:]
                    # a_t_2 = a_t_2[:, self.stack-1:]

                # get logits
                s_t_1 = s_t_1.reshape(-1, *s_t_1.shape[2:]) ## (B*L, S*C, H, W)
                a_t_1 = a_t_1.reshape(-1, a_t_1.shape[-1]) ## (B*L, A)
                s_t_2 = s_t_2.reshape(-1, *s_t_2.shape[2:])
                a_t_2 = a_t_2.reshape(-1, a_t_2.shape[-1])
                r_hat1 = self.r_hat_member(s_t_1, a_t_1, member=member) ## (B*L, 1)
                r_hat2 = self.r_hat_member(s_t_2, a_t_2, member=member)
                r_hat1 = r_hat1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), -1) ## (B, L, 1)
                r_hat2 = r_hat2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), -1)
                
                # shifting & cropping time
                if self.time_shift > 0 or self.time_crop > 0:
                    mask_1, mask_2 = self.get_cropping_mask(r_hat1, r_hat2)
                    r_hat1 = (mask_1*r_hat1).sum(axis=1) ## (B, 1)
                    r_hat2 = (mask_2*r_hat2).sum(axis=1)
                else:
                    r_hat1 = r_hat1.sum(axis=1) ## (B, 1)
                    r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1) ## (B, 2)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                curr_loss.backward()
                self.opt.step()
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        
        return ensemble_losses, list_debug_loss1, list_debug_loss2, ensemble_acc
    
    def train_soft_reward(self):
        raise NotImplementedError
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                self.opt.zero_grad()
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                s_t_1, a_t_1, _ = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1)) ## (B, L+S-1, C, H, W), (B, L, A)
                s_t_2, a_t_2, _ = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                temp_batch_size = labels.size(0)
                if member == 0:
                    total += labels.size(0)
                
                s_t_1 = torch.as_tensor(s_t_1, device=device).float()
                s_t_2 = torch.as_tensor(s_t_2, device=device).float()
                # image augmentation
                if self.img_aug.pad > 0:
                    orig_shape = s_t_1.shape
                    s_t_1 = s_t_1.reshape(orig_shape[0], -1, *orig_shape[3:]) ## (B, C*(L+S-1), H, W)
                    s_t_2 = s_t_2.reshape(orig_shape[0], -1, *orig_shape[3:])
                    s_t_1 = self.img_aug(s_t_1)
                    s_t_2 = self.img_aug(s_t_2)
                    s_t_1 = s_t_1.reshape(orig_shape) ## (B, L+S-1, C, H, W)
                    s_t_2 = s_t_2.reshape(orig_shape)

                # frame stacking
                if self.stack > 1:
                    s_t_1 = torch.index_select(s_t_1, dim=1, index=self.stack_index_torch.flatten())
                    s_t_1 = s_t_1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack, self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S, C, H, W)
                    s_t_1 = s_t_1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)
                    s_t_2 = torch.index_select(s_t_2, dim=1, index=self.stack_index_torch.flatten())
                    s_t_2 = s_t_2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack, self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S, C, H, W)
                    s_t_2 = s_t_2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)

                    # a_t_1 = a_t_1[:, self.stack-1:]
                    # a_t_2 = a_t_2[:, self.stack-1:]

                # get logits
                s_t_1 = s_t_1.reshape(-1, *s_t_1.shape[2:]) ## (B*L, S*C, H, W)
                a_t_1 = a_t_1.reshape(-1, a_t_1.shape[-1]) ## (B*L, A)
                s_t_2 = s_t_2.reshape(-1, *s_t_2.shape[2:])
                a_t_2 = a_t_2.reshape(-1, a_t_2.shape[-1])
                r_hat1 = self.r_hat_member(s_t_1, a_t_1, member=member) ## (B*L, 1)
                r_hat2 = self.r_hat_member(s_t_2, a_t_2, member=member)
                r_hat1 = r_hat1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), -1) ## (B, L, 1)
                r_hat2 = r_hat2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), -1)
                
                # shifting & cropping time
                if self.time_shift > 0 or self.time_crop > 0:
                    mask_1, mask_2 = self.get_cropping_mask(r_hat1, r_hat2)
                    r_hat1 = (mask_1*r_hat1).sum(axis=1) ## (B, 1)
                    r_hat2 = (mask_2*r_hat2).sum(axis=1)
                else:
                    r_hat1 = r_hat1.sum(axis=1) ## (B, 1)
                    r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1) ## (B, 2)

                # compute loss
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                curr_loss.backward()
                self.opt.step()
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct        
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_losses, list_debug_loss1, list_debug_loss2, ensemble_acc