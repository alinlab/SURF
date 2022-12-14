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
from tqdm import tqdm
# import utils

from agent.drqv2 import Encoder, RandomShiftsAug
from pixel_reward_model import weight_init, mlp, compute_smallest_dist, KCenterGreedy, KMeans, RewardPredictor, RewardModel
from scipy.stats import norm

device = 'cuda'
# device = 'cpu'


class RewardModelSemi(RewardModel):
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=20, activation='tanh', capacity=2000, 
                 teacher_type=0, teacher_noise=0.0, 
                 teacher_margin=0.0, teacher_thres=0.0, 
                 large_batch=1, label_margin=0.0, stack=1, 
                 inv_label_ratio=10,
                 threshold_u=0.95,
                 lambda_u=1,
                 mu=1,
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

        self.u_capacity = int(capacity*inv_label_ratio)
        self.u_buffer_seg1_index = np.empty((self.u_capacity, 2), dtype=np.uint32)
        self.u_buffer_seg2_index = np.empty((self.u_capacity, 2), dtype=np.uint32)
        self.u_buffer_label = np.empty((self.u_capacity, 1), dtype=np.float32) # for analysis
        self.u_buffer_index = 0
        self.u_buffer_full = False

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

        self.inv_label_ratio = inv_label_ratio
        self.threshold_u = threshold_u
        self.lambda_u = lambda_u
        self.mu = mu
        self.UCELoss = nn.CrossEntropyLoss(reduction='none')

    def put_unlabeled_queries(self, idx_1, idx_2, labels):
        total_sample = idx_1.shape[0]
        next_index = self.u_buffer_index + total_sample
        if next_index >= self.u_capacity:
            self.u_buffer_full = True
            maximum_index = self.u_capacity - self.u_buffer_index
            np.copyto(self.u_buffer_seg1_index[self.u_buffer_index:self.u_capacity], idx_1[:maximum_index])
            np.copyto(self.u_buffer_seg2_index[self.u_buffer_index:self.u_capacity], idx_2[:maximum_index])
            np.copyto(self.u_buffer_label[self.u_buffer_index:self.u_capacity], labels[:maximum_index])
        
            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.u_buffer_seg1_index[0:remain], idx_1[maximum_index:])
                np.copyto(self.u_buffer_seg2_index[0:remain], idx_2[maximum_index:])
                np.copyto(self.u_buffer_label[0:remain], labels[maximum_index:])

            self.u_buffer_index = remain
        else:
            np.copyto(self.u_buffer_seg1_index[self.u_buffer_index:next_index], idx_1)
            np.copyto(self.u_buffer_seg2_index[self.u_buffer_index:next_index], idx_2)
            np.copyto(self.u_buffer_label[self.u_buffer_index:next_index], labels)
            self.u_buffer_index = next_index

    
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
        
        # get unlabeled samples
        u_idx_1, u_idx_2 =  self.get_queries(mb_size=self.mb_size*self.inv_label_ratio)
        u_s_t_1, u_a_t_1, u_r_t_1 = self.replay_loader.dataset.get_segment_batch(u_idx_1, self.size_segment - (self.stack - 1))
        u_s_t_2, u_a_t_2, u_r_t_2 = self.replay_loader.dataset.get_segment_batch(u_idx_2, self.size_segment - (self.stack - 1))
        # for analysis
        *_, u_labels = self.get_label(u_s_t_1, u_s_t_2, u_a_t_1, u_a_t_2, u_r_t_1, u_r_t_2)
        self.put_unlabeled_queries(u_idx_1, u_idx_2, u_labels)
        
        return len(labels)    


    def semi_train_reward(self, num_iters):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        u_ensemble_select = np.array([0 for _ in range(self.de)])
        u_ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        u_max_len = self.u_capacity if self.u_buffer_full else self.u_buffer_index
        u_total_batch_index = self.shuffle_dataset(u_max_len)
        
        total = 0
        u_total = 0
        
        start_index = 0
        u_start_index = 0
        
        for epoch in tqdm(range(num_iters)):
            last_index = start_index + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            u_last_index = u_start_index + self.train_batch_size * self.mu
            if u_last_index > u_max_len:
                u_last_index = u_max_len
                
            for member in range(self.de):
                self.opt.zero_grad()

                # get random batch
                idxs = total_batch_index[member][start_index:last_index]
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
                    assert False ## not implemented for unlabeled batch
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

                # get random unlabeled batch
                u_idxs = u_total_batch_index[member][u_start_index:u_last_index]
                u_idx_1 = self.u_buffer_seg1_index[u_idxs]
                u_idx_2 = self.u_buffer_seg2_index[u_idxs]
                u_s_t_1, u_a_t_1, _ = self.replay_loader.dataset.get_segment_batch(u_idx_1, self.size_segment - (self.stack - 1)) ## (B, L+S-1, C, H, W), (B, L, A)
                u_s_t_2, u_a_t_2, _ = self.replay_loader.dataset.get_segment_batch(u_idx_2, self.size_segment - (self.stack - 1))
                u_temp_batch_size = u_last_index - u_start_index
                
                u_s_t_1 = torch.as_tensor(u_s_t_1, device=device).float()
                u_s_t_2 = torch.as_tensor(u_s_t_2, device=device).float()
                # frame stacking
                if self.stack > 1:
                    u_s_t_1 = torch.index_select(u_s_t_1, dim=1, index=self.stack_index_torch.flatten())
                    u_s_t_1 = u_s_t_1.reshape(u_temp_batch_size, self.size_segment - (self.stack - 1), self.stack, self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S, C, H, W)
                    u_s_t_1 = u_s_t_1.reshape(u_temp_batch_size, self.size_segment - (self.stack - 1), self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)
                    u_s_t_2 = torch.index_select(u_s_t_2, dim=1, index=self.stack_index_torch.flatten())
                    u_s_t_2 = u_s_t_2.reshape(u_temp_batch_size, self.size_segment - (self.stack - 1), self.stack, self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S, C, H, W)
                    u_s_t_2 = u_s_t_2.reshape(u_temp_batch_size, self.size_segment - (self.stack - 1), self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)

                    # u_a_t_1 = u_a_t_1[:, self.stack-1:]
                    # u_a_t_2 = u_a_t_2[:, self.stack-1:]
                
                # get logits
                u_s_t_1 = u_s_t_1.reshape(-1, *u_s_t_1.shape[2:]) ## (B*L, S*C, H, W)
                u_a_t_1 = u_a_t_1.reshape(-1, u_a_t_1.shape[-1]) ## (B*L, A)
                u_s_t_2 = u_s_t_2.reshape(-1, *u_s_t_2.shape[2:])
                u_a_t_2 = u_a_t_2.reshape(-1, u_a_t_2.shape[-1])
                u_r_hat1 = self.r_hat_member(u_s_t_1, u_a_t_1, member=member) ## (B*L, 1)
                u_r_hat2 = self.r_hat_member(u_s_t_2, u_a_t_2, member=member)
                u_r_hat1 = u_r_hat1.reshape(u_temp_batch_size, self.size_segment - (self.stack - 1), -1) ## (B, L, 1)
                u_r_hat2 = u_r_hat2.reshape(u_temp_batch_size, self.size_segment - (self.stack - 1), -1)

                # pseudo-labeling
                if self.time_shift > 0:
                    u_r_hat1_noaug = u_r_hat1[:, self.time_shift:-self.time_shift]
                    u_r_hat2_noaug = u_r_hat2[:, self.time_shift:-self.time_shift]
                else:
                    u_r_hat1_noaug = u_r_hat1
                    u_r_hat2_noaug = u_r_hat2
                with torch.no_grad():
                    u_r_hat1_noaug = u_r_hat1_noaug.sum(axis=1)
                    u_r_hat2_noaug = u_r_hat2_noaug.sum(axis=1)
                    u_r_hat_noaug = torch.cat([u_r_hat1_noaug, u_r_hat2_noaug], axis=-1)

                    pred = torch.softmax(u_r_hat_noaug, dim=1)
                    pred_max = pred.max(1)
                    mask = (pred_max[0] >= self.threshold_u)
                    pseudo_labels = pred_max[1].detach()

                # shifting & cropping time
                u_mask_1, u_mask_2 = self.get_cropping_mask(u_r_hat1, u_r_hat2)
                
                u_r_hat1 = (u_mask_1*u_r_hat1).sum(axis=1)
                u_r_hat2 = (u_mask_2*u_r_hat2).sum(axis=1)
                u_r_hat = torch.cat([u_r_hat1, u_r_hat2], axis=-1)

                curr_loss += torch.mean(self.UCELoss(u_r_hat, pseudo_labels) * mask) * self.lambda_u

                curr_loss.backward()
                self.opt.step()
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

                # for analysis
                u_ensemble_select[member] += mask.sum().item()
                u_labels = self.u_buffer_label[u_idxs]
                u_labels = torch.from_numpy(u_labels.flatten()).long().to(device)
                if member == 0:
                    u_total += u_labels.size(0)
                u_correct = ((pseudo_labels == u_labels)* mask).sum().item()
                u_ensemble_acc[member] += u_correct
            
            start_index += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_index = 0
            u_start_index += self.train_batch_size * self.mu
            if u_last_index == u_max_len:
                u_total_batch_index = self.shuffle_dataset(u_max_len)
                u_start_index = 0
                
        ensemble_acc = ensemble_acc / total
        u_ensemble_acc = u_ensemble_acc / u_ensemble_select
        u_ensemble_select = u_ensemble_select / u_total
        
        return ensemble_acc, u_ensemble_acc, u_ensemble_select
    