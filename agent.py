from collections import deque
from network import QNetwork
import numpy as np
import copy
import random
import torch

class Agent():
    def __init__(self, mem_len, possible_actions, initial_eps, eps_min, gamma, lr):
        self.mem_len = mem_len
        self.memory = deque(maxlen=mem_len)
        self.possible_actions = possible_actions
        self.eps = initial_eps
        self.eps_min = eps_min
        self.eps_max = initial_eps
        self.gamma = gamma
        self.lr = lr
        self.model = QNetwork(4, len(possible_actions), lr)
        self.model_target = copy.deepcopy(self.model)
        self.learns = 0
        self.timesteps = 0

    def get_action(self, state):
        "Choose epsilon-greedy action"
        self.timesteps += 1
        if np.random.rand() < self.eps:
            return random.sample(self.possible_actions, 1)[0]
        
        "Otherwise, choose the action with the highest Q-value"
        with torch.no_grad():
            pred = self.model(torch.from_numpy(np.array(state)).unsqueeze(0).cuda())
        Q, A = torch.max(pred, axis=1)
        # print(pred)
        # print(f"Taken action of index {int(A)} with approx. Q value of {float(Q)}")
        return self.possible_actions[A]
    
    def add_experience(self, current_frame, action, reward, next_frame, done):
        self.memory.append([current_frame, action, reward, next_frame, done])

    def sample_experience(self, sample_size):
        if len(self.memory) < sample_size:
            sample_size = len(self.memory)

        sample = random.sample(self.memory, sample_size)
        # to tensor
        s = torch.tensor([np.array(exp[0]) for exp in sample]).float().cuda()
        a = torch.tensor([exp[1] for exp in sample]).float().cuda()
        rn = torch.tensor([exp[2] for exp in sample]).float().cuda()
        sn = torch.tensor([np.array(exp[3]) for exp in sample]).float().cuda()   
        done = torch.tensor([exp[4] for exp in sample]).float().cuda()

        return s, a, rn, sn, done

    def train(self, batch_size):
        s, a, rn, sn, done = self.sample_experience(batch_size)

        # predicted expected return
        q_pred = self.model(s)
        predicted_return, _ = torch.max(q_pred, axis=1)

        # target return
        q_next = self.model_target(sn) 
        max_q_next, _ = torch.max(q_next, axis=1)
        target_return = rn + (self.gamma * max_q_next) * (1 - done)

        # loss
        loss = self.model.loss(predicted_return, target_return)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        self.learns += 1

        return loss.item()

    def update_target(self):
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)