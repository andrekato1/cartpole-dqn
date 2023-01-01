import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, lr):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(32, action_dim)
        init.xavier_uniform_(self.fc2.weight)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x