import numpy as np
import torch
import torch.optim as optim

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer
import torch.nn.functional as F  # TODO: remove


class DQNAgent:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(
            self,
            n_states, n_actions, layer_sizes,
            epsilon=.1,
            replay_buffer_size=int(1e4),
            batch_size=128,
            gamma=.99,
            alpha=5e-4, # learning rate
            tau=1e-3, # for soft update
            n_learn=4,
            seed=42,
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.seed = seed

        # learning parameters
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-networks
        self.q_network_local = QNetwork(
            n_states, layer_sizes, n_actions, seed
        ).to(DQNAgent.device)
        self.q_network_target = QNetwork(
            n_states, layer_sizes, n_actions, seed
        ).to(DQNAgent.device)  # TODO: use copy.deepcopy instead

        # optimizer and loss function
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=alpha)
        self.criterion = torch.nn.MSELoss()

        # replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, batch_size, seed)

        # counters and auxiliary variables
        self.n_experience_tuples = 0
        self.n_learn = n_learn
        self.start_learn = False
        print("Q-Network\n", self.q_network_target)

    def observe(self, s0, a, r, s1, done):
        # store experience in buffer and train
        self.n_experience_tuples += 1
        self.replay_buffer.append(ReplayBuffer.Experience(s0, a, r, s1, done))
        if self.n_experience_tuples % self.n_learn == 0:
            experiences = self.replay_buffer.sample(device=DQNAgent.device)
            if experiences is not None:
                self.start_learn = True
                self.learn(experiences)

    def learn(self, experiences):
        s0, a, r, s1, done = experiences
        q0_local = self.q_network_local(s0).gather(1, a)
        q1_target = self.q_network_target(s1).detach().max(1)[0].unsqueeze(1)
        q0_target = r + self.gamma * q1_target * (1 - done)
        # loss = self.criterion(q0_local, q0_target)  # TODO: uncomment
        loss = F.mse_loss(q0_local, q0_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network(self.tau)

    def get_action(self, state, train=True):
        s = torch.from_numpy(state).float().unsqueeze(0).to(DQNAgent.device)
        if not self.start_learn or (train and np.random.rand() < self.epsilon):
            # with probability epsilon select a random action
            return np.random.randint(self.n_actions)
        else:
            # with probability 1-epsilon select best action
            self.q_network_local.eval()
            with torch.no_grad():
                qa = self.q_network_local(s)
            self.q_network_local.train()
            return np.argmax(qa.cpu().data.numpy())

    def update_target_network(self, tau):
        """
        Update weights in target network, perform a soft update
        """
        for p_target, p_local in zip(self.q_network_target.parameters(), self.q_network_local.parameters()):
            p_target.data.copy_(tau * p_local.data + (1.0-tau) * p_target.data)
