from collections import namedtuple, deque
import numpy as np

import torch


class ReplayBuffer:

    # TODO: consider inheriting from data provider
    # use next(iter()) rather than sample() function

    Experience = namedtuple("Experience", ["s0", "a", "r", "s1", "done"])
    ExperienceBatch = namedtuple("ExperienceBatch", ["S0", "A", "R", "S1", "dones"])

    def __init__(self, buffer_size, batch_size, seed):
        self.batch_size = batch_size
        self.seed = seed
        self.experiences = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.experiences)

    def append(self, experience):
        self.experiences.append(experience)

    def sample(self, batch_size=None, device="cpu"):
        batch_size = batch_size or self.batch_size
        if len(self) < batch_size:
            return None

        batch_indices = np.random.choice(
            range(len(self)), batch_size, replace=False
            # TODO: p = probs for priority sampling
        )
        batch = [self.experiences[i] for i in batch_indices]

        return ReplayBuffer.ExperienceBatch(
            torch.from_numpy(np.vstack([e.s0 for e in batch])).float().to(device),
            torch.from_numpy(np.vstack([e.a for e in batch])).long().to(device),
            torch.from_numpy(np.vstack([e.r for e in batch])).float().to(device),
            torch.from_numpy(np.vstack([e.s1 for e in batch])).float().to(device),
            torch.from_numpy(np.vstack([float(e.done) for e in batch])).float().to(device)
        )
