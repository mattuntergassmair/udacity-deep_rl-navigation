import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, input_size, layer_sizes, output_size, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): Dimension of input layer
            layer_sizes (list[int]): Dimensions of hidden layers
            output_size (int): Dimension of output layer
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.decoding = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    # nn.Dropout(p=1./n_out),  # TODO: why are dropout layers bad for RL?
                    nn.ReLU()
                )
                for n_in, n_out in
                zip([input_size] + layer_sizes[:-1], layer_sizes)
            ),
            nn.Linear(layer_sizes[-1], output_size)
        )

    def forward(self, state):
        return self.decoding(state)

    def __repr__(self):
        return self.decoding.__repr__()

    def __str__(self):
        return self.decoding.__str__()
