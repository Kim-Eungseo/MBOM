import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input, output, hidden_layers_features, output_type=None, activation="relu"):
        super(MLP, self).__init__()
        self.input = input
        self.output = output
        self.hidden_layers_features = hidden_layers_features
        self.output_type = output_type
        self.activation = activation

        self.layers = []
        self.layers.append(nn.Linear(self.input, self.hidden_layers_features[0]))
        for i in range(len(hidden_layers_features) - 1):
            self.layers.append(nn.Linear(self.hidden_layers_features[i], self.hidden_layers_features[i + 1]))
        self.layers = nn.ModuleList(self.layers)

        if self.output_type == "gauss":
            self.mu = nn.Linear(self.hidden_layers_features[-1], self.output)
            self.sigma = nn.Linear(self.hidden_layers_features[-1], self.output)
        else:
            self.out = nn.Linear(self.hidden_layers_features[-1], self.output)

    def _activate(self, x):
        if self.activation == "relu":
            return torch.relu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        return torch.relu(x)

    def forward(self, x):
        assert type(x) is torch.Tensor, "net forward input type error"
        for i, layer in enumerate(self.layers):
            x = self._activate(layer(x))

        if self.output_type == "gauss":
            mu = torch.tanh(self.mu(x))
            sigma = nn.functional.softplus(self.sigma(x))
            return mu, sigma
        elif self.output_type == "prob":
            hidden_prob = x.detach()
            x = nn.functional.softmax(self.out(x), dim=-1)
            return x, x.detach()
        else:
            x = self.out(x)
            return x
