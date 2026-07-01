import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs


class DuelingQNetwork(QNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        action_dim = int(self.action_space.n)

        value_net = create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        self.value_net = nn.Sequential(*value_net)

        advantage_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.advantage_net = nn.Sequential(*advantage_net)

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)

        value = self.value_net(features)
        advantage = self.advantage_net(features)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


class DuelingDQN(DQN):
    policy_aliases = {"MlpPolicy": DuelingDQNPolicy}