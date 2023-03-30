from numpy.core.numeric import True_
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layer.self_atten import SelfAttention


class FMACDuelingCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FMACDuelingCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "q"
        self.hidden_states = None
        self.critic_hidden_dim = args.critic_hidden_dim 

        # Set up network layers

        # advantage
        self.a = nn.Sequential(nn.Linear(self.input_shape+self.n_actions, self.critic_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.critic_hidden_dim, 1),
                               )

        # state value
        self.v = nn.Sequential(nn.Linear(self.input_shape, self.critic_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.critic_hidden_dim, 1),
                               )

    def forward(self, inputs, actions, hidden_state=None):
        bs = inputs.batch_size
        ts = inputs.max_seq_length

        inputs = self._build_inputs(inputs)
        a_inputs = th.cat([inputs, actions], dim=-1)
        actions_onehot = th.eye(self.n_actions).to(self.args.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(bs, ts, self.n_agents, 1, 1)
        dueling_inputs = th.cat([inputs.unsqueeze(-2).repeat(1, 1, 1, self.n_actions, 1), actions_onehot], dim=-1)
        a1 = self.a(a_inputs) - self.a(dueling_inputs).mean(dim=-2)
        v1 = self.v(inputs)
        assert a1.shape == v1.shape
        q1 = a1 + v1

        return q1, hidden_state

    def _build_inputs(self, batch):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        ts = batch.max_seq_length
        inputs = []
        inputs.append(batch["obs"])  # b1av
        inputs.append(batch["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1))  # b1av
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device)\
                .unsqueeze(0).unsqueeze(0).expand(bs, ts, -1, -1))
        inputs = th.cat([x.reshape(bs, ts, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        input_shape += scheme["state"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape