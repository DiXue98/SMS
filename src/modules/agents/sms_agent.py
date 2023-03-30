import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SMSAgent(nn.Module):
    def __init__(self, input_shape, args, scheme):
        super(SMSAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        if args.use_rnn:
            self.policy = nn.Linear(args.rnn_hidden_dim+args.n_agents*args.msg_dim, args.n_actions)
        else:
            self.policy = nn.Linear(input_shape+args.n_agents*args.msg_dim, args.n_actions)

        self.msg_encoder = nn.Sequential(
            nn.Linear(scheme["obs"]["vshape"]+self.args.n_agents, args.msg_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.msg_hidden_dim, args.msg_dim*args.n_agents)
        )

        self.selector = nn.Sequential(nn.Linear(input_shape, args.selector_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.selector_hidden_dim, args.n_agents),
                                      )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def msg_rec(self, msg_mat, msg_mask_agent):
        """
        The received message under peer to peer mechanism with agents masked
        msg_mat.shape == (bs, n_agents, n_agents, msg_dim)
        msg_mask.shape == (bs, n_agents, n_agents)
        return: msg_received, which stands for the messages received by each agent with masked messages dropped
        """
        bs, n_agents, _, msg_dim = msg_mat.shape
        msg_mask = msg_mask_agent.unsqueeze(-1).expand(-1, -1, -1, msg_dim) # msg_mask[bs, n_agents, n_agents, msg_dim]
        # msg_mask = msg_mask_agent.unsqueeze(-1).expand(-1, -1, -1, self.args.msg_shape) # msg_mask[bs, n_agents, n_agents, msg_dim]
        assert msg_mat.shape == msg_mask.shape, "msg_mat does not match msg_mask"
        msg_received = msg_mat * msg_mask
        msg_received = msg_received.reshape(bs, n_agents, -1)
        return msg_received

    def forward(self, inputs, batch, t_ep, hidden_state=None, dropout=False, message_selector_on=False, msg_mask_agent=None, test_mode=False):
        assert not (dropout == True and msg_mask_agent != None), "dropout and msg_mask_agent can not be specified at the same time"
        assert not (message_selector_on == True and msg_mask_agent != None)
        
        b, a, e = inputs.size()
        
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)

        # sending and receiving message process -------------------------------------------
        obs = batch["obs"][:, t_ep]
        obs_id = (obs, th.eye(self.args.n_agents).unsqueeze(0).repeat(b, 1, 1).to(self.args.device))
        obs_id = th.cat(obs_id, dim=-1)

        if self.args.env == "listener_speaker":
            msg_cont = batch["msg"][:, t_ep]
            msg_mat = msg_cont.view(b, a, a, 2).transpose(1, 2)
        else:
            msg_cont = obs_id # msg_cont[bs, n_agents, obs_diim]: msg_cont[bs, i] the information contained in ith agent's message
            msg = self.msg_encoder(msg_cont)

            msg_mat = msg.view(b, a, a, -1).transpose(1, 2) # msg_mat[bs, n_agents, n_agents, msg_dim]: msg_mat[bs, i, j] the message agents i received from agent j

            if not test_mode:
                msg_mat += th.randn(msg_mat.shape).to(self.args.device) # noise injection

        msg_counter = None

        if msg_mask_agent==None:
            if self.args.env == "listener_speaker":
                msg_mask_agent = th.ones(b, a, a)
            else:
                msg_mask_agent = th.ones(b, a, a) - th.eye(a).unsqueeze(0).expand(b, -1, -1) # full communication mask
            msg_mask_agent = msg_mask_agent.to(self.args.device)

            # Apply message selector
            if message_selector_on:
                selector_mask_agent = (self.selector(inputs.view(b, a, -1))>0).float()

                msg_mask_agent *= selector_mask_agent

            # Dropout
            if dropout:
                msg_mask_agent *= (th.rand(b, a, a)>self.args.dropout_rate).float().to(self.args.device)


            msg_counter = msg_mask_agent # msg_counter[bs, n_agents, n_agents]

        # Pooling the messages
        msg_rec = self.msg_rec(msg_mat, msg_mask_agent) # msg_rec.shape[bs, n_agents, n_agents*msg_dim]
        # ---------------------------------------------------------------------------------
        if self.args.use_rnn:
            policy_inputs = th.cat([h, msg_rec.view(b*a, -1)], dim=-1)
        else:
            policy_inputs = th.cat([inputs.reshape(b*a, -1), msg_rec.reshape(b*a, -1)], dim=-1)
        action_logits = self.policy(policy_inputs)

        return action_logits.view(b, a, -1), h.view(b, a, -1), msg_counter

    def update_selector(self, cur_selector):
        self.selector.load_state_dict(cur_selector.state_dict())