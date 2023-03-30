import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.fmac_dueling_critic import FMACDuelingCritic
import torch as th
from torch.optim import RMSprop, Adam
from torch import nn
import torch.nn.functional as F
from modules.mixers.dop import DopMixer
from components.action_selectors import categorical_entropy
from utils.rl_utils import build_td_lambda_targets


class SMSLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic1 = FMACDuelingCritic(scheme, args)
        self.critic2 = FMACDuelingCritic(scheme, args)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())

        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1:  # if just 1 agent do not mix anything
            if args.mixer == "dop":
                self.mixer = DopMixer(args)
            else:
                raise ValueError("Mixer {} is not supported.".format(args.mixer))
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.selector = copy.deepcopy(self.mac.agent.selector)
        self.selector_optimizer = Adam(params=self.selector.parameters(), lr=args.msg_lr)
        self.selector_criterion = th.nn.MSELoss()

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0

    def shapley_message_value(self, batch, t):
        bs = batch.batch_size
        n_agents = self.args.n_agents
        device = self.args.device
        sample_size = self.args.sample_size
        shap_msg_val = th.zeros((bs, sample_size, n_agents, n_agents)).to(device)
        if self.args.env == "listener_speaker":
            probability_mat = th.ones(bs*sample_size, n_agents, n_agents).to(device)
            probability_mat = probability_mat.reshape(-1, n_agents) / n_agents
            grand_coalitions = th.multinomial(probability_mat,
                                              n_agents,
                                              replacement=False,
                                              ).to(device)
            grand_coalitions = grand_coalitions.reshape(bs, sample_size, n_agents, -1) # grand_coalition[bs, sample_size, n_agents, n_agents]
        else:
            # probability_mat helps to generate samples of grand_coalitions, which (grand coalition) is an ordering of all agents excepts the recipient
            probability_mat = (th.ones(bs*sample_size, n_agents, n_agents) - th.eye(n_agents).unsqueeze(0).expand(bs*sample_size, -1, -1)).to(device)
            probability_mat = probability_mat.reshape(-1, n_agents) / (n_agents-1)
            grand_coalitions = th.multinomial(probability_mat,
                                              n_agents-1,
                                              replacement=False,
                                              ).to(device)
            grand_coalitions = grand_coalitions.reshape(bs, sample_size, n_agents, -1) # grand_coalition[bs, sample_size, n_agents, n_agents-1]
            ego_agents = th.arange(n_agents).unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(bs, sample_size, -1, -1).to(device) # ego_agents[bs, sample_size, n_agents, 1]
            grand_coalitions = th.cat((ego_agents, grand_coalitions), dim=-1) # grand_coalitions[bs, sample_size, n_agents, n_agents]

        # initialize message mask
        msg_mask_agent = th.zeros((bs, sample_size, n_agents, n_agents)).to(device)
        for i in range(n_agents):
            # modify message mask
            if self.args.env == "listener_speaker" or i != 0:
                msg_mask_agent.scatter_(3, grand_coalitions[:, :, :, i:i+1], 1)
            for s in range(sample_size):
                msg_mask_agent_per_sample = msg_mask_agent[:, s, :, :] # msg_mask_agent_per_sample[bs, n_agents, n_agents]
                pi = self.target_mac.forward(batch, t, 0, gumbel=False, msg_mask_agent=msg_mask_agent_per_sample, eval_shap=True).unsqueeze(1) # pi[bs, 1, n_agents, n_actions] # consider the message selector
                Q1, _ = self.critic1(batch[:, t:t+1], pi) # Q1[bs, 1, n_agents, 1]
                Q2, _ = self.critic2(batch[:, t:t+1], pi) # Q2[bs, 1, n_agents, 1]
                Q = th.min(Q1, Q2).detach()
                shap_msg_val[:, s:s+1, :, :].scatter_(3, grand_coalitions[:, s:s+1, :, i:i+1], Q)

        for i in reversed(range(1, n_agents)):
            cur_val = shap_msg_val.gather(3, grand_coalitions[:, :, :, i:i+1])
            pre_val = shap_msg_val.gather(3, grand_coalitions[:, :, :, i-1:i])
            shap_msg_val.scatter_(3, grand_coalitions[:, :, :, i:i+1], cur_val-pre_val)

        # deal with the special case in listener-speaker
        if self.args.env == "listener_speaker":
            empty_msg_mask_agent = th.zeros((bs, n_agents, n_agents)).to(device) # empty_msg_mask_agent[bs, n_agents, n_agents]
            pi = self.target_mac.forward(batch, t, 0, gumbel=False, msg_mask_agent=empty_msg_mask_agent, eval_shap=True).unsqueeze(1) # pi[bs, 1, n_agents, n_actions] # consider the message selector
            Q = th.min(self.critic1(batch[:, t:t+1], pi)[0], self.critic2(batch[:, t:t+1], pi)[0]).detach() # Q[bs, 1, n_agents, 1]
            cur_val = shap_msg_val.gather(3, grand_coalitions[:, :, :, 0:1])
            pre_val = Q.repeat(1, sample_size, 1, 1)
            shap_msg_val.scatter_(3, grand_coalitions[:, :, :, 0:1], cur_val-pre_val)

        return shap_msg_val.mean(1).detach()

    def train_msg_selector(self, inputs, labels):
        for epoch in range(self.args.msg_n_epochs):
            outputs = self.selector(inputs) # outputs[bs, n_agents]
            if self.args.env != "listener_speaker":
                n_agents = self.args.n_agents
                mask = (1-th.eye(n_agents).to(self.args.device)).repeat(int(inputs.size(0)/n_agents), 1)
                assert mask.shape == outputs.shape
                outputs_masked = outputs * mask
                loss = self.selector_criterion(outputs_masked, labels) * (n_agents/(n_agents-1))
            else:
                loss = self.selector_criterion(outputs, labels)

            self.selector_optimizer.zero_grad()
            loss.backward()
            self.selector_optimizer.step()

        return loss.detach()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, off=False):
        # Get the relevant data
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1-terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Retrace Q target
        with th.no_grad():
            if off:
                # message relabeling
                self.target_mac.init_hidden(batch.batch_size)
                # actions_relabel_onehot = []
                pi = []
                for t in range(batch.max_seq_length):
                    agent_outs = self.target_mac.forward(batch, t_ep=t, t_env=t_env, gumbel=False) # agent_outs[bs, n_agents, n_actions]
                    pi.append(agent_outs)
                pi = th.stack(pi, dim=1) # pi[bs, time_length, n_agents, n_actions]
                q = th.min(self.target_critic1(batch, pi.detach())[0], self.target_critic2(batch, pi.detach())[0])
            else:
                q = th.min(self.target_critic1(batch, batch["actions_onehot"].detach())[0], self.target_critic2(batch, batch["actions_onehot"].detach())[0])

            target_vals = self.target_mixer(q, batch["state"])

            lambd = 0 if off else self.args.lambd
            target_vals = build_td_lambda_targets(rewards, terminated, mask, target_vals, self.n_agents, self.args.gamma, lambd)

        # Train the critic
        # Current Q network forward
        q1, _ = self.critic1(batch[:, :-1], actions_onehot.detach()) # q1[bs, time_length, n_agents, 1]
        q2, _ = self.critic2(batch[:, :-1], actions_onehot.detach()) # q2[bs, time_length, n_agents, 1]
        q1_taken = self.mixer(q1, batch["state"][:, :-1]) # q1_taken[bs, time_length, 1]
        q2_taken = self.mixer(q2, batch["state"][:, :-1]) # q2_taken[bs, time_length, 1]
        critic_loss = 0.5 * ((q1_taken - target_vals.detach()) * mask).pow(2).sum() / mask.sum() + 0.5 * ((q2_taken - target_vals.detach()) * mask).pow(2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # Train the actor
        if not off:
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.mac.forward(batch, t_ep=t, t_env=0, dropout=self.args.message_dropout, test_mode=True) # not apply selector, no noise
                pi.append(agent_outs)
            pi = th.stack(pi, dim=1) # Concat over time b, t, a, probs # pi.shape == (batch_size, time_length, n_agents, n_actions)

            q1, _ = self.critic1(batch[:, :-1], pi) # q1.shape == (batch_size, time_length, n_agents, 1)
            q = self.mixer(q1, batch["state"][:, :-1]) # q.shape == (batch_size, time_length, 1)
            pg_loss = -(q * mask).sum() / mask.sum() # mask.shape == (batch_size, time_length, 1)

            entropy_loss = categorical_entropy(pi).mean(-1, keepdim=True) # mean over agents
            entropy_loss[mask == 0] = 0 # fill nan
            entropy_loss = (entropy_loss * mask).sum() / mask.sum()
            loss = pg_loss - self.args.entropy_coef * entropy_loss / entropy_loss.item()

            self.agent_optimiser.zero_grad()
            loss.backward()
            agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        # train message selector
        selector_loss = None
        if self.args.enable_msg_selector and not off:
            inputs = []
            smvs = []
            self.mac.init_hidden(batch.batch_size)
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                self.mac.forward(batch, t_ep=t, t_env=t_env) # update the hidden state of agents
                agent_input = self.mac._build_inputs(batch, t)
                inputs.append(agent_input)
                smv = self.shapley_message_value(batch, t)
                smvs.append(smv)
            inputs = th.cat(inputs, dim=0).view(batch.batch_size * (batch.max_seq_length - 1) * self.args.n_agents, -1)
            smvs = th.cat(smvs, dim=0).view(-1, self.n_agents)  # smvs[bs*time_length*n_agents, n_agents]
            k = self.mixer.k(batch["state"][:, :-1]).unsqueeze(-1).repeat(1, 1, 1, self.args.n_agents).view(-1, self.args.n_agents)
            labels = smvs * k

            selector_loss = self.train_msg_selector(inputs, labels.detach())

        # target_update
        if not off and (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # log
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm.item(), t_env)
            self.logger.log_stat("target_vals", (target_vals * mask).sum().item() / mask.sum().item(), t_env)
            if self.args.enable_msg_selector and selector_loss != None:
                self.logger.log_stat("selector_loss", selector_loss.item(), t_env)

            if not off:
                self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
                self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
                self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
                agent_mask = mask.repeat(1, 1, self.n_agents)
                self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * agent_mask).sum().item() / agent_mask.sum().item(), t_env)
                self.policy_log_stats_t = t_env
                self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        if self.args.enable_msg_selector:
            self.mac.agent.update_selector(self.selector)

    def cuda(self, device="cuda"):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic1.to(device=device)
        self.critic2.to(device=device)
        self.target_critic1.to(device=device)
        self.target_critic2.to(device=device)
        if self.mixer is not None:
            self.mixer.to(device=device)
            self.target_mixer.to(device=device)
        self.selector.to(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if self.args.enable_msg_selector:
            th.save(self.selector.state_dict(), "{}/selector.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))