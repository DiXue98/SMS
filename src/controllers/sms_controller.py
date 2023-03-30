from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class SMSMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape, scheme)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None
        self.msg_count = 0

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        agent_outputs = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode, gumbel=not (test_mode and self.args.test_greedy), dropout=(self.args.name == "drop"))
        actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        return actions

    def forward(self, ep_batch, t_ep, t_env, test_mode=False, gumbel=False, msg_mask_agent=None, dropout=False, eval_shap=False):
        # When self.args.enable_msg_selector and (msg_mask_agent!=None or t_env>=self.args.start_msg_select_timestep), the message selector is activated.
        agent_inputs = self._build_inputs(ep_batch, t_ep)
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        # Decide whether the message selector should be activated
        message_selector_on = (self.args.enable_msg_selector and t_env>=self.args.start_msg_select_timestep) and (not msg_mask_agent!=None)

        if eval_shap==False:
            agent_outs, self.hidden_states, self.msg_count = self.agent(agent_inputs, ep_batch, t_ep, self.hidden_states, dropout=dropout, message_selector_on=message_selector_on, msg_mask_agent=msg_mask_agent, test_mode=test_mode)
        else:
            agent_outs, _, self.msg_count = self.agent(agent_inputs, ep_batch, t_ep, self.hidden_states, dropout=dropout, message_selector_on=message_selector_on, msg_mask_agent=msg_mask_agent, test_mode=test_mode)


        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            if gumbel:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_agents(self, input_shape, scheme):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args, scheme)