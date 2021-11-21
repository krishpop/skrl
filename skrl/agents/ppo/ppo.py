from typing import Union, Dict

import gym
import torch
import torch.nn.functional as F

from ...env import Environment
from ...memories import Memory
from ...models.torch import Model

from .. import Agent


PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    
    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.99,                 # TD(lambda) coefficient (lam) for computing returns and advantages
    
    "policy_learning_rate": 1e-3,   # policy learning rate
    "value_learning_rate": 1e-3,    # value learning rate

    "random_timesteps": 1000,       # random exploration steps
    "learning_starts": 1000,        # learning starts after this many steps

    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold

    "device": None,                 # computing device
}


class PPO(Agent):
    def __init__(self, env: Union[Environment, gym.Env], networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
        """
        Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347
        """
        PPO_DEFAULT_CONFIG.update(cfg)
        super().__init__(env=env, networks=networks, memory=memory, cfg=PPO_DEFAULT_CONFIG)

        # networks
        if not "policy" in self.networks.keys():
            raise KeyError("Policy network not found in networks. Use 'policy' key to define the policy network")
        if not "value" in self.networks.keys():
            raise KeyError("Value-network not found in networks. Use 'value' key to define the Value-network")
        
        self.policy = self.networks["policy"]
        self.value = self.networks["value"]

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._policy_learning_rate = self.cfg["policy_learning_rate"]
        self._value_learning_rate = self.cfg["value_learning_rate"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        # set up optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._policy_learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self._value_learning_rate)

        # create tensors in memory
        self.memory.create_tensor(name="states", size=self.env.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.env.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)
        self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

        self.tensors_names = ["states", "actions", "rewards", "dones", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, inference: bool = False, timestep: Union[int, None] = None, timesteps: Union[int, None] = None) -> torch.Tensor:
        """
        Process the environments' states to make a decision (actions) using the main policy

        Parameters
        ----------
        states: torch.Tensor
            Environments' states
        inference: bool
            Flag to indicate whether the network is making inference
        timestep: int or None
            Current timestep
        timesteps: int or None
            Number of timesteps
            
        Returns
        -------
        torch.Tensor
            Actions
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(states)

        # sample stochastic actions
        actions, log_prob, actions_mean = self.policy.act(states, inference=inference)
        self._current_log_prob = log_prob

        return actions, log_prob, actions_mean

    def record_transition(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, timestep: int, timesteps: int) -> None:
        """
        Record an environment transition in memory
        
        Parameters
        ----------
        states: torch.Tensor
            Observations/states of the environment used to make the decision
        actions: torch.Tensor
            Actions taken by the agent
        rewards: torch.Tensor
            Instant rewards achieved by the current actions
        next_states: torch.Tensor
            Next observations/states of the environment
        dones: torch.Tensor
            Signals to indicate that episodes have ended
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        super().record_transition(states, actions, rewards, next_states, dones, timestep, timesteps)

        self._current_next_states = next_states

        if self.memory is not None:
            values, _, _ = self.value.act(states=states, inference=True)
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones, 
                                    log_prob=self._current_log_prob, values=values)            

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """
        Callback called before the interaction with the environment

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """
        Callback called after the interaction with the environment

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self._update(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int):
        # compute returns and advantages
        last_values, _, _ = self.value.act(states=self._current_next_states, inference=True)
        computing_hyperparameters = {"discount_factor": self._discount_factor,
                                     "lambda_coefficient": self._lambda,
                                     "normalize_returns": False,
                                     "normalize_advantages": True}
        self.memory.compute_functions(returns_dst="returns", advantages_dst="advantages", last_values=last_values, hyperparameters=computing_hyperparameters)

        # sample a batch from memory
        sampled_states, sampled_actions, _, _, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages = self.memory.sample_all(self.tensors_names)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            _, next_log_prob, _ = self.policy.act(states=sampled_states, taken_actions=sampled_actions)

            # early stopping with KL divergence
            if self._kl_threshold:
                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl = ((torch.exp(ratio) - 1) - ratio).mean()
                
                if kl > self._kl_threshold:
                    print("[INFO] Early stopping (learning epoch: {}). KL divergence ({}) > KL divergence threshold ({})".format(epoch, kl, self._kl_threshold))
                    break

            # entropy loss
            if self._entropy_loss_scale:
                entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy().mean()
            else:
                entropy_loss = 0
            
            # policy loss
            ratio = torch.exp(next_log_prob - sampled_log_prob)
            surrogate = sampled_advantages * ratio
            surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
            
            policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

            self.policy_optimizer.zero_grad()
            (policy_loss + entropy_loss).backward()
            self.policy_optimizer.step()

            # value loss
            predicted_values, _, _ = self.value.act(states=sampled_states)

            if self._clip_predicted_values:
                predicted_values = sampled_values + torch.clip(predicted_values - sampled_values, -self._value_clip, self._value_clip)
            value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # update cumulative losses
            cumulative_policy_loss += policy_loss.item()
            cumulative_value_loss += value_loss.item()
            if self._entropy_loss_scale:
                cumulative_entropy_loss += entropy_loss.item()

        # record data
        self.writer.add_scalar('Loss/policy', cumulative_policy_loss / self._learning_epochs, timestep)
        self.writer.add_scalar('Loss/value', cumulative_value_loss / self._learning_epochs, timestep)

        if self._entropy_loss_scale:
            self.writer.add_scalar('Loss/entropy', cumulative_entropy_loss / self._learning_epochs, timestep)
