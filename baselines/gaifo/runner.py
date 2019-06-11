import numpy as np
from baselines import logger
from baselines.common.runners import AbstractEnvRunner
import time

class Runner(AbstractEnvRunner):
    """
    Largely based on Runner from openai/baselines/ppo
    Daniel Zvara 2019

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, reward_giver, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        # Adversary reward giver
        self.reward_giver = reward_giver

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_true_rewards, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [],[],[],[],[],[]
        mb_states = self.states = self.model.initial_state

        curr_ep_ret = np.zeros((self.env.num_envs,1))
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            obs_next, true_rewards, self.dones, infos = self.env.step(actions)
            rewards = self.reward_giver.get_reward(self.obs, obs_next)
            self.obs[:] = obs_next
            curr_ep_ret += rewards

            for i in range(len(self.dones)):
                if self.dones[i]:
                    #if self.states:
                    self.states[i] = self.model.initial_state[0]
                    if 'episode' in infos[i]:
                        infos[i]['episode']['ar'] = curr_ep_ret[i]

                    curr_ep_ret[i] = 0

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(np.squeeze(rewards))
            mb_true_rewards.append(true_rewards)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_true_rewards = np.asarray(mb_true_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = np.expand_dims(mb_advs, axis=1) + (mb_values)
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


