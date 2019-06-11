import isaacenv
import numpy as np
from gym import spaces
from baselines.common import tf_util as U

class HopperRunner():
    def __init__(self, policy_func, load_model_path, stochastic_policy, reuse=False):
        # FIXME: this is an ugly hack
        observation_space = spaces.Box(
            np.ones(3) * -np.Inf,
            np.ones(3) * np.Inf)

        action_space = spaces.Box(
            np.ones(2) * -1.,
            np.ones(2) * 1.)

        self.ob_space = observation_space
        self.ac_space = action_space
        self.pi = policy_func("pi", self.ob_space, self.ac_space, reuse=reuse)
        self.stochastic_policy = stochastic_policy

        U.initialize()
        # Prepare for rollouts
        # ----------------------------------------
        U.load_state(load_model_path)

        self.env = isaacenv.make("HopperPlay-v0", num_envs=1, spacing=0.5)

        self.obs_list = []
        self.acs_list = []
        self.terr_acs_list = []
        self.len_list = []
        self.ret_list = []

    def run(self, number_of_trials =-1, save=False):
        unlimited = number_of_trials == -1
        t = 0
        num_envs = self.env.num_envs
        assert num_envs == 1, "For running the environment, parallel environments are not supported"

        ob = self.env.reset()
        while t < number_of_trials or unlimited:
            obs = []
            acs = []
            terr_acs = []

            kill = [False]
            contact = False

            while not kill[0]:
                if not contact:
                    ac, _, kill, info = self.env.step([[None]*2])
                else:
                    ac, _ = self.pi.act(self.stochastic_policy, ob)
                    ob, _, kill, info = self.env.step(ac)
                    terr_ac = info[0]["terrain_action"]
                    obs.append(ob[0])
                    acs.append(ac[0])
                    terr_acs.append(terr_ac)

                contact = info[0]["contact"]
                self.env.render()

            self.obs_list.append(obs)
            self.acs_list.append(acs)
            self.terr_acs_list.append(terr_acs)
            t += 1
            print("Trial: " + str(t))

        if save:
            np.savez("stateac", ac=self.acs_list, tac=self.terr_acs_list, ob=self.obs_list)
