__all__ = ['Monitor', 'get_monitor_files', 'load_results']

import gym
from gym.core import Wrapper
import time
from glob import glob
import csv
import os.path as osp
import json
import numpy as np
import tensorflow as tf

# added by Daniel
from baselines import logger

class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.results_writer = ResultsWriter(
            filename,
            header={"t_start": time.time(), 'env_id' : env.spec and env.spec.id},
            extra_keys=reset_keywords + info_keywords
        )
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.rewards_v = None
        self.rewards_h = None
        self.step_rewards = None
        self.needs_reset = True
        self.steps = 0
        self.last_obs = None
        self.episode_actions = None
        self.episode_obs = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        self.reset_state()
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset'%k)
            self.current_reset_info[k] = v
        self.last_obs = self.env.reset(**kwargs)
        return self.last_obs

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.steps = 0
        self.last_obs = None
        self.rewards = []
        self.rewards_h = []
        self.rewards_v = []
        self.step_rewards = []
        self.episode_actions = []
        self.episode_obs = []
        self.needs_reset = False


    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)

        # log per-step parameters - Daniel
        logger.logkv("step_reward", info["step_reward"])

        self.update(ob, rew, done, info, action[0])

        logger.dumpkvs()
        return (ob, rew, done, info)

    def update(self, ob, rew, done, info, ac):
        self.steps += 1
        self.rewards.append(rew)
        self.step_rewards.append(info["step_reward"])
        self.rewards_v.append(info['r_v'])
        self.rewards_h.append(info['r_h'])
        self.episode_actions.append(ac)
        self.episode_obs.append(self.last_obs)
        self.last_obs = ob
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eprew_h = sum(self.rewards_h)
            eprew_v = sum(self.rewards_v)
            taskrew = sum(self.step_rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6),
                      "l": eplen,
                      "t": round(time.time() - self.tstart, 6),
                      "a": self.episode_actions,
                      "o": self.episode_obs}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            self.results_writer.write_row(epinfo)

            if isinstance(info, dict):
                info['episode'] = epinfo

            # log per-episode stats -- Daniel
            logger.logkv("terminal_reward", info["terminal_reward"])
            logger.logkv("collected_termrew", info["collected_termrew"])
            logger.logkv("task_reward", round(taskrew, 6))
            logger.logkv("eprew", epinfo["r"])
            logger.logkv("eprew_v", round(eprew_v, 6))
            logger.logkv("eprew_h", round(eprew_h, 6))
            logger.logkv("eplen", epinfo["l"])
            logger.logkv("task_length", info["task_length"])

        self.total_steps += 1


    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times

class LoadMonitorResultsError(Exception):
    pass


class ResultsWriter(object):
    def __init__(self, filename=None, header='', extra_keys=()):
        self.DEFAULT_KEYS = ('r', 'l', 't', 'a', 'o')
        self.extra_keys = extra_keys
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            if isinstance(header, dict):
                header = '# {} \n'.format(json.dumps(header))
            self.f.write(header)
            self.logger = csv.DictWriter(self.f, fieldnames=self.DEFAULT_KEYS+tuple(extra_keys))
            self.logger.writeheader()
            self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()



def get_monitor_files(dir):
    return glob(osp.join(dir, "*" + Monitor.EXT))

def load_results(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*monitor.json")) +
        glob(osp.join(dir, "*monitor.csv"))) # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                if not firstline:
                    continue
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'): # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers # HACK to preserve backwards compatibility
    return df

def test_monitor():
    env = gym.make("CartPole-v1")
    env.seed(0)
    mon_file = "/tmp/baselines-test-%s.monitor.csv" % uuid.uuid4()
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(1000):
        _, _, done, _ = menv.step(0)
        if done:
            menv.reset()

    f = open(mon_file, 'rt')

    firstline = f.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 'gym_version', 't_start'},  "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(f, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    f.close()
    os.remove(mon_file)
