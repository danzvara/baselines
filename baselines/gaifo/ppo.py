import os
import time
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.tf_util import get_session
from baselines.common.policies import build_policy, build_policy_noenv
from baselines.common import explained_variance, zipsame, dataset, fmt_row

from baselines.gaifo.runner import Runner
from baselines.common import colorize, tf_util as U
from contextlib import contextmanager


def constfn(val):
    def f(_):
        return val
    return f


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def learn(*, network, env, reward_giver, expert_dataset, total_timesteps, eval_env = None, seed=None,
            nsteps=512, ent_coef=0.00, lr=3e-4, g_step = 1, d_step = 10,
            vf_coef=0.9,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=1, nminibatches=1, noptepochs=3, cliprange=0.2,
            save_interval=5, max_timesteps=0, load_path=None, model_fn=None, **network_kwargs):

    '''

    :param total_timesteps: number of timesteps to execute, overall
    :param nsteps: number of steps to run per each environment, per update, batch size = nsteps * nenv
    :param nminibatches: number of minibatches per update. For recurrent, nenv % nminibatches == 0
    :param noptepochs: Number of epochs when optimizing surrogate
    '''
    '''

    Learn GAIFO policy using PPO
    Based on PPO2 implementation from openai/baselines

    Daniel Zvara (06/2019)

    '''

    @contextmanager
    def timed(msg):
        print(colorize(msg, color='magenta'))
        tstart = time.time()
        yield
        print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))

    # save algo config -- Daniel
    logger.save_config(locals())

    # get summary writer
    network_summary_writer = network_summary()

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    d_adam = tf.train.AdamOptimizer(name="d_adam").minimize(reward_giver.total_loss)
    with timed("Building policy"):
        policy = build_policy(env, network, **network_kwargs)

    sess = tf.get_default_session()

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    with timed("Building ppo model"):
        model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm) #var_scope='ppo2_model_imitation')

    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    with timed("Creating runner:"):
        runner = Runner(env=env, model=model, reward_giver=reward_giver, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)

    nupdates = total_timesteps // nbatch
    update, iteration = 0, 0

    while update < nupdates:
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")

        ob_expert, obn_expert = expert_dataset.get_next_batch(nbatch)
        batch_size = nbatch // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch

        # ob_t[i] => ob_tn[i] is one transition
        ob_t = obs[:-1]
        ob_tn = obs[1:]

        for ob_batch, obn_batch in dataset.iterbatches((ob_t, ob_tn),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            ob_expert, obn_expert = expert_dataset.get_next_batch(batch_size)
            ob_expert = ob_expert[:,:3]
            obn_expert = obn_expert[:,:3]
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"):
                ob_all = np.concatenate((ob_batch, obn_batch), 0)
                ob_all = np.concatenate((ob_all, ob_expert), 0)
                ob_all = np.concatenate((ob_all, obn_expert), 0)
                reward_giver.obs_rms.update(ob_all)

            feed_dict = {
                reward_giver.generator_obs_ph: ob_batch,
                reward_giver.generator_obsn_ph: obn_batch,
                reward_giver.expert_obs_ph: ob_expert,
                reward_giver.expert_obsn_ph: obn_expert
            }

            _, newlosses = sess.run([d_adam, reward_giver.losses], feed_dict)

            d_losses.append(newlosses)

        # ------------------ Update Policy ------------------
        logger.log("Optimizing policy")
        for policy_step in range(g_step):
            assert nbatch % nminibatches == 0

            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = lr(frac)

            cliprangenow = cliprange(frac)
            if policy_step > 0:
                obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()

            epinfobuf.extend(epinfos)

            # Here what we're going to do is for each minibatch calculate the loss and append it.
            mblossvals = []
            if states is None: # nonrecurrent version
                # Index of each element of batch_size
                # Create the indices array
                inds = np.arange(nbatch)
                for _ in range(noptepochs):
                    # Randomize the indexes
                    np.random.shuffle(inds)
                    # 0 to batch_size with batch_train_size step
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        metrics = model.train(lrnow, cliprangenow, *slices)
                        losses = metrics[:-1]
                        mblossvals.append(losses)
            else: # recurrent version
                assert nenvs % nminibatches == 0
                envsperbatch = nenvs // nminibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
                envsperbatch = nbatch_train // nsteps

                # only one update, as we do separate policy update steps with fresh data
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    metrics = model.train(lrnow, cliprangenow, *slices, states=mbstates)
                    losses = metrics[:-1]
                    mblossvals.append(losses)

            update += 1
            lossvals = np.mean(mblossvals, axis=0)

        ev = explained_variance(values, returns)
        logger.record_tabular("Explained Variance", ev)
        logger.record_tabular("EpLenMean", safemean([epinfo['l'] for epinfo in epinfobuf]))
        logger.record_tabular("EpRewMean", safemean([epinfo['ar'] for epinfo in epinfobuf]))
        logger.record_tabular("EpTrueRewMean", safemean([epinfo['r'] for epinfo in epinfobuf]))
        for (lossval, lossname)  in zip(lossvals, model.loss_names):
            logger.record_tabular(lossname, lossval)
        logger.dump_tabular()


        logger.log(fmt_row(13, reward_giver.loss_name))
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))
        iteration += 1
        logger.log("___ Iteration " + str(iteration) + "_______")


def network_summary():
    """tf summary from network"""
    writer = tf.summary.FileWriter(logger.get_dir() + '/tb/histo')
    return writer