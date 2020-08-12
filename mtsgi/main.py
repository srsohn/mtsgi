import os, time, math, copy
from tqdm import tqdm
from collections import deque

import numpy as np
import tensorflow as tf

from mtsgi.graph.ILP import ILP
from mtsgi.graph.grprop import GRProp
from mtsgi.graph.graph_utils import dotdict
from mtsgi.rl import algo
from mtsgi.rl.arguments import get_args

from mtsgi.rl.utils import (prepare_logging, _save_eval_log,
                            _print_eval_log, tensorboard_eval)

# TODO: Change here for new website
from mtsgi.environment.toywob.batch_toywob_correspond import BatchToyWoBEnv
from mtsgi.environment.walmart.batch_walmart import BatchWalmartEnv
from mtsgi.environment.dicks.batch_dicks import BatchDicksEnv


def main(args):
    # It currently supports the MSGI-Random setup only.
    assert args.algo in ['random']
    assert args.infer
    assert args.mode == 'meta_eval'
    args.summary_interval = 1

    if args.seed > 0:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    args.save_interval = max(args.num_updates // 5, 1)
    args.cuda = False
    args.save_flag = False

    value_window, score_tr_window, score_tst_window, success_window = np.zeros((4, 10))
    print('num-processes=', args.num_processes)
    logs = prepare_logging(args)

    if args.env_name == 'toywob':
        envs = BatchToyWoBEnv(args)
    elif args.env_name == 'walmart':
        envs = BatchWalmartEnv(args)
    elif args.env_name == 'dicks':
        envs = BatchDicksEnv(args)
    else:
        raise NotImplementedError("Unknown env: " + args.env_name)

    args.act_dim = envs.action_space.n

    agent = algo.Random(args)

    print('feat_dim=', envs.feat_dim)
    #
    ilp = ILP(args)
    #
    logs.str, logs.stst, logs.act_tst, logs.ret, logs.succ_rate_tst = np.zeros((5, args.tr_epi))
    meta_eval(args, envs, agent, ilp, logs)

    # close the env
    if hasattr(envs, 'close'):
        envs.close()


def meta_eval(args, envs, agent, ilp, logs):
    assert not args.train
    value_loss, action_loss, dist_entropy = np.zeros((3, args.tr_epi))

    start = time.time()
    for g_ind in range(envs.num_graphs):
        # 0. init trial
        ep_logs = dotdict()
        ep_logs.str, ep_logs.stst, ep_logs.act_tst, ep_logs.ret = np.zeros((4, args.tr_epi))

        for epi_ind in tqdm(range(args.tr_epi), ncols=100, desc='Episode progress'):
            # 1. adaptation phase
            print('-' * 40)
            print('Adaptation')
            # collect samples
            score_tr, act_tr, ilp = rollout_trial(
                args, envs, agent, 1, train=args.train,
                infer=args.infer, ilp=ilp, reset_task = (epi_ind==0), gind=g_ind)
            vloss = 0
            aloss = 0
            dloss = 0

            # 2. inference
            if args.infer:
                graphs = ilp.infer_graph(ep=epi_ind, g_ind=g_ind)
                test_agent = GRProp(graphs, args)
            else:
                test_agent = agent

            # 3. test phase
            print('-' * 40)
            print('Testing')
            score_tst, act_tst, _ = rollout_trial(args, envs, test_agent, args.test_epi, p=(epi_ind == args.tr_epi - 1))

            # 4. logging
            logs.stst[epi_ind] += score_tst.mean()
            logs.act_tst[epi_ind] += act_tst
            ep_logs.stst[epi_ind] = score_tst.mean()
            ep_logs.act_tst[epi_ind] = act_tst
            logs.str[epi_ind] += score_tr.mean()
            logs.succ_rate_tst[epi_ind] += float(score_tst.mean() > 0)
            ep_logs.str[epi_ind] = score_tr.mean()
            # ### episode logging
            tensorboard_eval(args, logs, epi_ind, ep_logs=ep_logs, g_ind=g_ind,
                             value_loss=vloss, action_loss=aloss, dist_entropy=dloss)

        if args.method == 'rlrl':
            logs.stst[:-1] = logs.str[1:]

        # logging
        _print_eval_log(args, logs, g_ind+1, envs.num_graphs, start)
        logs.csv_filename = os.path.join('logs', args.folder_name, f'eval_log-g_ind{g_ind}-ep{epi_ind}.csv')
        _save_eval_log(args, logs, envs.num_graphs, value_loss, action_loss, dist_entropy)


def rollout_trial(args, envs, agent, nb_epi,
                  train=False, infer=False, ilp=None, reset_task=False, gind=-1, p=False):
    # infer = True for adaptation phase, False for test phase
    act_sum, score = 0, 0

    active = np.ones([args.num_processes, 1], dtype=np.int32)
    step_done = np.ones([args.num_processes, 1], dtype=np.float32)

    # 1. initialization: 20%
    obs, feats = envs.reset_trial(nb_epi=nb_epi, reset_graph=reset_task, graph_index=gind)

    # Initialize ILP
    if infer:
        if reset_task:
            ilp.reset(envs)
        _, tp_ind, elig_ind = envs.get_indexed_states() # only for inference
        ilp.insert(active, step_done, tp_ind, elig_ind)

    # Unroll an episode
    for step in range(args.ntasks * nb_epi * 2):
        if agent.algo == 'random' or agent.algo == 'greedy':
            action = agent.act(active, feats)
        elif agent.algo == 'grprop':
            mask_ind, tp_ind, elig_ind = envs.get_indexed_states()
            action = agent.act(active, mask_ind, tp_ind, elig_ind, eval_flag=True, p=False)
        else:
            assert(False)

        # 2. env step 80%
        assert isinstance(action, np.ndarray)
        obs, feats, reward, active, time_cost = envs.step(action) # 58% of step1 (train trial)
        score += reward
        act_sum += active.sum().item()

        # 3. record 0%
        if infer:
            prev_active, step_done, mask_ind, tp_ind, elig_ind = envs.get_delayed_indexed_states() # only for inference
            ilp.insert(prev_active, step_done, tp_ind, elig_ind, action, reward)
            reward = ilp.compute_bonus(prev_active, step_done, tp_ind, elig_ind)

        if active.sum(0).item() == 0:
            break

    score /= nb_epi
    act_sum /= nb_epi * args.num_processes
    return score, act_sum, ilp


if __name__ == "__main__":
    args = get_args()
    main(args)
