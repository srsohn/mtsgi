import os, time, csv, copy
from mtsgi.graph.graph_utils import dotdict
from tensorboardX import SummaryWriter
import numpy as np


def _update_print_log(args, logs, it, num_updates, start, Score_tr, Score_tst, succ_tr, succ_tst, rollouts):
    #1.
    s_tr    = Score_tr.cpu().mean().item()
    if type(Score_tst)==int:
        s_tst = Score_tst
    else:
        s_tst   = Score_tst.cpu().mean().item()
    Val     = rollouts.value_preds[0].cpu().mean().item()
    Return  = rollouts.rewards.cpu().sum(0).mean().item()
    #print('ret=', Return, 'rsum=',rollouts.rewards.cpu().sum(0).mean().item())
    total_inter = succ_tr*args.num_processes*args.tr_epi

    #2. update windows
    _update_window(logs.window.str, s_tr)
    _update_window(logs.window.stst, s_tst)
    _update_window(logs.window.val, Val)
    _update_window(logs.window.ret, Return)

    #3. update log
    logs.iter   = it
    logs.tot_iter = num_updates
    logs.inter_count = (logs.inter_count or 0) + total_inter
    logs.s_tr, logs.s_tst       = s_tr, s_tst
    logs.value, logs.ret        = Val, Return
    logs.succ_tr, logs.succ_tst = succ_tr, succ_tst

    #4. print
    if it % args.summary_interval==0 and it>1:
        elapsed_time = time.time() - start
        rem_time = (num_updates - it ) / it *elapsed_time
        fps = int(logs.inter_count / elapsed_time)
        fmt = "[{:-3d}/{}] T={:3d}K| Str={:1.2f}| Stst={:1.2f}| R={:1.02f}| V(s)={:1.2f}| Suc={:.1f}| Fps={}| Elp={:3.01f}| Rem={:3.01f}"
        print(fmt.format(it, num_updates, round(logs.inter_count/1000),
                        logs.window.str.mean(), logs.window.stst.mean(),
                        logs.window.ret.mean(), logs.window.val.mean(), logs.succ_tst,
                        fps, elapsed_time/60.0, rem_time/60.0 ) )
    return logs

def _print_eval_log(args, logs, it, tot_it, start ):
    if it % args.summary_interval==0:
        elapsed_time = time.time() - start
        rem_time = (tot_it - it ) / it *elapsed_time
        if args.mode == 'meta_eval':
            fmt = "[{:-3d}/{}]: Elp={:3.01f}| Rem={:3.01f}"
            print(fmt.format(it, tot_it, elapsed_time/60.0, rem_time/60.0))
            print('Stst=', logs.stst/it)
            print('Str=', logs.str/it)
        else:
            fmt = "[{:-3d}/{}]: Elp={:3.01f}| Rem={:3.01f} | Stst={:1.2f}"
            print(fmt.format(it, tot_it, elapsed_time/60.0, rem_time/60.0, logs.stst/it ) )

def tensorboard_eval(args, logs, it, ep_logs=None, g_ind=None, value_loss=None, action_loss=None, dist_entropy=None, tot_it=None):
    if args.save:
        if args.writer is None:
            args.writer = SummaryWriter(log_dir=args.run_path)

        if args.mode == 'meta_eval':
            # cum_ep : current episode * num_graph --> cumulative episode
            cum_ep = g_ind*args.tr_epi + it if g_ind > 0 else it

            # trial scores & losses
            if (it + 1) % args.tr_epi == 0:
                s_tr = logs.str/(g_ind + 1)
                s_tst = logs.stst/(g_ind + 1)
                args.writer.add_scalar('performance/score_train_per_trial', s_tr[-1], g_ind)
                args.writer.add_scalar('performance/score_test_per_trial', s_tst[-1], g_ind)

            # episode scores & losses
            args.writer.add_scalar('performance/score_train_per_episode', ep_logs.str[it], cum_ep)
            args.writer.add_scalar('performance/score_test_per_episode', ep_logs.stst[it], cum_ep)
            args.writer.add_scalar('eval_loss/ep_total', value_loss + action_loss - dist_entropy, cum_ep)
            args.writer.add_scalar('eval_loss/ep_policy', action_loss, cum_ep)
            args.writer.add_scalar('eval_loss/ep_value', value_loss, cum_ep)
            args.writer.add_scalar('eval_loss/ep_entropy', -dist_entropy, cum_ep)
        else:  # random /grprop
            score_test = logs.stst/it
            args.writer.add_scalar('performance/rollout_score_tests', score_test, it)

def _save_log(args, logs, agent, value_loss, action_loss, dist_entropy):
    if args.save:
        n_iter = logs.iter
        num_updates = logs.tot_iter

        #1. save tensorboard summary
        if n_iter % args.summary_interval==0 and n_iter>1:
            if args.writer is None:
                args.writer = SummaryWriter(log_dir=args.run_path)
            _write_summary(logs, args.writer, agent, value_loss, action_loss, dist_entropy)

        #2. write csv
        if n_iter==1:
            with open(logs.csv_filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow([n_iter, logs.s_tr, logs.s_tst, logs.ret, logs.succ_rate_tst])
        else:
            with open(logs.csv_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([n_iter, logs.s_tr, logs.s_tst, logs.ret, logs.succ_rate_tst])

        #2. save for every interval-th episode or for the last epoch
        if args.save_flag and (n_iter % args.save_interval == 0 or n_iter == num_updates) and n_iter > 1:
            save_path = os.path.join('trained_model', args.folder_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            # A really ugly way to save a model to CPU
            save_model = agent.actor_critic
            if args.cuda:
                save_model = copy.deepcopy(agent.actor_critic).cpu()
            #torch.save(save_model, os.path.join(save_path, args.env_name + '_' + str(logs.iter) + ".pt"))
            raise NotImplementedError("Need to use other checkpointing mechanism.")

def _save_eval_log(args, logs, tot_it, value_loss=None, action_loss=None, dist_entropy=None):
    if args.save:
        if args.writer is None:
            writer = SummaryWriter(log_dir=args.run_path)
        else:
            writer = args.writer

        if args.mode=='meta_eval':
            s_tr    = logs.str/tot_it
            s_tst   = logs.stst/tot_it
            s_tst   = logs.stst/tot_it
            ret     = logs.ret/tot_it
            succ_rate = logs.succ_rate_tst/tot_it
            data = np.stack([np.arange(1,args.tr_epi+1), s_tr, s_tst, ret, succ_rate])
            for n_iter in range(0, args.num_updates, args.summary_interval):
                writer.add_scalar('performance/final_score_train', s_tr[-1], n_iter)
                writer.add_scalar('performance/final_score_test', s_tst[-1], n_iter)

            for n_iter in range(0, args.tr_epi):
                v, a, d = value_loss[n_iter].item()/tot_it, action_loss[n_iter].item()/tot_it, dist_entropy[n_iter].item()/tot_it
                writer.add_scalar('eval_loss/total', v+a-d, n_iter)
                writer.add_scalar('eval_loss/policy', a, n_iter)
                writer.add_scalar('eval_loss/value', v, n_iter)
                writer.add_scalar('eval_loss/entropy', -d, n_iter)
        else: # random / grprop
            s_tst   = logs.stst/tot_it
            succ_rate = logs.succ_rate_tst/tot_it
            data = np.array([[0, 0, s_tst, 0, succ_rate]]).T
            for n_iter in range(0, args.num_updates, args.summary_interval):
                writer.add_scalar('performance/avg_score_test', s_tst, n_iter)

        #2. write csv
        np.savetxt(logs.csv_filename, data.T, delimiter=",")

def _write_summary(logs, writer, agent, value_loss, action_loss, dist_entropy):
    n_iter = logs.iter
    writer.add_scalar('loss/total', value_loss+action_loss-dist_entropy*agent.entropy_coef, n_iter)
    writer.add_scalar('loss/policy', action_loss, n_iter)
    writer.add_scalar('loss/value', value_loss, n_iter)
    writer.add_scalar('loss/entropy', -dist_entropy*agent.entropy_coef, n_iter)
    #import ipdb; ipdb.set_trace()
    if agent.algo=='a2c':
        writer.add_scalar('optim/lr', agent.optimizer.param_groups[0]['lr'], n_iter)
        writer.add_scalar('optim/rho', agent.entropy_coef, n_iter)
        writer.add_scalar('optim/entropy', dist_entropy, n_iter)
    else:
        writer.add_scalar('optim/lr', 0, n_iter)
        writer.add_scalar('optim/rho', 0, n_iter)
        writer.add_scalar('optim/entropy', 0, n_iter)
    writer.add_scalar('performance/score_train', logs.s_tr, n_iter)
    writer.add_scalar('performance/score_test', logs.s_tst, n_iter)
    writer.add_scalar('performance/return', logs.ret, n_iter)
    writer.add_scalar('performance/init_value', logs.value, n_iter)
    writer.add_scalar('performance/Success_test', logs.succ_tst, n_iter)

def prepare_logging(args):
    # 1. naming
    args.writer = None
    args.folder_name = args.method + '_'

    if args.env_name=='mining':
        args.folder_name = 'Mine_' + args.folder_name
    if args.env_name=='playground':
        args.suffix+= 'lv' +str(args.level) +'_'
    if args.env_name == 'thor':
        args.folder_name = 'Thor_' + args.folder_name

    if args.method=='SGI' and args.algo in ['a2c', 'ppo']: # SGI-meta / SGI-meta-eval
        if args.bonus_mode==0:
            args.suffix = '_ext_'
        elif args.bonus_mode==1:
            args.suffix = '_uniform_'
        elif args.bonus_mode==2:
            args.suffix = '_UCB_'
        else:
            assert(False)
    else:
        args.suffix = '_'

    config = 'tr_epi-{}_test_epi-{}_lr-{}_'.format(args.tr_epi, args.test_epi, args.lr)
    args.suffix += config

    if args.mode =='meta_eval':
        args.folder_name += 'eval_'
    args.folder_name += args.algo + '_env-{}'.format(args.env_name) + args.suffix + 'exp_id-'+str(args.exp_id) + '_seed-'+str(args.seed)

    #1 tensorboard dir
    #import socket
    #from datetime import datetime
    #current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    #args.run_path = os.path.join('runs', args.folder_name + '_' + current_time + '_' + socket.gethostname())
    args.run_path = os.path.join('runs', args.folder_name)
    logs = dotdict()

    #2 print summary
    window = dotdict()
    window.val, window.ret, window.str, window.stst = np.zeros((4, args.summary_interval))
    logs.window = window

    #3 log
    save_path = os.path.join('logs', args.folder_name)

    if args.save:
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        if args.mode=='meta_train':
            logs.csv_filename = os.path.join('logs', args.folder_name, 'log.csv')
        else:
            logs.csv_filename = os.path.join('logs', args.folder_name, 'eval_log.csv')
    return logs

def _anneal_param(vst, ved, t, tot, tst, ted):
    progress = t / tot
    clamped_progress = min(max( (progress - tst) / (ted-tst), 0.0), 1.0)
    return vst + (ved - vst) * clamped_progress

def _update_window(window, new_value):
    if len(window.shape)==1:
        window[:-1] = window[1:]
        window[-1] = new_value
    else:
        window[:,:-1] = window[:,1:]
        window[:,-1] = new_value
    return window

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None
