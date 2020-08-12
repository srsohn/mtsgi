import argparse
import sys


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--load_dir', default='',
                        help='directory of trained model to load')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index (default: 0)')
    parser.add_argument('--exp-id', type=int, default=0,
                        help='experiment id (default: 0)')
    parser.add_argument('--algo', default='random',
                        help='algorithm to use: random | a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='learning rate (default: 2e-3)')
    parser.add_argument('--lr_decay', type=float, default=1.0,
                        help='learning rate decay (default: 1.0)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='discount factor for rewards (default: 1.0)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--rho-v-st', type=float, default=0.05,
                        help='initial entropy term coefficient (default: 0.05)')
    parser.add_argument('--rho-v-ed', type=float, default=0,
                        help='final entropy term coefficient (default: 0)')
    parser.add_argument('--rho-t-st', type=float, default=0.15,
                        help='entropy term coefficient annealing start step (default: 0.15)')
    parser.add_argument('--rho-t-ed', type=float, default=0.4,
                        help='entropy term coefficient annealing finish step (default: 0.4)')
    parser.add_argument('--v-coef', type=float, default=0.03,
                        help='value loss coefficient (default: 0.03)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='max norm of gradients (default: 1.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--nworker', type=int, default=24,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='whether we save logs or not.')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-updates', type=int, default=5000,
                        help='number of episodes to train (default: 1e6)')
    parser.add_argument('--env-name', default='playground',
                        help='environment to train on (default: playground)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=True,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--pudb', action='store_true', default=False,
                        help='Run pudb with the process.')
    parser.add_argument('--quality', type=str, default='vl', choices=['vl', 'l', 'u'],
                        help='Run pudb with the process.')

    ### Thor setting
    parser.add_argument('--screen_size', type=int, default=300, help='Width & Height size of screen')

    ### Mazebase setting
    parser.add_argument('--level', type=int, default=1, help='difficulty of subtask graph')
    parser.add_argument('--ntasks', type=int, default=13, help='number of subtasks')
    parser.add_argument('--max_step', type=int, default=60, help='episode length')
    parser.add_argument('--render', action='store_true', default=False, help="whether rendering")
    parser.add_argument('--random_step', default=False, action='store_true', help='whether using random_step')
    parser.add_argument('--verbose_level', type=int, default=0, help='debug message level')
    parser.add_argument('--debug', action='store_true', default=False, help='debug message level')

    ### Meta setting
    parser.add_argument('--mode', default='meta_eval', help='exp type: meta_train|meta_eval|eval')
    parser.add_argument('--method', default='baseline', help='method name: SGI|rlrl|baseline')
    parser.add_argument('--tr_epi', default=10, type=int, help='the number of episode per training trial')
    parser.add_argument('--test_epi', default=4, type=int, help='the number of episode per testing trial')
    parser.add_argument('--load_epi', default=10, type=int, help='the number of episode per training trial of experiment being loaded from')
    parser.add_argument('--infer', action='store_true', default=True, help="whether doing graph inference")
    parser.add_argument('--train', action='store_true', default=False, help="whether updating network")
    parser.add_argument('--bonus', default=0, type=int, help='bonus mode')
    parser.add_argument('--graph-visualize', action='store_true', default=False, help='visualize the inferred graph.')

    ### Baseline setting
    parser.add_argument('--num_steps', default=4, type=int, help='number of bootstrapping step')

    ### model param
    parser.add_argument('--flat_ldim', default=512, type=int, help='flat module embedding dimension')
    parser.add_argument('--gru_ldim', default=512, type=int, help='cnn module embedding dimension')

    ### GRProp param
    parser.add_argument('--fast_teacher_mode', default='RProp', help='fast teacher_mode')
    parser.add_argument('--determ_teacher', default=False, action='store_true', help='deterministic teacher policy')
    parser.add_argument('--temp', default=40.0, type=float, help='RProp param')
    parser.add_argument('--beta_a', default=3.0, type=float, help='RProp param')
    parser.add_argument('--w_a', default=2.0, type=float, help='RProp param')
    parser.add_argument('--ep_or', default=0.6, type=float, help='RProp param')
    parser.add_argument('--temp_or', default=2.0, type=float, help='RProp param')

    ### MAML param
    parser.add_argument('--outer_lr', type=float, default=2e-3,
                        help='learning rate (default: 2e-3)')
    parser.add_argument('--outer_lr_mult', type=float, default=0.999,
                        help='learning rate decay (default: 0.999)')


    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args.num_processes = args.nworker
    args.bonus_mode = args.bonus
    return args
