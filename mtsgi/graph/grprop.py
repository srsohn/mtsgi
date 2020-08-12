import os
import numpy as np
from mtsgi.graph.batch_teacher import Teacher


class GRProp(object):
    """The GRProp policy."""

    def __init__(self, graphs, args):
        self.algo = 'grprop'
        self.graphs = graphs
        self.args = args
        self.teacher =  Teacher( graphs, args )

    """def act(self, active, mask_ind, tp_ind, elig_ind):
        action_id = []
        for i in range(self.Nbatch):
            if active[i]==0:
                action_id.append(-1)
            else:
                state = (None, None, None, None, mask_ind[i], tp_ind[i], elig_ind[i])
                tid = self.teacher[i].choose_action(state)
                action_id.append(tid)

        return action_id"""

    def act(self, active, mask_ind, tp_ind, elig_ind, eval_flag=False, p=False):
        assert(eval_flag)
        state = (None, None, None, None, mask_ind, tp_ind, elig_ind)
        a = self.teacher.choose_action( state, eval_flag, p )

        return np.expand_dims(a, -1)
