#from mtsgi.environment.thor.utils import SUBTASKS
import numpy as np

import tensorflow_probability as tfp
tfd = tfp.distributions


class Random():
    def __init__(self, args):
        self.algo = 'random'
        self.act_dim = args.act_dim

    def update(self, rollouts):
        return 0, 0, 0

    def act(self, actives, feats):
        feats = np.array(feats)
        # feats = [mask, tp, elig, time]. We use elig as policy-mask
        # TODO Remove this hack
        masks = feats[:, 2*self.act_dim:3*self.act_dim] * feats[:, :self.act_dim]
        masks[np.where(actives == 0)[0], :] = 1.0

        prob_logits = np.zeros_like(masks)
        prob_logits[masks == 0] = -np.inf

        action = tfd.Categorical(logits=prob_logits).sample()   # [B]

        #print("[  MASKS  ]")
        #for i, mask in enumerate(masks):
        #    print("Available options for agent {}".format(i))
        #    for j, subtask in enumerate(SUBTASKS):
        #        if mask[j]:
        #            print("  {}. {}".format(j, subtask['name']))
        #    print("---------------------"*2)
        #print("==========================="*2)

        return np.expand_dims(action, axis=1)  # [B, 1]
