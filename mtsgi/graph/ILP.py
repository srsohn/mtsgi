import os
import numpy as np
import math

import mtsgi.environment
from mtsgi.graph.graph_utils import SubtaskGraph, dotdict, batch_bin_encode, _to_multi_hot, _transform


class ILP(object):

    def __init__(self, args):
        # these are shared among batch
        self.verbose, self.nenvs = args.verbose_level, args.num_processes
        self.action_dim = args.act_dim
        self.tr_epi = args.tr_epi
        self.bonus_mode = args.bonus_mode
        self.visualize = args.graph_visualize

        # visualization
        if self.visualize:
            self._env_name = args.env_name
            fname = f'graph_env-{args.env_name}_tr_epi-{args.tr_epi}_test_epi-{args.test_epi}_exp_id-{args.exp_id}'
            self.dirname = os.path.join(os.getcwd(), 'visualize', fname)
            if not os.path.isdir(self.dirname):
                os.makedirs(self.dirname)

        # rollout.
        self.step = 0

    def reset(self, envs):
        self.ntasks = envs.ntasks
        self.step = 0

        self.tp_inds = np.zeros(((self.ntasks + 1) * self.tr_epi * 2, self.nenvs, self.ntasks), dtype=np.bool)
        self.elig_inds = np.zeros(((self.ntasks + 1) * self.tr_epi * 2, self.nenvs, self.ntasks), dtype=np.bool)
        self.reward_sum = np.zeros((self.nenvs, self.ntasks))
        self.reward_count = np.zeros((self.nenvs, self.ntasks), dtype=np.long)

        if self.bonus_mode > 0:
            self.hash_table = [set() for i in range(self.nenvs)]
            self.base_reward = min(10.0 / self.tr_epi / self.ntasks, 1.0)
            if self.bonus_mode == 2:
                self.pn_count = np.zeros( ( self.nenvs, self.ntasks, 2) )

        # reset from envs
        self.tid_to_tind = envs.get_tid_to_tind()
        self.tlists = envs.get_subtask_lists()

    def insert(self, prev_active, step_done, tp_ind, elig_ind,
               action_id=None, reward=None):
        self.tp_inds[self.step] = tp_ind.copy()
        self.elig_inds[self.step] = elig_ind.copy()

        # reward
        if not (reward is None):
            active = np.expand_dims(
                prev_active * (1 - step_done).astype(dtype=np.int32), -1)
            assert active.ndim == 2 and active.shape[-1] == 1
            act_inds = self._get_ind_from_id(action_id) * active

            mask = _to_multi_hot(act_inds, self.ntasks).astype(dtype=np.int32) * active
            reward_mat = np.zeros_like(self.reward_sum)
            np.put_along_axis(reward_mat, act_inds, reward, axis=1)
            self.reward_sum += reward_mat * (mask.astype(np.float))
            self.reward_count += mask
        self.step += 1

    def compute_bonus(self, prev_active, step_done, tp_ind, elig_ind):
        batch_size = len(tp_ind)
        rewards = np.zeros(batch_size)
        if self.bonus_mode == 0:
            pass
        elif self.bonus_mode == 1: # novel tp_ind
            tp_code = batch_bin_encode(tp_ind)
            for i in range(batch_size):
                if prev_active[i]==1 and step_done[i]==0:
                    code = tp_code[i].item()
                    if not code in self.hash_table[i]:
                        rewards[i] = self.base_reward
                        self.hash_table[i].add(code)
        elif self.bonus_mode == 2: # novel tp_ind & UCB weight ( branch: pos/neg )
            tp_code = batch_bin_encode(tp_ind)
            for i in range(batch_size):
                if prev_active[i] == 1 and step_done[i] == 0:
                    elig = elig_ind[i]
                    pn_count = self.pn_count[i] #referencing
                    num_task = elig.shape[0]
                    code = tp_code[i].item()
                    if code not in self.hash_table[i]:
                        self.hash_table[i].add(code)

                        # 1. add shifted-ones
                        shifted_ones = np.zeros_like(pn_count)
                        np.put_along_axis(shifted_ones, np.expand_dims(elig, 1).astype(dtype=np.int32), 1, axis=1)
                        pn_count += shifted_ones

                        # 2. compute weight
                        N_all = pn_count.sum(1)
                        n_current = pn_count.gather(dim=1, index=np.expand_dims(elig.astype(dtype=np.int32), 1)).squeeze()
                        #UCB_weight = (np.log(N_all)/n_current).sqrt().sum() / num_task
                        UCB_weight = (25/n_current).sqrt().sum() / num_task   # version 2 (removed log(N_all) since it will be the same for all subtasks)
                        rewards[i] = self.base_reward * UCB_weight
                        """ # slower version
                        reward_test = 0.
                        for ind in range(num_task):
                            pn_ind = elig[ind].item() # 0: neg, 1: pos
                            pn_count[ind][pn_ind] += 1
                            N_all = pn_count[ind].sum()
                            n_current = pn_count[ind][pn_ind]
                            UCB_weight = math.sqrt(math.log(N_all)/n_current) / num_task
                            reward_test += self.base_reward * UCB_weight"""

        elif self.bonus_mode == 3:
            # update in hash table. (if the value of updated element is different from the previous (presumed-dont-care) value, give reward)
            # TODO: update k-map every time (is too slow...)
            pass

        return np.expand_dims(rewards, 1)

    def infer_graph(self, ep=None, g_ind=None):
        self._ep = ep
        self._g_ind = g_ind

        batch_size = self.nenvs
        num_step = self.step
        tp_tensors      = self.tp_inds[:num_step,:,:]              # T x batch x 13
        elig_tensors    = self.elig_inds[:num_step,:,:]            # T x batch x 13
        rew_counts      = self.reward_count             # batch x 13
        rew_tensors     = self.reward_sum               # batch x 13
        tlists          = self.tlists

        graphs = []
        for i in range(batch_size):
            tp_tensor   = tp_tensors[:,i,:]     # T x Ntask
            elig_tensor = elig_tensors[:,i,:]   # T x Ntask
            rew_count  = rew_counts[i]          # Ntask
            rew_tensor  = rew_tensors[i]        # Ntask
            tlist = tlists[i]
            graph = self._init_graph(tlist) #0. initialize graph

            #1. update subtask reward
            graph.rmag = self._infer_reward(rew_count, rew_tensor) #mean-reward-tracking

            #2. find the correct layer for each node
            is_not_flat, subtask_layer, tind_by_layer, tind_list = self._locate_node_layer( tp_tensor, elig_tensor ) #20%
            if is_not_flat:
                #3. precondition of upper layers
                Kmap_set = self._infer_precondition(tp_tensor, elig_tensor, subtask_layer, tind_by_layer) #75%

                # XXX visualize
                if self.visualize and self._ep % 10 == 9:
                    self._visualize_graph(Kmap_set, subtask_layer)
                    #self._visualize_small_graph(Kmap_set, subtask_layer)

                #4. fill-out params
                graph.W_a, graph.W_o, graph.ANDmat, graph.ORmat = self._fillout_mat(Kmap_set, tind_by_layer, tind_list)
            else:
                graph.W_a, graph.W_o, graph.ANDmat, graph.ORmat = [], [], np.zeros( (0) ), np.zeros( (0) )

            #5. fill-out other params.
            graph.tind_by_layer = tind_by_layer
            graph.tind_list = tind_list # doesn't include unknown subtasks (i.e., never executed)
            graph.numP, graph.numA = [], []
            for i in range( len(tind_by_layer) ):
                graph.numP.append( len(tind_by_layer[i]) )
            for i in range( len(graph.W_a) ):
                graph.numA.append( graph.W_a[i].shape[0] )
            graphs.append(graph)

        return graphs

    def _init_graph(self, t_list):
        graph = SubtaskGraph()
        graph.numP, graph.numA = [self.ntasks], []
        self.tid_list = t_list
        graph.ind_to_id = np.zeros( (self.ntasks) ).astype(dtype=np.int32)
        graph.id_to_ind = np.zeros( (self.action_dim) ).astype(dtype=np.int32)
        for tind in range(self.ntasks):
            tid = t_list[tind]
            graph.ind_to_id[tind] = tid
            graph.id_to_ind[tid] = tind
        return graph

    def _locate_node_layer(self, tp_tensor, elig_tensor):
        debug = False
        tind_list = []
        subtask_layer = np.ones(self.ntasks, dtype=np.int) * (-1)
        #1. update subtask_layer / tind_by_layer
        #1-0. masking
        p_count = elig_tensor.sum(0)
        comp_count = tp_tensor.sum(0)
        num_update = tp_tensor.shape[0]
        first_layer_mask = (p_count==num_update)
        never_elig_mask = (p_count==0)
        never_comp_mask = comp_count==0
        always_comp_mask = comp_count==num_update
        #print('putdown tomato:', PUTDOWN_DEFAULT_OBJ_TO_IDS['Tomato'])
        #print(comp_count[len(PICKUP_SUBTASKS)+PUTDOWN_DEFAULT_OBJ_TO_IDS['Tomato']])
        ignore_mask = (never_elig_mask & (never_comp_mask | always_comp_mask))
        extra_first_layer_mask = (never_elig_mask & ~(never_comp_mask | always_comp_mask))
        num_first = first_layer_mask.sum().item() + extra_first_layer_mask.sum().item()
        num_ignored_subtasks = ignore_mask.sum().item()
        infer_flag = (num_first + num_ignored_subtasks < self.ntasks)
        #print('%d/%d subtasks are in the first layer'%(num_first, self.ntasks))
        #print('%d/%d subtasks will be ignored'%(num_ignored_subtasks, self.ntasks))

        #1-1. first layer & unknown (i.e. never executed).
        subtask_layer[first_layer_mask] = 0
        subtask_layer[extra_first_layer_mask] = 0
        subtask_layer[ignore_mask] = -2

        #
        cand_ind_list, cur_layer = [], []
        remaining_tind_list = []
        for tind in range(self.ntasks):
            if first_layer_mask[tind] == 1 or extra_first_layer_mask[tind] == 1: # first layer subtasks.
                cand_ind_list.append(tind)
                cur_layer.append(tind)
            elif ignore_mask[tind]==1: # if never executed, assign first layer, but not in cand_list.
                cur_layer.append(tind)
            else:
                remaining_tind_list.append(tind)
        tind_by_layer = [ cur_layer ]
        tind_list += cur_layer # add first layer
        for layer_ind in range(1,self.ntasks):
            cur_layer = []
            for tind in range(self.ntasks):
                if subtask_layer[tind] == -1: # among remaining tind
                    inputs = tp_tensor[:, cand_ind_list]
                    targets = elig_tensor[:, tind]

                    assert inputs.ndim == 2 and targets.ndim == 1
                    #inputs = tp_tensor.index_select(1, np.longTensor(cand_ind_list) ) #nstep x #cand_ind
                    #targets = elig_tensor.index_select(1, np.longTensor([tind]) ).view(-1) #nstep
                    is_valid = self._check_validity(inputs, targets, debug)
                    if is_valid: # add to cur layer
                        subtask_layer[tind] = layer_ind
                        cur_layer.append(tind)

            if len(cur_layer)>0:
                tind_by_layer.append(cur_layer)
                tind_list += cur_layer
                cand_ind_list = cand_ind_list + cur_layer
            else: # no subtask left.
                nb_subtasks_left = (subtask_layer==-1).astype(dtype=np.int32).sum().item()
                if nb_subtasks_left > 0:
                    assert False, 'Error! There is a bug in the precondition!'
                    #print('1. Possible subtasks that has problematic preconditions:')
                    #problematic_subtasks = []
                    #for i, layer in enumerate(subtask_layer):
                    #    if layer == -1:
                    #        problematic_subtasks.append(i)
                    #        print(SUBTASKS[i]['name'])
                    #print('2. Candidate subtasks:')
                    #for tind in cand_ind_list:
                    #    print(SUBTASKS[tind]['name'])
                    #print('Eligibility:')

                    ## 3. target
                    #sub_id = problematic_subtasks[0]
                    #targets = elig_tensor.index_select(1, np.longTensor([sub_id]) ).view(-1) #nstep
                    #print('Elig:', targets)
                    #print('Comps:')
                    #print(inputs)
                    #for i, subtask in enumerate(SUBTASKS):
                    #    print(i, subtask['name'])
                assert nb_subtasks_left == 0
                break

        # result:
        if self.verbose>0:
            print('subtask_layer:', subtask_layer)
            print('tind_by_layer:', tind_by_layer)
        return infer_flag, subtask_layer, tind_by_layer, tind_list

    def _check_validity(self, inputs, targets, debug=False):
        # check if there exists any i1 and i2 such that inputs[i1]==inputs[i2] and targets[i1]!=targets[i2]
        # if there exists, it means the node is not valid, and it should be in the higher layer in the graph.
        tb = {}
        nstep = inputs.shape[0] #new
        code_batch = batch_bin_encode(inputs)
        for i in range(nstep):
            code = code_batch[i]
            target = targets[i].item()
            if code in tb:
                if tb[code]!=target:
                    # if debug: breakpoint()
                    return False
            else:
                tb[code] = target

        return True

    def _infer_reward(self, reward_count, reward_tensor): #. mean-reward
        # reward_count: Ntasks
        # reward_tensor: Ntasks
        rmean = (reward_tensor.sum() / reward_count.sum()).item()
        rmag = np.ones(self.ntasks) * rmean
        for tind  in range(self.ntasks):
            count = reward_count[tind].item()
            if count > 0:
                rmag[tind] = reward_tensor[tind] / count
        if self.verbose:
            print('rmag=')
            print(rmag)
        return rmag

    def _infer_precondition(self, tp_tensor, elig_tensor, subtask_layer, tind_by_layer):
        ever_elig = elig_tensor.sum(0)
        #'subtask_layer' is already filled out in 'update()'
        Kmap_set, cand_ind_list = [None] * self.ntasks, []
        max_layer = subtask_layer.max()

        if self.verbose:
            print('ever_elig=',ever_elig)

        for layer_ind in range(1,max_layer+1):
            cand_ind_list = cand_ind_list + tind_by_layer[layer_ind -
                                                          1]  # previous layers
            nFeature = len(cand_ind_list)
            for ind in range(self.ntasks):
                if subtask_layer[ind] == layer_ind:
                    if ever_elig[ind] > 0:
                        #print('ind=',ind)

                        inputs = tp_tensor[:, cand_ind_list]
                        targets = elig_tensor[:, ind]

                        mask = np.ones(nFeature, dtype=np.int)
                        #print('inputs=',inputs)
                        #print('targets=',targets)
                        root = self.cart_train(mask, inputs, targets) #1.8
                        Kmap_tensor = self.decode_cart(root, nFeature) #0.08
                        Kmap_tensor = self.simplify_Kmap(Kmap_tensor) #0.12
                        assert Kmap_tensor.ndim == 2
                        Kmap_set[ind] = Kmap_tensor
                        #
                    else:
                        print('ind=', ind)
                        print('ever_elig=',ever_elig)
                        assert False

        return Kmap_set

    def decode_cart(self, root, nFeature):
        Kmap = []
        stack = []
        instance = dotdict()
        instance.lexp = np.zeros( (nFeature), dtype=np.int8)
        instance.node = root
        stack.append(instance)
        while len(stack)>0:
            node = stack[0].node
            lexp = stack[0].lexp
            stack.pop(0)
            featId = node.best_ind
            if node.gini>0 : #leaf node && positive sample
                assert(featId>=0)
                if node.left.best_ind>=0: #negation
                    instance = dotdict()
                    instance.lexp = lexp.copy()
                    instance.lexp[featId] = -1 # negative
                    instance.node = node.left
                    stack.append(instance)

                if node.right.best_ind>=0: #positive
                    instance = dotdict()
                    instance.lexp = lexp.copy()
                    instance.lexp[featId] = 1 # positive
                    instance.node = node.right
                    stack.append(instance)

            elif node.sign==0:
                lexp[featId] = -1
                Kmap.append(lexp[None,:])
            else:
                lexp[featId] = 1
                Kmap.append(lexp[None,:])
        Kmap_tensor = np.concatenate(Kmap, axis=0)

        return Kmap_tensor

    def cart_train(self, mask, inputs, targets):
        assert inputs.ndim == 2
        nstep, ncand = inputs.shape
        root = dotdict()
        minval = 2.5  #range: 0~2.0
        assert (mask.sum() > 0)
        for i in range(ncand - 1, -1, -1):
            if mask[i] > 0:
                left, right = self.compute_gini(inputs[:,i], targets)
                gini = left.gini + right.gini
                if minval > gini:
                    minval = gini
                    best_ind = i
                    best_left = left
                    best_right = right

        root.best_ind = best_ind
        root.gini = minval
        #print('best_ind=',best_ind,best_left, best_right)
        mask[best_ind] = 0
        if minval>0:
            if best_left.gini>0:
                best_lind = np.nonzero(np.equal(inputs[:,best_ind], 0) )[0]
                left_input = inputs[best_lind]
                left_targets = targets[best_lind]
                root.left = self.cart_train(mask, left_input, left_targets)
            else:
                root.left = dotdict()
                root.left.gini = 0
                root.left.sign = best_left.p1
                root.left.best_ind = -1

            if best_right.gini>0:
                best_rind = np.nonzero(inputs[:,best_ind] )[0]
                right_input = inputs[best_rind]
                right_targets = targets[best_rind]
                root.right = self.cart_train(mask, right_input, right_targets)
            else:
                root.right = dotdict()
                root.right.gini = 0
                root.right.sign = best_right.p1
                root.right.best_ind = -1
        else:
            root.sign = best_right.p1 #if right is all True,: sign=1

        mask[best_ind] = 1
        return root

    def compute_gini(self, input_feat, targets):
        neg_input = ~input_feat
        neg_target = ~targets
        nn = (neg_input*neg_target).sum().item()   # count[0]
        np = (neg_input*targets).sum().item()      # count[1]
        pn = (input_feat*neg_target).sum().item()  # count[2]
        pp = (input_feat*targets).sum().item()     # count[3]
        assert(nn+np+pn+pp == input_feat.shape[0])

        left, right = dotdict(), dotdict()
        if nn+np>0:
            p0_left = nn / (nn+np)
            p1_left = np / (nn+np)
            left.gini = 1 - pow(p0_left, 2) - pow(p1_left, 2)
            left.p0 = p0_left
            left.p1 = p1_left
        else:
            left.gini = 1
            left.p0 = 1
            left.p1 = 1

        if pn + pp > 0:
            p0_right = pn / (pn + pp)
            p1_right = pp / (pn + pp)
            right.gini = 1 - pow(p0_right, 2) - pow(p1_right, 2)
            right.p0 = p0_right
            right.p1 = p1_right
        else:
            right.gini = 1
            right.p0 = 1
            right.p1 = 1

        return left, right

    def _fillout_mat(self, Kmap_set, tind_by_layer, tind_list):
        W_a, W_o, cand_tind = [], [], []
        num_prev_or = 0
        numA_all = 0
        #1. fillout W_a/W_o
        for layer_ind in range(1, len(tind_by_layer)):
            num_prev_or = num_prev_or + len(tind_by_layer[layer_ind-1])
            num_cur_or = len(tind_by_layer[layer_ind])
            W_a_layer, W_a_layer_padded = [], []
            cand_tind += tind_by_layer[layer_ind - 1]
            OR_table = [None] * self.ntasks
            numA = 0
            # fill out 'W_a_layer' and 'OR_table'
            for ind in tind_by_layer[layer_ind]:
                Kmap = Kmap_set[ind]
                if Kmap is None:
                    print('ind=', ind)
                    print('tind_by_layer', tind_by_layer[layer_ind] )
                    assert False, "Kmap should not be None"

                if len(Kmap)>0: #len(Kmap)==0 if no positive sample exists
                    OR_table[ind] = []
                    for j in range(Kmap.shape[0]):
                        ANDnode = Kmap[j,:].astype(np.float)
                        #see if duplicated
                        duplicated_flag = False
                        for row in range(numA):
                            if np.all( np.equal(W_a_layer[row],ANDnode) ):
                                duplicated_flag = True
                                and_node_index = row
                                break

                        if duplicated_flag==False:
                            W_a_layer.append(ANDnode)
                            cand_tind_tensor = np.array(cand_tind).astype(np.int32)
                            assert cand_tind_tensor.shape[0] == ANDnode.shape[0]

                            padded_ANDnode = np.zeros( (self.ntasks) )
                            np.put_along_axis(padded_ANDnode, cand_tind_tensor, ANDnode, axis=0)
                            #padded_ANDnode = np.zeros( (self.ntasks) ).scatter_(0, cand_tind_tensor, ANDnode)
                            W_a_layer_padded.append(padded_ANDnode)
                            OR_table[ind].append(numA) #add the last one
                            numA = numA + 1
                        else:
                            OR_table[ind].append(and_node_index) #add the AND node
            if numA>0:
                numA_all = numA_all + numA
                W_a_tensor = np.stack(W_a_layer_padded, axis=0)
                W_a.append(W_a_tensor)
            # fill out 'W_o_layer' from 'OR_table'
            W_o_layer = np.zeros( (self.ntasks, numA) )
            for ind in tind_by_layer[layer_ind]:
                OR_table_row = OR_table[ind]
                for j in range(len(OR_table_row)):
                    and_node_index = OR_table_row[j]
                    W_o_layer[ind][and_node_index] = 1

            W_o.append(W_o_layer)

        if self.verbose:
            print('W_a')
            for i in range(len(W_a)): print(W_a[i])
            print('W_o')
            for i in range(len(W_o)): print(W_o[i])

        #2. fillout ANDmat/ORmat
        assert len(W_a) > 0
        ANDmat  = np.concatenate(W_a, axis=0)
        ORmat   = np.concatenate(W_o, axis=1)

        if numA_all == 0 or self.ntasks == 0:
            print('self.ntasks=', self.ntasks)
            print('numA_all=', numA_all)
            print('Kmap_set=', Kmap_set)
            print('tind_by_layer=', tind_by_layer)
            print('tind_list=', tind_list)
            assert False

        if self.verbose:
            print('Inference result:')
            print('ANDmat=', ANDmat)
            print('ORmat=', ORmat)

        return W_a, W_o, ANDmat, ORmat

    def simplify_Kmap(self, Kmap_tensor):
        """
        # This function performs the following two reductions
        # A + AB  -> A
        # A + A'B -> A + B
        ###
        # Kmap_bin: binarized Kmap. (i.e., +1 -> +1, 0 -> 0, -1 -> +1)
        """
        numAND = Kmap_tensor.shape[0]
        mask = np.ones(numAND)
        max_iter = 20
        for jj in range(max_iter):
            done = True
            remove_list = []
            Kmap_bin = np.not_equal(Kmap_tensor,0).astype(np.uint8)
            for i in range(numAND):
                if mask[i]==1:
                    kbin_i = Kmap_bin[i]
                    for j in range(i+1, numAND):
                        if mask[j]==1:
                            kbin_j = Kmap_bin[j]
                            kbin_mul = kbin_i * kbin_j
                            if np.all(kbin_mul == kbin_i): #i subsumes j. Either 1) remove j or 2) reduce j.
                                done = False
                                Kmap_common_j = Kmap_tensor[j] * kbin_i # common parts in j.
                                difference_tensor = Kmap_common_j != Kmap_tensor[i] # (A,~B)!=(A,B) -> 'B'
                                num_diff_bits = np.sum(difference_tensor)
                                if num_diff_bits==0: # completely subsumes--> remove j.
                                    mask[j]=0
                                else: #turn off the different bits
                                    dim_ind = np.nonzero(difference_tensor)[0]
                                    #print('dim_ind=',dim_ind)
                                    Kmap_tensor[j][dim_ind] = 0

                            elif np.all(kbin_mul == kbin_j): #j subsumes i. Either 1) remove i or 2) reduce i.
                                done = False
                                Kmap_common_i = Kmap_tensor[i] * kbin_j
                                difference_tensor = Kmap_common_i != Kmap_tensor[j]
                                num_diff_bits = np.sum(difference_tensor)
                                if num_diff_bits == 0:  # completely subsumes--> remove i.
                                    mask[i] = 0
                                else: #turn off the different bit.
                                    dim_ind = np.nonzero(difference_tensor)[0]
                                    #print('dim_ind=',dim_ind)
                                    Kmap_tensor[i][dim_ind] = 0

            if done: break

        if mask.sum()< numAND:
            Kmap_tensor = Kmap_tensor[mask.nonzero()[0],:]
            #Kmap_tensor = Kmap_tensor.index_select(0,mask.nonzero().view(-1))

        return Kmap_tensor

    ### Util
    def _get_ind_from_id(self, input_ids):
        if self.tid_to_tind is None:
            return input_ids
        else:
            return _transform(input_ids, self.tid_to_tind)

    def _visualize_graph(self, cond_kmap_set, subtask_layer):
        from graphviz import Digraph
        filename = os.path.join(self.dirname, f'subtask_graph-g_ind{self._g_ind}-ep{self._ep}-step{self.step}')
        g = Digraph(comment='subtask graph', format='pdf', filename=filename)
        g.graph_attr['rankdir'] = 'LR'
        g.attr(nodesep="0.2", ranksep="0.3")
        g.node_attr.update(fontsize="14", fontname='Arial')

        #cond_kmap_set
        for ind in range(self.action_dim):
            if subtask_layer[ind]>-2:
                LABEL_NAME = mtsgi.environment.get_subtask_label(self._env_name)
                label = LABEL_NAME[ind]
                g.node('OR'+str(ind),label, shape='rect',height="0.2",width="0.2", margin="0")
                Kmap_tensor = cond_kmap_set[ind]
                if Kmap_tensor is None:
                    continue
                numA, feat_dim = Kmap_tensor.shape
                for aind in range(numA):
                    anode_name = 'AND'+str(ind)+'_'+str(aind)
                    g.node(anode_name, "&", shape='rect', style='filled',
                           height="0.15", width="0.15", margin="0")
                    # OR-AND
                    g.edge(anode_name, 'OR'+str(ind) )
                    sub_indices = Kmap_tensor[aind, :].nonzero()[0]
                    for sub_ind in sub_indices:
                        sub_ind = sub_ind.item()
                        target = 'OR'+str(sub_ind)

                        if Kmap_tensor[aind, sub_ind]>0: #this looks wrong but it is correct since we used '>' instead of '<='.
                            # AND-OR
                            g.edge( target, anode_name  )
                        else:
                            g.edge( target, anode_name, style="dashed")

        g.render()
        print('Saved graph @', filename)

    def _visualize_small_graph(self, cond_kmap_set, subtask_layer):
        from graphviz import Digraph
        filename = os.path.join(self.dirname, f'subtask_graph_small-g_ind{self._g_ind}-ep{self._ep}-step{self.step}')
        g = Digraph(comment='subtask graph', format='pdf', filename=filename)
        g.graph_attr['rankdir'] = 'LR'
        g.attr(nodesep="0.1", ranksep="0.15")
        g.node_attr.update(fontsize="14", fontname='Arial')
        #cond_kmap_set
        target_indices = list(range(39))
        target_indices.remove(33)
        target_indices.remove(35)
        target_indices.remove(16)
        for ind in range(self.action_dim):
            if subtask_layer[ind] > -2 and ind in target_indices:
                LABEL_NAME = mtsgi.environment.get_subtask_label(self._env_name)
                label = LABEL_NAME[ind]
                g.node('OR'+str(ind),label, shape='rect',height="0.2",width="0.2", margin="0")
                Kmap_tensor = cond_kmap_set[ind]
                if Kmap_tensor is None:
                    continue
                numA, feat_dim = Kmap_tensor.shape
                for aind in range(numA):
                    sub_indices = Kmap_tensor[aind, :].nonzero()[0]
                    anode_name = 'AND'+str(ind)+'_'+str(aind)

                    count = self._count_children(sub_indices, target_indices)
                    if count>0:
                        g.node(anode_name, "&", shape='rect', style='filled',
                               height="0.15", width="0.15", margin="0")
                        # OR-AND
                        g.edge(anode_name, 'OR'+str(ind), arrowhead='ediamond')

                        for sub_ind in sub_indices:
                            sub_ind = sub_ind.item()
                            if sub_ind<39 and sub_ind in target_indices:
                                target = 'OR'+str(sub_ind)

                                if Kmap_tensor[aind, sub_ind]>0:
                                    # AND-OR
                                    g.edge(target, anode_name )
                                else:
                                    g.edge(target, anode_name, style="dashed")
        g.render()

    def _count_children(self,sub_indices, target_indices):
        count = 0
        for sub_ind in sub_indices:
            sub_ind = sub_ind.item()
            if sub_ind < 39 and sub_ind in target_indices:
                count += 1
        return count
