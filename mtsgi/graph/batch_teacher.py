import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from mtsgi.graph.graph_utils import _to_multi_hot, _transform


class Teacher(object):

    def __init__(self, graphs, args, infer=False):
        if args.env_name == 'mining':
            self.temp = 200
            self.beta_a = 8
            self.w_a = 3
            self.ep_or = 0.8
            self.temp_or = 2
        else:
            self.temp = args.temp
            self.beta_a = args.beta_a
            self.w_a = args.w_a
            self.ep_or = args.ep_or
            self.temp_or = args.temp_or
        self.verbose = args.verbose_level > 0
        self.fast_teacher_mode = 'RProp'

        if isinstance(graphs, list):
            self.init_graph(graphs)   # graph from ILP
        else:
            raise NotImplementedError("Loading graph from a file is not supported.")


    def init_graph(self, graphs):
        ### initialize self.Wa_tensor, Wo_tensor, rew_tensor

        # prepare
        batch_size = len(graphs)
        self.num_layers = np.array([len(g.tind_by_layer) for g in graphs])
        self.max_num_layer = max([len(g.tind_by_layer) for g in graphs]) - 1
        max_NA = max([g.ANDmat.shape[0] for g in graphs])  #max total-#-A
        max_NP = max([len(g.rmag) for g in graphs])   #max total-#-P
        self.rew_tensor = np.zeros([batch_size, max_NP], dtype=np.float32)
        self.tind_to_tid = np.zeros([batch_size, max_NP], dtype=np.int32)

        for bind, graph in enumerate(graphs):
            self.rew_tensor[bind, :] = graph.rmag
            if isinstance(graph.ind_to_id, dict):
                for k, v in graph.ind_to_id.items():
                    self.tind_to_tid[bind, k] = v
            else:
                self.tind_to_tid[bind, :] = graph.ind_to_id

        if self.max_num_layer == 0:
            print('Warning!! flat graph!!!')
            self.fast_teacher_mode = 'GR'
        else:
            self.Wa_tensor  = np.zeros([self.max_num_layer, batch_size, max_NA, max_NP], dtype=np.float32)
            self.Wa_neg     = np.zeros([batch_size, max_NA, max_NP], dtype=np.float32)
            self.Wo_tensor  = np.zeros([self.max_num_layer, batch_size, max_NP, max_NA], dtype=np.float32)
            self.Pmask      = np.zeros([self.max_num_layer+1, batch_size, max_NP, 1], dtype=np.float32)

            for bind, graph in enumerate(graphs):
                tind_by_layer = graph.tind_by_layer
                num_layer = len(tind_by_layer) - 1
                #
                if isinstance(graph.rmag, list):
                    graph.rmag = np.array(graph.rmag)

                if num_layer > 0:
                    # W_a_neg
                    ANDmat  = graph.ANDmat
                    ORmat   = graph.ORmat
                    self.Wa_neg[bind, :ANDmat.shape[0], :ANDmat.shape[1]] = (ANDmat < 0).astype(np.float32) # only the negative entries
                    abias, tbias = 0, graph.numP[0]
                    tind_tensor = np.array(tind_by_layer[0], dtype=np.int32)
                    mask = np.zeros(max_NP)
                    mask[tind_tensor] = 1
                    self.Pmask[0, bind, :] = np.expand_dims(mask, axis=-1)

                    for lind in range(num_layer):
                        # W_a
                        na, _ = graph.W_a[lind].shape
                        wa = ANDmat[abias:abias+na, :]
                        # output is current layer only.
                        self.Wa_tensor[lind, bind, abias:abias+na, :] = wa

                        # W_o
                        tind = tind_by_layer[lind + 1]
                        wo = ORmat[:, abias:abias+na]    # numA x numP_cumul
                        nt, _ = graph.W_o[lind].shape

                        # re-arrange to the original subtask order
                        # input (or) is cumulative. output is current layer only.
                        self.Wo_tensor[lind, bind,:, abias:abias+na] = wo
                        abias += na

                        tind_tensor = np.array(tind, dtype=np.int32)
                        mask = np.zeros(max_NP)
                        mask[tind_tensor] = 1
                        self.Pmask[lind + 1, bind, :] = np.expand_dims(mask, axis=-1)
                        tbias += nt

            # only the positive entries
            self.Wa_tensor = (self.Wa_tensor > 0).astype(np.float32)

    @tf.function
    def compute_RProp_grad(self, tp, reward):
        """Computes the grprop gradient."""
        tp = tf.convert_to_tensor(tp, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(tp)
            r = self.RProp(tp, reward)
        return tape.gradient(r, tp)

    def RProp(self, tp, reward):
        # tp: Nb x NP x 1         : tp input. {0,1}
        # p: precondition progress [0~1]
        # a: output of AND node. I simply added the progress.
        # or: output of OR node.
        # p = softmax(a).*a (soft version of max function)
        # or = max (x, \lambda*p + (1-\lambda)*x). After execution (i.e. x=1), gradient should be blocked. So we use max(x, \cdot)
        # a^ = Wa^{+}*or / Na^{+} + 0.01   -       Wa^{-}*or.
        #     -> in [0~1]. prop to #satisfied precond.       --> If any neg is executed, becomes <0.
        # a = max( a^, 0 ) # If any neg is executed, gradient is blocked
        # Intuitively,
        # p: soft version of max function
        # or: \lambda*p + (1-\lambda)*x
        # a: prop to portion of satisfied precond
        #############

        # 1. forward (45% time)
        or_ = tf.maximum(tp, self.ep_or + (0.99 - self.ep_or) * tp) * self.Pmask[0]
        Wa_neg = tf.convert_to_tensor(self.Wa_neg, dtype=tf.float32)
        A_neg = tf.matmul(Wa_neg, tp)     #(Nb x NA x NP) * (Nb x NP x 1) = (Nb x NA x 1)
        or_list = [or_]
        for lind in range(self.max_num_layer):
            #Init
            wa = tf.convert_to_tensor(self.Wa_tensor[lind], dtype=tf.float32)
            wo = tf.convert_to_tensor(self.Wo_tensor[lind], dtype=tf.float32)
            Pmask = tf.convert_to_tensor(self.Pmask[lind + 1], dtype=tf.float32)

            or_concat = tf.concat(or_list, axis=2)

            #AND layer
            a_pos = tf.reduce_sum(tf.matmul(wa, or_concat), axis=-1) / (
                tf.maximum(tf.reduce_sum(wa, axis=-1), 1))
            a_pos = tf.expand_dims(a_pos, axis=-1)  # (Nb x NA x 1)
            a_hat = a_pos - self.w_a * A_neg                             #Nb x Na x 1

            #and_ = nn.Softplus(self.beta_a)(a_hat)                      #Nb x Na x 1 (element-wise)
            and_ = tf.nn.softplus(a_hat * self.beta_a) / self.beta_a

            #soft max version2
            num_next_or = wo.shape[1]
            and_rep = tf.tile(tf.transpose(and_, [0, 2, 1]), [1, num_next_or, 1])
            p_next = (self.masked_softmax(self.temp_or * and_rep, wo) * and_rep)
            p_next = tf.reduce_sum(p_next, axis=-1, keepdims=True)

            or_ = tf.maximum(tp, self.ep_or * p_next + (0.99 - self.ep_or) * tp) * Pmask  # Nb x Np_sum x 1
            or_list.append(or_)

        # loss (soft reward)  (should be scalar)
        or_mat = tf.concat(or_list, axis=2)
        soft_reward = tf.matmul(tf.transpose(or_mat, [0, 2, 1]), reward) # (Nb x Nt).*(Nb x Nt) (element-wise multiply)
        soft_reward = tf.reduce_sum(soft_reward)
        return soft_reward

    def masked_softmax(self, mat, mask, dim=2, epsilon=1e-6):
        nb, nr, nc = mat.shape
        masked_mat = mat * mask
        masked_min = tf.tile(tf.reduce_min(masked_mat, axis=dim, keepdims=True), [1, 1, nc])
        masked_nonneg_mat = (masked_mat - masked_min) * mask
        max_mat = tf.tile(tf.reduce_max(masked_nonneg_mat, axis=dim, keepdims=True), [1, 1, nc])
        exps = tf.exp(masked_nonneg_mat - max_mat)
        masked_exps = exps * mask
        masked_sums = tf.reduce_sum(masked_exps, axis=dim, keepdims=True) + epsilon
        prob = masked_exps / masked_sums
        tf.debugging.Assert(tf.reduce_all(prob >= 0), [prob])
        return prob

    def choose_action(self, state, eval_flag=False, p=False):
        _, _, _, _, mask, tp, elig = state
        mask_tensor = np.asarray(mask, dtype=np.float32)
        tp_tensor   = np.asarray(tp, dtype=np.float32)
        elig_tensor = np.asarray(elig, dtype=np.float32)
        self.ntasks = len(elig[0])
        batch_size = len(tp)
        tid = np.full(batch_size, -1, dtype=np.int32)

        # set RProp network
        if self.fast_teacher_mode == "RProp":
            # 1. prepare input
            # tp: task progression (= completion)
            x = np.expand_dims(tp_tensor, axis=-1)
            r = np.expand_dims(self.rew_tensor * mask_tensor, axis=-1)

            # 2. compute grad (48% time)
            grads = self.compute_RProp_grad(x, r)   # gradient is w.r.t. x
            if self.verbose:
                print('gradient=', grads)
            assert grads.shape == x.shape

            logits = grads.numpy()
            logits = tf.squeeze(logits, axis=-1)
            assert logits.shape == x.shape[:-1]

            # if flat, switch to greedy.
            logits += np.tile(
                (self.num_layers == 1).astype(np.float32).reshape(-1, 1),
                (1, self.ntasks)) * self.rew_tensor


        elif self.fast_teacher_mode == "GR":
            logits = self.rew_tensor

        else:
            raise RuntimeError("unknown teacher mode :" + self.fast_teacher_mode)

        # 3. masking (46% time)
        masked_elig_batch = np.multiply(mask_tensor, elig_tensor)   # .detach()
        active = masked_elig_batch.sum(axis=1) > 0.5
        sub_logit = logits[active]
        sub_mask = masked_elig_batch[active]

        if eval_flag: # TODO:implement
            sub_logit = sub_logit - np.min(sub_logit) + 1.0
            tind = np.argmax(np.multiply(sub_logit, sub_mask.astype(np.float32)), axis=1)
            tind = np.expand_dims(tind, -1)
            tid[active] = _transform(tind, self.tind_to_tid[active]).squeeze()
        else:
            # MSGI-GRProp as adaptation policy
            #tid = self.tind_to_tid[:,0] # init with any subtask.
            masked_logit = sub_logit * sub_mask
            masked_logit = self.temp * (
                masked_logit - np.expand_dims(masked_logit.min(axis=1), -1)
            ) * sub_mask
            prob = tf.nn.softmax(masked_logit, axis=1).numpy()
            prob_masked = prob * sub_mask
            if np.any(prob_masked < 0.0) or np.any(prob_masked.sum(axis=1) < 0.01):
                assert False, "oops!"
            m = tfd.Categorical(prob_masked)
            tind = m.sample().numpy()       # Nbatch x 1
            tid[active] = _transform(tind, self.tind_to_tid[active]).squeeze()

        return tid
