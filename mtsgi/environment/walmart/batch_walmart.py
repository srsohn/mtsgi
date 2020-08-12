import random
import random
import traceback
import copy
import numpy as np
from collections import defaultdict

from .walmart import Walmart, BASE_SUBTASK_IDS, DELIVERY_ADDR_SUBTASK_IDS, \
    DELIVERY_ADDR_REQ_SUBTASK_IDS, SELECT_PAYMENT_SUBTASK_IDS, FILL_CREDIT_SUBTASK_IDS, \
    FILL_GIFT_SUBTASK_IDS, REVIEW_ORDER_SUBTASK_IDS, FAILURE_SUBTASK_IDS

from mtsgi.graph.graph_utils import _to_multi_hot, _transform
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


class BatchWalmartEnv(object):

    def __init__(self, args=None):
        self.verbose_level = args.verbose_level if args else 0
        self.num_envs = args.num_processes if args else 1
        self.num_graphs = 5  # XXX num of trials

        # ToyWoB configuration
        self.config = Walmart()
        self.subtask_list = self.config.subtask_list
        self.num_subtasks = len(self.subtask_list)
        self.ntasks = self.num_subtasks
        self.subtask_name_to_id = self.config.subtask_name_to_id

        # XXX Need configurations for subtask graph
        self.graph = None
        self.action_space = Discrete(self.num_subtasks) # XXX
        #self.observation_space = Box(low=0, high=1, shape=(1, 1), dtype=np.float32) # TODO: remove if possible
        self.feat_dim = 3 * self.num_subtasks
        self.feed_time = True
        if self.feed_time:
            self.feat_dim += 2
        self.feed_prev_ard = True
        if self.feed_prev_ard:
            self.feat_dim += self.num_subtasks + 2

        # trial-based params
        self.epi_count = 0
        self.max_steps = self.config.max_steps  # TODO(srsohn) How is this determined?

        # episode-based params
        self.active = np.ones(self.num_envs)
        self.env_dones = np.zeros(self.num_envs)
        self.prev_active = np.ones(self.num_envs)
        self.step_count = np.zeros(self.num_envs)
        self.prev_actions = [None] * self.num_envs

        # step-based params
        self.rewards = np.zeros(self.num_envs)

        # subtasks
        self.num_objects = np.zeros((self.num_envs))
        self.object_list = []
        #self.sliced_objects = [dict() for _ in range(self.num_envs)]

    def reset_trial(self, nb_epi, reset_graph, graph_index=-1):
        # number of episode to run for current trial
        self.max_epi = nb_epi

        # reset task-related params
        if reset_graph and graph_index is not None:
            if graph_index < 0:
                graph_index = random.randint(0, self.num_graphs - 1)
            self.subtask_rewards = self.config.subtask_reward

        # reset trial-based params
        self.epi_count = 0

        # 2. Reset episode
        self._reset_episode()
        return self.obs, self.feats

    def _reset_episode(self):
        # 1. Reset episode-based params
        self.active.fill(0)
        self.step_count.fill(0)
        self.env_dones.fill(0)

        # 2. Reset & update batch_params
        self._update_batch_params(reset=True)
        self.feats = self._get_feature()
        self.obs = None

    def _update_batch_params(self, actions=None, steps=None, reset=False):
        '''
        Reset & Update comp, elig, mask
        '''
        if reset:
            comps = np.zeros((self.num_envs, self.num_subtasks))
            self.current_levels = np.zeros((self.num_envs, 1))
            self.masks = self._update_mask(comps)
            self.eligs = self._update_eligibility(comps)
            self.comps = comps
        else:
            # 1. mask,comp,elig, obs (delayed_)
            comps = self._update_completion(actions)
            self.current_levels = self._update_level(comps)
            self.eligs = self._update_eligibility(comps)
            self.masks = self._update_mask(comps, actions)
            self.comps = comps

        # 2. active, step_dones, previous (active_)action
        self.prev_active = np.copy(self.active)

        if steps is not None:
            steps *= self.active
            self.step_count += steps
        step_dones = self.step_count >= self.max_steps
        task_dones = (self.masks*self.eligs).sum(axis=1) < 1

        if actions is not None:
            # failures
            self.env_dones = np.logical_or(self.env_dones, actions.squeeze(-1) == self.subtask_name_to_id['Click SP'])
            self.env_dones = np.logical_or(self.env_dones, actions.squeeze(-1) == self.subtask_name_to_id['Click RP'])
            self.env_dones = np.logical_or(self.env_dones, actions.squeeze(-1) == self.subtask_name_to_id['Click Feedback'])
            self.env_dones = np.logical_or(self.env_dones, actions.squeeze(-1) == self.subtask_name_to_id['Click Privacy'])
            self.env_dones = np.logical_or(self.env_dones, actions.squeeze(-1) == self.subtask_name_to_id['Click ToU'])

            # success
            self.env_dones = np.logical_or(self.env_dones, actions.squeeze(-1) == self.subtask_name_to_id['Place Order'])

        dones = np.logical_or(np.logical_or(step_dones, task_dones), self.env_dones)
        self.active = 1 - dones
        self.step_done = step_dones.astype(np.uint8)
        #TODO: episode_count should be +=1 when we reset episode of all the environment at once.

    def _get_feature(self, actions=None):
        if actions is None:
            action_one_hot = np.zeros((self.num_envs, self.num_subtasks))
        else:
            action_one_hot = _to_multi_hot(actions, self.num_subtasks)

        # TODO action -> onehot
        # TODO self.done
        # TODO env.get_log_step
        # TODO epi_dones -> epi_count

        feat_list = [self.masks.astype(np.float32), self.comps.astype(np.float32), self.eligs.astype(np.float32)]

        if self.feed_time:
            remaining_epi = np.zeros((self.num_envs, 1))
            remaining_epi.fill(self.max_epi + 1 - self.epi_count)
            feat_list.append(np.log10(np.expand_dims((self.max_steps + 1 - self.step_count).astype(np.float32), axis=-1)))
            feat_list.append(np.log10(remaining_epi).astype(np.float32))
        if self.feed_prev_ard: # action/reward/done
            #feat_list.append(action_one_hot.float())
            feat_list.append(action_one_hot.astype(np.float32))
            feat_list.append(np.expand_dims(self.rewards.astype(np.float32), axis=-1))
            feat_list.append(np.expand_dims(self.active.astype(np.float32), axis=-1))
        return np.concatenate(feat_list, axis=1)

    def step(self, actions):
        #actions = actions.cpu()

        # 1. Reward
        self.rewards.fill(0)
        for i, iteritem in enumerate(zip(self.active, actions)):
            active, sub_id = iteritem
            if active: # if correctly executed & active
                self.rewards[i] = self.subtask_rewards[sub_id]

        # XXX steps
        steps = np.ones(self.num_envs)

        # 2. Get next (comps, eligs, masks, info, active, dones)
        self._update_batch_params(actions, steps)
        self._update_delayed_states()

        # 2-2. Sanity check (TODO(hjwoo): re-implement for toyminiwob)
        self._do_sanity_check(self.comps, self.masks, self.eligs)

        # 3. Reset episode if done
        episode_reset = False
        if self.active.sum() == 0:
            # 2-1. If done, reset and get init state
            self.epi_count += 1
            if self.epi_count < self.max_epi: # trial is not done.
                self._reset_episode()  # feats is reset in here!
                episode_reset = True

        # 4. Set up self.feats
        if not episode_reset:
            # 2-3. Compute next state
            ### This should be done after computing self.{comp/elig/mask/obs/reward/done}
            self.feats = self._get_feature(actions)

        # 5. prepare Outputs
        rewards = np.expand_dims(self.rewards, axis=-1).astype(np.float32)
        active = np.expand_dims(self.active, axis=-1).astype(np.float32)

        #return observations, self.feats, rewards, active, steps
        self.obs = None
        return self.obs, self.feats, rewards, active, steps

    def _update_eligibility(self, comps):
        eligs = np.zeros_like(comps) # if no precondition, it's eligible
        for comp, elig in zip(comps, eligs):
            # Delivery options & misc
            elig[BASE_SUBTASK_IDS] = 1

            #if comp[self.subtask_name_to_id['Click Pickup']]:
            #    elig[self.subtask_name_to_id['Click OneTrip']] = 1
            if comp[self.subtask_name_to_id['Click Zip']]:
                elig[self.subtask_name_to_id['Fill Zip']] = 1
            if comp[self.subtask_name_to_id['Click Items']]:
                elig[self.subtask_name_to_id['Hide Items']] = 1

            # Base continue
            if comp[self.subtask_name_to_id['Click Continue']]:
                elig[DELIVERY_ADDR_SUBTASK_IDS] = 1

            # Complete delivery address
            if np.all(comp[DELIVERY_ADDR_REQ_SUBTASK_IDS]):
                elig[self.subtask_name_to_id['Click Continue2']] = 1

            # Delivery address continue
            #if comp[self.subtask_name_to_id['Click Continue2']]:
            #    elig[self.subtask_name_to_id['Confirm Addr']] = 1
            #    elig[self.subtask_name_to_id['Edit Addr']] = 1

            # Payment method
            #if comp[self.subtask_name_to_id['Confirm Addr']]:
            #    elig[SELECT_PAYMENT_SUBTASK_IDS] = 1
            if comp[self.subtask_name_to_id['Click Continue2']]:
                elig[SELECT_PAYMENT_SUBTASK_IDS] = 1

            # Credit information
            if comp[self.subtask_name_to_id['Click Credit']]:
                elig[FILL_CREDIT_SUBTASK_IDS] = 1

            # Gift information
            if comp[self.subtask_name_to_id['Click Gift']]:
                elig[FILL_GIFT_SUBTASK_IDS] = 1

            # Gift apply
            #if comp[self.subtask_name_to_id['Fill G_NUM']] and comp[self.subtask_name_to_id['Fill G_PIN']]:
            #    elig[self.subtask_name_to_id['Click G_Apply']] = 1

            # Complete payment information
            #if np.all(comp[FILL_CREDIT_SUBTASK_IDS]) or np.all(comp[FILL_GIFT_SUBTASK_IDS]):
            if np.all(comp[FILL_CREDIT_SUBTASK_IDS]):
                elig[self.subtask_name_to_id['Click Continue3']] = 1

            # Payment continue
            if comp[self.subtask_name_to_id['Click Continue3']]:
                elig[REVIEW_ORDER_SUBTASK_IDS] = 1

        return eligs

    def _update_mask(self, comps, actions=None):
        masks = 1. - comps
        for i, (mask, comp) in enumerate(zip(masks, comps)):
            # if current level >= 1
            if self.current_levels[i][0] >= 1:
                mask[self.subtask_name_to_id['Click Delivery']] = 0
                #mask[self.subtask_name_to_id['Click Pickup']] = 0
                #mask[self.subtask_name_to_id['Click OneTrip']] = 0
                mask[self.subtask_name_to_id['Click SP']] = 0
                mask[self.subtask_name_to_id['Click RP']] = 0
                mask[self.subtask_name_to_id['Click Feedback']] = 0

            # if current level >= 2
            if self.current_levels[i][0] >= 2:
                mask[self.subtask_name_to_id['Fill Apt']] = 0

            # if current level >= 3
            #if self.current_levels[i][0] >= 3:
            #    mask[self.subtask_name_to_id['Edit Addr']] = 0

            # if current level >= 4
            if self.current_levels[i][0] >= 3:
                # credit
                mask[self.subtask_name_to_id['Click Credit']] = 0
                mask[FILL_CREDIT_SUBTASK_IDS] = 0

                # gift
                mask[self.subtask_name_to_id['Click Gift']] = 0
                mask[FILL_GIFT_SUBTASK_IDS] = 0
                #mask[self.subtask_name_to_id['Click G_Apply']] = 0

        return masks

    def _update_level(self, comps):
        levels = np.zeros((self.num_envs, 1))
        for i, comp in enumerate(comps):
            if comp[self.subtask_name_to_id['Click Continue']]:
                levels[i][0] = 1
            if comp[self.subtask_name_to_id['Click Continue2']]:
                levels[i][0] = 2
            #if comp[self.subtask_name_to_id['Confirm Addr']]:
            #    levels[i][0] = 3
            if comp[self.subtask_name_to_id['Click Continue3']]:
                levels[i][0] = 3
        return levels

    def _update_completion(self, actions):
        comps = copy.deepcopy(self.comps)
        for i, comp in enumerate(comps):
            # See/Hide Items
            if actions[i] == self.subtask_name_to_id['Hide Items']:
                comp[self.subtask_name_to_id['Click Items']] = 0

            # Current zipcode
            elif actions[i] == self.subtask_name_to_id['Fill Zip']:
                comp[self.subtask_name_to_id['Click Zip']] = 0

            # Edit (delivery level)
            #elif actions[i] == self.subtask_name_to_id['Edit Delivery']:
            #    # Delivery continue
            #    comp[self.subtask_name_to_id['Click Continue']] = 0

            #    # Delivery address
            #    comp[DELIVERY_ADDR_SUBTASK_IDS] = 0

            #    comp[self.subtask_name_to_id['Click Continue2']] = 0
            #    comp[self.subtask_name_to_id['Confirm Addr']] = 0

            #    # credit
            #    comp[self.subtask_name_to_id['Click Credit']] = 0
            #    comp[FILL_CREDIT_SUBTASK_IDS] = 0

            #    # gift card
            #    comp[self.subtask_name_to_id['Click Gift']] = 0
            #    comp[FILL_GIFT_SUBTASK_IDS] = 0

            #    # Payment continue
            #    comp[self.subtask_name_to_id['Click Continue3']] = 0

            ## Edit (address level)
            #elif actions[i] == self.subtask_name_to_id['Edit Addr']:
            #    comp[self.subtask_name_to_id['Click Continue2']] = 0

            #elif actions[i] == self.subtask_name_to_id['Edit Addr2']:
            #    comp[self.subtask_name_to_id['Click Continue2']] = 0
            #    comp[self.subtask_name_to_id['Confirm Addr']] = 0

            #    # credit
            #    comp[self.subtask_name_to_id['Click Credit']] = 0
            #    comp[FILL_CREDIT_SUBTASK_IDS] = 0

            #    # gift card
            #    comp[self.subtask_name_to_id['Click Gift']] = 0
            #    comp[FILL_GIFT_SUBTASK_IDS] = 0

            #    # Payment continue
            #    comp[self.subtask_name_to_id['Click Continue3']] = 0

            ## Edit (payment level)
            #elif actions[i] == self.subtask_name_to_id['Edit Payment']:
            #    comp[self.subtask_name_to_id['Click Continue3']] = 0
            else:
                # Credit/Gift toggle
                if actions[i] == self.subtask_name_to_id['Click Credit']:
                    comp[self.subtask_name_to_id['Click Gift']] = 0
                if actions[i] == self.subtask_name_to_id['Click Gift']:
                    comp[self.subtask_name_to_id['Click Credit']] = 0

                comp[actions[i].item()] = 1
        return comps

    def _update_delayed_states(self):# Used for ILP module
        # never gets the initial state. Instead, get the final state.
        self.masks_delayed = self.masks
        self.comps_delayed = self.comps
        self.eligs_delayed = self.eligs


    def get_indexed_states(self):
        return self.masks, self.comps, self.eligs

    def get_delayed_indexed_states(self):
        return self.prev_active, self.step_done, self.masks_delayed, self.comps_delayed, self.eligs_delayed

    # utils (TODO: change to property)
    def get_subtask(self, sub_id):
        id = self.subtask_list[sub_id]['id']
        name = self.subtask_list[sub_id]['name']
        return id, name

    def _get_id_from_ind(self, input_inds):
        return _transform(input_inds, self.ind_to_id)

    def _get_ind_from_id(self, input_ids):
        return _transform(input_ids, self.id_to_ind)

    def get_subtask_lists(self):
        return np.tile(np.arange(self.num_subtasks), [self.num_envs, 1])

    def get_tid_to_tind(self):
        return np.tile(np.arange(self.num_subtasks), [self.num_envs, 1])

    def _do_sanity_check(self, comps, masks, eligs):
        pass

    def get_graphs(self):
        # TODO: implement
        raise NotImplementedError
        return self.graph
