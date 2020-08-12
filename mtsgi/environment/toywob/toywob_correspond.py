'''Configuration script for ToyWoB environment.'''
import numpy as np


def check_type(name, typeset):
    for prefix in typeset:
        if prefix in name:
            return True
    return False


CLICK_SUBTASKS = [ # Click button or radio button or check box
    # Billing
    'Click Check',
    # Payment
    'Select Credit',
    'Select Debit',
    'Select PayPal',
]

FILL_IN_SUBTASKS = [
    # Billing
    'Fill First',
    'Fill Last',
    'Fill Email',
    'Fill Addr',
    'Fill Addr2',
    'Fill Zip',
    'Fill Promo',
    # Credit
    'Fill C_name',
    'Fill C_CVV',
    'Fill C_number',
    'Fill C_date',
    # Debit
    'Fill D_name',
    'Fill D_CVV',
    'Fill D_number',
    'Fill D_date',
    # PayPal
    'Fill PP_id',
    'Fill PP_pw'
]

SUBTASK_NAMES = FILL_IN_SUBTASKS + CLICK_SUBTASKS

SUBTASK_LIST = []
for idx, subtask_name in enumerate(SUBTASK_NAMES):
    SUBTASK_LIST.append(dict(
        name=subtask_name,
        id=idx
    ))

LABEL_NAME = {subtask['id']: subtask['name'] for subtask in SUBTASK_LIST}

# Fill Billing/Misc + Select
BASE_SUBTASKS = [
    'Fill First',
    'Fill Last',
    'Fill Email',
    'Fill Addr',
    'Fill Addr2',
    'Fill Zip',
    'Fill Promo',
    'Select Credit',
    'Select Debit',
    'Select PayPal'
]

BASE_SUBTASK_IDS = []
for subtask in SUBTASK_LIST:
    for base_subtask in BASE_SUBTASKS:
        if base_subtask == subtask['name']:
            BASE_SUBTASK_IDS.append(subtask['id'])

###
FILL_CREDIT_SUBTASK_IDS = []
FILL_DEBIT_SUBTASK_IDS  = []
FILL_PAYPAL_SUBTASK_IDS = []
SEL_PAYMENT_SUBTASK_IDS = []
for subtask in SUBTASK_LIST:
    if 'Fill C_' in subtask['name']:
        FILL_CREDIT_SUBTASK_IDS.append(subtask['id'])
    if 'Fill D_' in subtask['name']:
        FILL_DEBIT_SUBTASK_IDS.append(subtask['id'])
    if 'Fill PP_' in subtask['name']:
        FILL_PAYPAL_SUBTASK_IDS.append(subtask['id'])

    if 'Select' in subtask['name']:
        SEL_PAYMENT_SUBTASK_IDS.append(subtask['id'])
###

class ToyWoB(object):
    def __init__(self):
        # map
        self.env_name = 'toywob'

        # subtasks
        subtask_list = SUBTASK_LIST
        subtask_name_to_id = dict()
        for subtask in subtask_list:
            subtask_name_to_id[subtask['name']] = subtask['id']

        # XXX no param for now
        #subtask_param_to_id = dict()
        #subtask_param_list = []
        #for i in range(len(subtask_list)):
        #    subtask = subtask_list[i]
        #    par = subtask['param']
        #    subtask_param_list.append ( par )
        #    subtask_param_to_id[ par ] = i
        #nb_obj_type = len(object_list)
        #nb_operation_type = len(operation_list)

        #self.operation_list=operation_list
        #self.nb_operation_type=nb_operation_type

        #self.object_list = object_list
        #self.nb_obj_type = nb_obj_type
        #self.item_name_to_iid = item_name_to_iid
        #self.nb_block = nb_block
        self.subtask_list = subtask_list
        self.subtask_name_to_id = subtask_name_to_id
        #self.object_image_list=object_image_list

        # XXX no param for now
        #self.subtask_param_list = subtask_param_list
        #self.subtask_param_to_id = subtask_param_to_id

        self.nb_subtask_type = len(subtask_list)
        self.max_steps = 35  # TODO (srsohn) How is this determined?
        self.width = 10
        self.height = 10
        self.feat_dim = 4*self.nb_subtask_type+4
        self.subtask_reward = np.zeros((self.nb_subtask_type))
        self.subtask_reward[subtask_name_to_id['Click Check']] = 1.
