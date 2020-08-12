'''Configuration script for Walmart environment.'''
import numpy as np


def check_type(name, typeset):
    for prefix in typeset:
        if prefix in name:
            return True
    return False

# ---------------------------------------------------------- v v v Level 0
BASE_SUBTASKS = [
    'Click Delivery',
    #'Click Pickup',
    'Click Continue',
    'Click Items',     # See item details
    'Click Zip',
    'Click Feedback',  # Leave feedback
    'Click SP',        # Do not share personal information
    'Click RP',        # Request my personal information
]

BASE_EXPAND_SUBTASKS = [
    #'Click OneTrip',  # Make one trip instead
    'Fill Zip',       # Current Zipcode
    'Hide Items'      # Hide item details
]

# ---------------------------------------------------------- v v v Level 1
DELIVERY_ADDR_SUBTASKS = [
    'Fill First',
    'Fill Last',
    'Fill Street',
    'Fill Apt',    # XXX Optional
    'Fill Phone',
    'Fill Email',
    'Fill City',
    'Fill State',
    'Fill Zipcode',
    #'Edit Delivery'
]

CONTINUE2_SUBTASK = ['Click Continue2']

# ---------------------------------------------------------- v v v Level 2
# XXX seems too redundant
#CONFIRM_ADDR_SUBTASKS = [
#    'Confirm Addr',
#    'Edit Addr'
#]

# ---------------------------------------------------------- v v v Level 3
SELECT_PAYMENT_SUBTASKS = [
    #'Edit Addr2',
    'Click Credit',
    'Click Gift'
]

FILL_CREDIT_SUBTASKS = [
    'Fill C_First',
    'Fill C_Last',
    'Fill C_NUM',
    'Fill C_EXPMM',
    'Fill C_EXPYY',
    'Fill C_CVV',
    'Fill C_Phone'
]

FILL_GIFT_SUBTASKS = [
    'Fill G_NUM',
    'Fill G_PIN',
    'Click No G_PIN',
    'Click G_Apply'
]

#APPLY_GIFT_SUBTASK = ['Click G_Apply']

CONTINUE3_SUBTASK = ['Click Continue3']

# ---------------------------------------------------------- v v v Level 4
REVIEW_ORDER_SUBTASKS = [
    #'Edit Payment',
    'Place Order',    # XXX Success
    'Click Privacy',  # Privacy Policy
    'Click ToU'       # Terms of Use
]
# ------------------------------------------------------------------------

FAILURE_SUBTASKS = [
    'Click Feedback',  # Leave feedback
    'Click SP',        # Do not share personal information
    'Click RP',        # Request my personal information
    'Click Privacy',   # Privacy Policy
    'Click ToU'        # Terms of Use
]

# List of subtask names
SUBTASK_NAMES = BASE_SUBTASKS + BASE_EXPAND_SUBTASKS + DELIVERY_ADDR_SUBTASKS + \
    CONTINUE2_SUBTASK + SELECT_PAYMENT_SUBTASKS + FILL_CREDIT_SUBTASKS + \
    FILL_GIFT_SUBTASKS + CONTINUE3_SUBTASK + REVIEW_ORDER_SUBTASKS

# List of subtasks
SUBTASK_LIST = []
for idx, subtask_name in enumerate(SUBTASK_NAMES):
    SUBTASK_LIST.append(dict(
        name=subtask_name,
        id=idx
    ))

LABEL_NAME = {subtask['id']: subtask['name'] for subtask in SUBTASK_LIST}

###
BASE_SUBTASK_IDS = []
DELIVERY_ADDR_SUBTASK_IDS = []
DELIVERY_ADDR_REQ_SUBTASK_IDS = []
SELECT_PAYMENT_SUBTASK_IDS = []
FILL_CREDIT_SUBTASK_IDS = []
FILL_GIFT_SUBTASK_IDS = []
REVIEW_ORDER_SUBTASK_IDS = []
FAILURE_SUBTASK_IDS = []

for subtask in SUBTASK_LIST:
    # base
    if subtask['name'] in BASE_SUBTASKS:
        BASE_SUBTASK_IDS.append(subtask['id'])

    # delivery address
    if subtask['name'] in DELIVERY_ADDR_SUBTASKS:
        DELIVERY_ADDR_SUBTASK_IDS.append(subtask['id'])
    #if subtask['name'] in DELIVERY_ADDR_SUBTASKS and subtask['name'] not in ['Fill Apt', 'Edit Delivery']:
    #    DELIVERY_ADDR_REQ_SUBTASK_IDS.append(subtask['id'])
    if subtask['name'] in DELIVERY_ADDR_SUBTASKS and subtask['name'] != 'Fill Apt':
        DELIVERY_ADDR_REQ_SUBTASK_IDS.append(subtask['id'])

    # payment
    if subtask['name'] in SELECT_PAYMENT_SUBTASKS:
        SELECT_PAYMENT_SUBTASK_IDS.append(subtask['id'])
    if subtask['name'] in FILL_CREDIT_SUBTASKS:
        FILL_CREDIT_SUBTASK_IDS.append(subtask['id'])
    if subtask['name'] in FILL_GIFT_SUBTASKS:
        FILL_GIFT_SUBTASK_IDS.append(subtask['id'])

    # review order
    if subtask['name'] in REVIEW_ORDER_SUBTASKS:
        REVIEW_ORDER_SUBTASK_IDS.append(subtask['id'])

    # failures
    if subtask['name'] in FAILURE_SUBTASKS:
        FAILURE_SUBTASK_IDS.append(subtask['id'])
###

class Walmart(object):
    def __init__(self):
        # map
        self.env_name = 'walmart'

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
        self.max_steps = 200  # TODO (srsohn) How is this determined?
        self.width = 10
        self.height = 10
        self.feat_dim = 4*self.nb_subtask_type+4

        # subtask reward
        self.subtask_reward = np.zeros((self.nb_subtask_type))
        self.subtask_reward[subtask_name_to_id['Place Order']] = 3.
        self.subtask_reward[subtask_name_to_id['Click SP']] = -1.
        self.subtask_reward[subtask_name_to_id['Click RP']] = -1.
        self.subtask_reward[subtask_name_to_id['Click Feedback']] = -1.
        self.subtask_reward[subtask_name_to_id['Click Privacy']] = -1.
        self.subtask_reward[subtask_name_to_id['Click ToU']] = -1.
