'''Configuration script for ToyWoB environment.'''
import numpy as np


def check_type(name, typeset):
    for prefix in typeset:
        if prefix in name:
            return True
    return False

# TODO: change below for new website
# ---------------------------------------------------------- v v v Level 0 : Guest Checkout
BASE_SUBTASKS = [
    'Click Guest Checkout'  # Click guest checkout
]

L0_SUBTASKS = BASE_SUBTASKS
# ---------------------------------------------------------- v v v Level 1 : Contact & Billing/Shipping Address
CONTACT_SUBTASKS = [
    'Fill First',
    'Fill Last',
    'Fill Email',
    'Fill Phone'
]

BILLING_ADDR_SUBTASKS = [
    'Fill Street',
    'Fill Zipcode'
]

SHIPPING_ADDR_SUBTASKS = [
    'Fill ShipFirst',
    'Fill ShipLast',
    'Fill ShipStreet',
    'Fill ShipZipcode'
]

SAME_ADDR_SUBTASKS = [
    'Click Same Address',
    'Unclick Same Address'
]  # Billing & Shipping are same

CONTINUE_SUBTASK = ['Click Continue']  # Continue to payment

L1_SUBTASKS = CONTACT_SUBTASKS + BILLING_ADDR_SUBTASKS + SAME_ADDR_SUBTASKS + SHIPPING_ADDR_SUBTASKS + \
    CONTINUE_SUBTASK

# ---------------------------------------------------------- v v v Level 2 : Payment
SELECT_PAYMENT_SUBTASKS = [
    'Click Credit',
    'Click Gift',
    'Click PayPal'
]

FILL_CREDIT_SUBTASKS = [
    'Fill C_NUM',
    'Fill C_EXP',
    'Fill C_CVV'
]

FILL_COUPON_SUBTASK = ['Fill Coupon Code']
APPLY_COUPON_SUBTASK = ['Click Apply Code']

CONTINUE3_SUBTASK = ['Click Place Order']

L2_SUBTASKS = SELECT_PAYMENT_SUBTASKS + FILL_CREDIT_SUBTASKS + FILL_COUPON_SUBTASK + \
    APPLY_COUPON_SUBTASK + CONTINUE3_SUBTASK

# ---------------------------------------------------------- END

# List of subtask names
SUBTASK_NAMES = L0_SUBTASKS + L1_SUBTASKS + L2_SUBTASKS

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
CONTACT_BILLING_SUBTASK_IDS = []    # Contact + Billing
SHIPPING_ADDR_SUBTASK_IDS = []      # Shipping
SAME_ADDR_SUBTASK_IDS = []
#DELIVERY_ADDR_REQ_SUBTASK_IDS = []  

SELECT_PAYMENT_SUBTASK_IDS = []
FILL_CREDIT_SUBTASK_IDS = []

L0_SUBTASK_IDS = []
L1_SUBTASK_IDS = []
L2_SUBTASK_IDS = []

for subtask in SUBTASK_LIST:
    # base
    if subtask['name'] in BASE_SUBTASKS:
        BASE_SUBTASK_IDS.append(subtask['id'])

    # contact & billing
    if subtask['name'] in CONTACT_SUBTASKS + BILLING_ADDR_SUBTASKS:
        CONTACT_BILLING_SUBTASK_IDS.append(subtask['id'])

    # shipping
    if subtask['name'] in SHIPPING_ADDR_SUBTASKS:
        SHIPPING_ADDR_SUBTASK_IDS.append(subtask['id'])

    # same shipping & billing
    if subtask['name'] in SAME_ADDR_SUBTASKS:
        SAME_ADDR_SUBTASK_IDS.append(subtask['id'])

    #if subtask['name'] in DELIVERY_ADDR_SUBTASKS and subtask['name'] != 'Fill Apt':
    #    DELIVERY_ADDR_REQ_SUBTASK_IDS.append(subtask['id'])

    # payment
    if subtask['name'] in SELECT_PAYMENT_SUBTASKS:
        SELECT_PAYMENT_SUBTASK_IDS.append(subtask['id'])
    if subtask['name'] in FILL_CREDIT_SUBTASKS:
        FILL_CREDIT_SUBTASK_IDS.append(subtask['id'])

    # subtask ids by level
    if subtask['name'] in L0_SUBTASKS:
        L0_SUBTASK_IDS.append(subtask['id'])
    if subtask['name'] in L1_SUBTASKS:
        L1_SUBTASK_IDS.append(subtask['id'])
    if subtask['name'] in L2_SUBTASKS:
        L2_SUBTASK_IDS.append(subtask['id'])
###

class Dicks(object):
    def __init__(self):
        # map
        self.env_name = 'dicks'

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
        self.max_steps = 200  
        self.width = 10
        self.height = 10
        self.feat_dim = 4*self.nb_subtask_type+4

        # subtask reward
        self.subtask_reward = np.zeros((self.nb_subtask_type))
        self.subtask_reward[subtask_name_to_id['Click Place Order']] = 3.
