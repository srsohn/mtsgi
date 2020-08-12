from .walmart.batch_walmart import BatchWalmartEnv
from .dicks.batch_dicks import BatchDicksEnv
from .toywob.batch_toywob_correspond import BatchToyWoBEnv


def get_subtask_label(env_id):
    if env_id == 'toywob':
        from mtsgi.environment.toywob.toywob_correspond import LABEL_NAME
        return LABEL_NAME
    elif env_id == 'walmart':
        from mtsgi.environment.walmart.walmart import LABEL_NAME
        return LABEL_NAME
    elif env_id == 'dicks':
        from mtsgi.environment.dicks.dicks import LABEL_NAME
        return LABEL_NAME
    else:
        return ValueError(env_id)
