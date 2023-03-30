from .nq_learner import NQLearner
from .sms_learner import SMSLearner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["sms_learner"] = SMSLearner