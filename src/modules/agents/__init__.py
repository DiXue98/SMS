REGISTRY = {}

from .n_rnn_agent import NRNNAgent
from .sms_agent import SMSAgent

REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["sms"] = SMSAgent