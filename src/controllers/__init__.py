REGISTRY = {}

from .n_controller import NMAC
from .sms_controller import SMSMAC

REGISTRY["n_mac"] = NMAC
REGISTRY["sms_mac"] = SMSMAC