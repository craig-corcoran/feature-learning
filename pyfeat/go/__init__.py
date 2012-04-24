from go import *
from . import hash_maps
from . import py_hash_maps
from . import feature_maps

try:
    from fuego import *
except ImportError:
    # XXX
    logger.warning("failed to import Fuego; no Go functionality available")

