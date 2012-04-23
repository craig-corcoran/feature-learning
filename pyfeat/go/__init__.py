from go import *

try:
    from fuego import *
except ImportError:
    # XXX
    logger.warning("failed to import Fuego; no Go functionality available")

