import logging
import enum

log = logging.getLogger('thermostat')
log.setLevel(logging.DEBUG)
log = logging

class DisorderMeasure(enum.Enum):
    INV = 1
    DISLOC = 2
    W_DISLOC = 3


class ErrorCostType(enum.Enum):
    """
    How to simulate comparison errors:
    * UNIT:
        Every comparison has the same error p_err
    * VALUE:
        The error probability depends on the value difference exponentially,
        as if the energy difference was the value difference.
    * VALUE_DIST: 
        The error probability depends on the value and distance of compared
        items, as if energy difference was `dist * (a - b)`.
        This allows for reversible chain with non-adjacent swaps.
    * DIST: 
        The error probability depends on the distance of compared
        items, as if energy difference was `dist`.
        (Distant errors are more likely corrected, but no other motivation.)
    """
    UNIT = 0
    VALUE = 1
    VALUE_DIST = 2
    DIST = 3

from . import sortinstance, utils, statprocess, sortcomp
