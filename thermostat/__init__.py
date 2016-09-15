import logging

log = logging.getLogger('thermostat')
log.setLevel(logging.DEBUG)
log = logging

from . import sortinstance, utils, statprocess, sortcomp
