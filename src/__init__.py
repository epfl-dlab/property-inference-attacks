import logging
import logging.config

from os import path, mkdir
from time import strftime

config = path.abspath(path.join(path.dirname(__file__), '..', 'logging.ini'))


logdir = path.abspath(path.join(path.dirname(__file__),"../logs"))
if not path.isdir(logdir):
    mkdir(logdir)
logfile = logdir + '/logs_property-inference-framework_' + strftime('%d%m%y_%H:%M:%S')


logging.config.fileConfig(config, defaults={'logfilename': logfile})

# create logger
logger = logging.getLogger('property-inference-framework')