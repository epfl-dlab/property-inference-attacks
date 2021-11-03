import logging
logger = logging.getLogger('propinfer')

for handler in logger.handlers:
    if handler.name == 'consoleHandler':
        handler.setLevel(logging.WARNING)