import logging
logger = logging.getLogger('pia')

for handler in logger.handlers:
    if handler.name == 'consoleHandler':
        handler.setLevel(logging.WARNING)