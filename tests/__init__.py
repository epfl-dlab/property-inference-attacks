import logging

from src import logger

for handler in logger.handlers:
    if handler.name == 'consoleHandler':
        handler.setLevel(logging.WARNING)