import os
import sys
from loguru import logger


loglevel = 'DEBUG' if os.getenv('DEBUG') is not None else 'INFO'
logger.remove()
logger.add(sys.stdout, colorize=True, level=loglevel)
