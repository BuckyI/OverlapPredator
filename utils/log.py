import sys

from loguru import logger


def simplify():
    # remove the default handlers
    if 0 in logger._core.handlers:
        logger.remove(0)
    logger.add(sys.stderr, format="<light-yellow>[{level}]</> {message}")
    logger.info("Log format simplified :)")
