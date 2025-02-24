import sys

from loguru import logger

log_formats = [
    "<light-yellow>[{level}]</> {message}",
    "<level>[{level}]</> {message} <light-yellow>({function}:{line})</>",
]


def simplify(format: str = log_formats[0]):
    # remove the default handlers
    if 0 in logger._core.handlers:
        logger.remove(0)
    logger.add(sys.stderr, format=format)
    logger.info("Log format simplified :)")
