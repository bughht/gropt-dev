import logging


def set_log_level(level: int) -> None:
    """Set gropt-torch log level (0=debug, 1=info, 2=warning, 3=error)."""
    logger = logging.getLogger("gropt_torch")
    if level <= 0:
        logger.setLevel(logging.DEBUG)
    elif level == 1:
        logger.setLevel(logging.INFO)
    elif level == 2:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)


def setup_logging(level: int = 1) -> None:
    """Configure gropt-torch logging output."""
    logger = logging.getLogger("gropt_torch")
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("| {levelname:>8} |  {message}", style="{"))
        logger.addHandler(handler)
    set_log_level(level)
