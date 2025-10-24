import colorlog
import logging


def logging_config(name: str = "my_logger"):
    """Configure logging with color support, and avoid duplicate handlers."""
    logger = colorlog.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s[%(levelname)s] - %(asctime)s - %(funcName)s - %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
        logger.addHandler(handler)

    return logger
