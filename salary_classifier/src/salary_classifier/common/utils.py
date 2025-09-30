import colorlog
import logging

def logging_config():
    """Configure logging with color support."""

    # Configure colorlog for colored logging output
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

    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
