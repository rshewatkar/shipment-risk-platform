import logging
from logging import Logger


class StructuredLogger(Logger):
    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        stack_info=False,
        stacklevel=1,
        **kwargs,
    ):
        extra = kwargs.pop("extra", {})
        if kwargs:
            extra = {**extra, **kwargs}
        super()._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
        )


def get_logger(name: str) -> Logger:
    """Return a configured logger for the given module name."""
    logger = StructuredLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
