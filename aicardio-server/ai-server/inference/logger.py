import logging


__all__ = ["Logger"]


class Logger(logging.Logger):
    r"""Logger used to log progress of inference steps"""
    def __init__(self, name="logger"):
        r"""Initialize a logger

        Args:
            name (str): Logger name
        """
        super(Logger, self).__init__(name)
