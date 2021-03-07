"""Data-writer & logger class"""
import logging


class Historian:
    def __init__(self, fmtstr: str = None):
        self._fmtstr = fmtstr
        self._storage = list()

    def append(self, *args, **kwargs):
        if self._fmtstr is not None:
            logging.info(self._fmtstr.format(*args, **kwargs))
        self._storage.append(args)

    @property
    def storage(self):
        return self._storage
