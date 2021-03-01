"""Data-writer & logger class"""
import logging


class Historian:
    def __init__(self, fmtstr: str = ''):
        self._fmtstr = fmtstr
        self._storage = list()

    def append(self, *args):
        if self._fmtstr:
            logging.info(self._fmtstr.format(*args))
        self._storage.append(args)

    @property
    def storage(self):
        return self._storage
