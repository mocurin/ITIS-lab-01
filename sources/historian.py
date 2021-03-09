"""Data-writer & logger class"""
import logging
import pickle


class Historian:
    def __init__(self, fmtstr: str = None):
        self._fmtstr = fmtstr
        self._storage = list()

    def append(self, **kwargs):
        if self._fmtstr is not None:
            logging.info(self._fmtstr.format(**kwargs))
        values = [value for value in kwargs.values()]
        self._storage.append(values)

    @property
    def storage(self):
        return self._storage

    def save(self, path: str):
        with open(path, 'wb+') as file:
            pickle.dump(self.storage, file)

    @staticmethod
    def load(path: str):
        with open(path, 'r') as file:
            pickle.load(file)
