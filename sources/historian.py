"""Data-writer & logger class"""
import logging
import json


class Historian:
    def __init__(self, fmtstr: str = None):
        self._fmtstr = fmtstr
        self._storage = list()

    def append(self, **kwargs):
        if self._fmtstr is not None:
            logging.info(self._fmtstr.format(**kwargs))
        self._storage.append(kwargs)

    @property
    def storage(self):
        return self._storage

    def save(self, path: str):
        with open(path, 'w+') as file:
            json.dump(self.storage, file, indent=2)

    @staticmethod
    def load(path: str):
        with open(path, 'r') as file:
            json.load(file)
