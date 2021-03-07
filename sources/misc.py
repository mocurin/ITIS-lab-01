"""Miscellaneous functions for `source` package"""
from typing import List


def subclasshook_helper(subclass, methods: List[str]) -> bool:
    """Check presence of every method from `methods` in a subclass"""

    def checker(method: str):
        """Trivial subclass closure"""
        return hasattr(subclass, method) and callable(getattr(subclass, method))

    # Create generator so checking could early stop & preserve memory
    methods = (checker(method) for method in methods)

    # Either `True` or `NotImeplemeted`
    return all(methods) or NotImplemented
