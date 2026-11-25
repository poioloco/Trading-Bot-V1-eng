"""Config: reading and SURGICAL editing of params.yaml.

params.yaml is full of comments that explain every parameter in words: they
are valuable documentation and must NOT be lost. That is why there is no
yaml.dump round-trip (which would throw away all the comments): edits change
ONLY the line of the key concerned, leaving everything else untouched.
"""
from .editor import InvalidValue, ParamsEditor

__all__ = ["ParamsEditor", "InvalidValue"]
