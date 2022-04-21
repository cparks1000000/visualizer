from __future__ import annotations

import \
    os
from abc import ABC
from typing import Any, Optional

import \
    torch
from bidict import bidict
from torch import nn

from util.utils import build_signature


class Savable(nn.Module, ABC):
    _default_root: str = "_network_weights"
    _file_name_dictionary: bidict[str, type] = dict()

    # The file_name parameter must be a property of the inheriting class.
    def __init__(self, file_name: str, *signature_args: Any):
        super().__init__()
        self._signature = build_signature(*signature_args)
        assert file_name not in self._file_name_dictionary or self._file_name_dictionary[file_name] == type(self), "The file name " + file_name + " has been used twice."
        self._file_name_dictionary[file_name] = type(self)
        self._file_name = file_name

    def save(self,  file_name: Optional[str] = None, root: str = _default_root) -> None:
        path = self.get_path(file_name, root)
        try:
            torch.save(self.state_dict(), path)
        except FileNotFoundError:
            os.makedirs(root)
            torch.save(self.state_dict(), path)

    def load(self, file_name: Optional[str] = None, root: str = _default_root) -> Savable:
        path = self.get_path(file_name, root)
        self.load_state_dict(torch.load(path))
        return self

    def get_path(self, file_name: Optional[str] = None, root: str = _default_root):
        if file_name is None:
            file_name = self._file_name
        return root + "/" + file_name + "_" + self._signature + ".pth"
