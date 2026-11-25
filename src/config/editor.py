"""Surgical editor for params.yaml (keeps ALL the comments).

`ParamsEditor.set("vix.voting_method", "gate")` finds the key's line by
following the dotted path through the indentation levels, replaces ONLY the
value (any inline comment stays), checks that the re-read file is valid YAML
with the expected value in the right place, and saves with a `.bak` backup.
Used by the `update-data` command (for `data.history_end`) and by the
Config UI.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml


class InvalidValue(ValueError):
    """The proposed value fails validation (type/choices/range)."""


def _format(value) -> str:
    """Inline YAML representation of the value (scalars or inline dicts)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        # inline dict in params.yaml style: {sell_spx: -3.5, longvol: -2.5}
        return "{" + ", ".join(f"{k}: {_format(v)}" for k, v in value.items()) + "}"
    if value is None or value == "":
        return '""'
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    s = str(value)
    # quote strings that YAML might interpret (dates, numbers, on/off)
    needs_quote = (
        s != s.strip() or s.lower() in {"true", "false", "null", "yes", "no",
                                        "on", "off"}
        or any(c in s for c in ":#{}[],&*!|>'\"%@`")
        or s.replace(".", "").replace("-", "").isdigit()
    )
    return f'"{s}"' if needs_quote else s


class ParamsEditor:
    def __init__(self, path: str | Path = "params.yaml") -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------ #
    def read(self) -> dict:
        with open(self.path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def value(self, key: str):
        """Current value of a dotted key, e.g. 'vix.voting_method'."""
        node = self.read()
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    # ------------------------------------------------------------------ #
    def _find_line(self, lines: list[str], key: str) -> int:
        """Index of the line of the final key of the dotted path.

        Goes down level by level: each segment must be INSIDE the block of
        the previous one (greater indentation, before the block closes).
        Raises KeyError if the path does not exist in the file.
        """
        parts = key.split(".")
        i, parent_indent = 0, -1
        for depth, part in enumerate(parts):
            found = False
            while i < len(lines):
                l = lines[i]
                s = l.strip()
                if not s or s.startswith("#"):
                    i += 1
                    continue
                indent = len(l) - len(l.lstrip())
                if indent <= parent_indent:          # parent block closed
                    break
                if indent == parent_indent + 2 or (parent_indent == -1 and indent == 0):
                    name = s.split(":", 1)[0].strip().strip("'\"")
                    if name == part and ":" in s:
                        found = True
                        if depth == len(parts) - 1:
                            return i
                        parent_indent = indent
                        i += 1
                        break
                i += 1
            if not found:
                raise KeyError(f"key '{key}' not found in {self.path}"
                               f" ('{part}' is missing)")
        raise KeyError(f"key '{key}' not found in {self.path}")

    # ------------------------------------------------------------------ #
    def set(self, key: str, value, backup: bool = True) -> None:
        """Replaces the key's value, single line, comments untouched.

        After the edit the file is re-read: it must be valid YAML and the
        key must have the expected value, otherwise everything is restored.
        """
        if value is None:
            value = ""            # None is written as "" (= empty in YAML)
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        idx = self._find_line(lines, key)
        l = lines[idx]
        indent = l[: len(l) - len(l.lstrip())]
        name = l.strip().split(":", 1)[0]
        rest = l.split(":", 1)[1]
        comment = ""
        if "#" in rest:
            pos = rest.index("#")
            comment = "   " + rest[pos:].rstrip()
        lines[idx] = f"{indent}{name}: {_format(value)}{comment}"
        new = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

        # sanity: valid YAML and the value in the right place
        parsed = yaml.safe_load(new)
        node = parsed
        for part in key.split("."):
            # quadrant keys are INTEGERS in the YAML (1: {...})
            if part not in node and part.isdigit() and int(part) in node:
                part = int(part)
            node = node[part]
        if node != value and str(node) != str(value):
            raise RuntimeError(
                f"after the edit '{key}' is {node!r}, expected {value!r}")

        if backup:
            shutil.copyfile(self.path, str(self.path) + ".bak")
        self.path.write_text(new, encoding="utf-8")
