#!/usr/bin/env python3
"""Validate and normalize the release candidate identity used by release gates."""
import re


_TARGET = re.compile(r"^v1\.3\.(?P<minor>[5-9]|[1-9][0-9]+)-evolve$")


def validate_target(value: str) -> str:
    """Return *value* when it is a supported v1.3 evolve candidate."""
    if not isinstance(value, str) or not _TARGET.fullmatch(value):
        raise ValueError("release target must match v1.3.<5+>-evolve")
    return value


def target_for_tag(tag: str) -> str:
    return validate_target(tag)
