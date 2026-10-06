"""Shared enum types used across Nirnaya."""

from enum import StrEnum


class Primitive(StrEnum):
    """Decision primitive supported by every model variant."""

    CHOICE = "choice"
    NOUL = "noul"
    SCORE = "score"


class ModelVariant(StrEnum):
    """Available Nirnaya model families."""

    FLASH = "flash"
    CORE = "core"


class SplitRole(StrEnum):
    """Permitted roles for a canonical dataset artifact."""

    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"
    EVAL_ONLY = "eval_only"
