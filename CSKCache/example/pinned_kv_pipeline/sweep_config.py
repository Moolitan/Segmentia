"""Configuration for the Skill-length/calibration-length sweep."""

from config import OUTPUT_ROOT


SKILL_TOKEN_VALUES = (1000, 3000, 5000, 8000)
CALIBRATION_TOKEN_VALUES = (8, 16, 32, 48, 64)
REPETITIONS = 1

# Pair C/R/I(layer) with the concurrently issued H2D(layer + 1).
# Python ranges exclude the stop value, so this selects layers 5--34.
STABLE_LAYER_START = 5
STABLE_LAYER_STOP = 35

SWEEP_OUTPUT_ROOT = OUTPUT_ROOT / "skill_calibration_sweep"
