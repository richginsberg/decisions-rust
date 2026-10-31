#!/usr/bin/env bash
# Download the checkpoint to the volume on the first start, then serve it.
#
# The Pod's settings carry over from the Python server: AUTOJEV_API_KEY (required), DECISION_MODEL_ID /
# DECISION_MODEL_REVISION (the checkpoint), MODEL_HOME (the volume), PORT, and the DECISION_* limits that the
# README lists.
set -euo pipefail

AUTOJEV_CHECKPOINT=$(python3 -u /fetch_model.py)
export AUTOJEV_CHECKPOINT
exec /usr/local/bin/decision-server
