"""Download the decision checkpoint into $MODEL_HOME unless it is already there; print its path.

DECISION_MODEL_ID and DECISION_MODEL_REVISION pick the checkpoint (the Dockerfile bakes in the defaults; setting
them on the Pod swaps the model without a rebuild). The checkpoint must be a protocol-v3 NVFP4 export:
joint_config.json, decision_config.json, readout.safetensors and the compressed-tensors shards.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_id = os.environ["DECISION_MODEL_ID"]
revision = os.environ.get("DECISION_MODEL_REVISION") or None
target = Path(os.environ.get("MODEL_HOME", "/workspace/models")) / model_id

print(f"Fetching {model_id}@{(revision or 'main')[:7]} into {target} (~25 GB the first time)", file=sys.stderr, flush=True)
snapshot_download(
    model_id,
    revision=revision,
    local_dir=target,
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "source/**"],
)
for name in ("decision_config.json", "joint_config.json", "readout.safetensors", "model.safetensors.index.json"):
    if not (target / name).is_file():
        sys.exit(f"{model_id} has no {name}; it is not a protocol-v3 (pplx-decider v1.2) decision checkpoint")
print(target)
