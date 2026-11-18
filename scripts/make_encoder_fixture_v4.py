#!/usr/bin/env python3
"""Record what a version 4 (chunked) checkpoint's own encoder produces, so the Rust encoder can be checked token for token.

Usage: make_encoder_fixture_v4.py <checkpoint dir> <fixtures/encoder.json> > fixtures/encoder_v4.json

Unlike make_encoder_fixture.py (version 3), nothing of the protocol is reproduced here: the checkpoint's
`source/src/pplx_decider/protocol.py` is loaded and its `Encoder.plan(request, None, "separate")` is recorded, with the
checkpoint's processor (tokenizer, chat template, image processor). Needs transformers, torch (CPU is enough),
torchvision and Pillow, and from the checkpoint: the tokenizer and processor files, decision_config.json and source/.

The requests are the version 3 fixture's (its edge cases and labeled requests), so both protocols are checked on the
same inputs, plus cases for what version 4 adds. Image placeholders are replaced by real PNGs of several sizes; each
case records its images' patch grids, which the Rust test passes to the encoder as the server would.
"""

import base64
import importlib.util
import io
import json
import sys
import types
import typing
from pathlib import Path

try:  # the reference needs Python 3.11's typing.NotRequired; older interpreters take it from typing_extensions
    typing.NotRequired
except AttributeError:
    import typing_extensions

    typing.NotRequired = typing_extensions.NotRequired

from PIL import Image
from transformers import AutoProcessor

checkpoint, requests_path = Path(sys.argv[1]), sys.argv[2]

# Load pplx_decider.types and pplx_decider.protocol without the package's __init__ (which imports the model).
source = checkpoint / "source" / "src" / "pplx_decider"
package = types.ModuleType("pplx_decider")
package.__path__ = [str(source)]
sys.modules["pplx_decider"] = package
for name in ("types", "protocol"):
    spec = importlib.util.spec_from_file_location(f"pplx_decider.{name}", source / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
protocol = sys.modules["pplx_decider.protocol"]
if protocol.PROTOCOL["version"] != 4:
    sys.exit(f"{checkpoint} carries protocol version {protocol.PROTOCOL['version']}, not 4")

decision = json.load(open(checkpoint / "decision_config.json"))
max_pixels = decision.get("max_pixels") or protocol.LEGACY_MAX_PIXELS
processor = AutoProcessor.from_pretrained(checkpoint, min_pixels=protocol.MIN_PIXELS, max_pixels=max_pixels)
encoder = protocol.Encoder(processor, decision["codes"])
tokenizer = processor.tokenizer


def png(width: int, height: int, seed: int) -> str:
    image = Image.new("RGB", (width, height))
    image.putdata([((x * 7 + seed) % 256, (y * 5 + seed * 3) % 256, (x + y + seed) % 256) for y in range(height) for x in range(width)])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


SIZES = [(64, 64), (320, 200), (200, 480), (1024, 768)]

EXTRA_CASES = [
    {"id": "v4-separator-escaped", "state": "A literal <|fim_middle|> and <|fim_suffix|> in the state.",
     "questions": {"q": {"type": "noul", "instructions": "Does it mention <|fim_middle|>?"},
                   "r": {"type": "choice", "criteria": {"a<|fim_middle|>": "odd key", "b": None}}}},
    {"id": "v4-trailing-whitespace", "state": "ends with spaces and newlines   \n\n\t",
     "questions": {"q": {"type": "noul"}, "s": {"type": "score", "criteria": ["low", "high"]}}},
    {"id": "v4-leading-whitespace", "state": "\n\n  starts with whitespace",
     "questions": {"q": {"type": "noul", "instructions": "\n padded instructions \n"}}},
    {"id": "v4-empty-state", "state": "", "questions": {"q": {"type": "noul"}}},
    {"id": "v4-many-questions", "state": "Seven questions over one state.",
     "questions": {f"q{i}": {"type": "choice", "instructions": f"Question number {i}?", "criteria": {"yes": "Yes", "no": "No", "maybe": None}}
                   for i in range(7)}},
    {"id": "v4-three-images", "state": "Three pictures of different sizes.", "images": ["x", "x", "x"],
     "questions": {"q": {"type": "noul", "instructions": "Are they alike?"}, "r": {"type": "score", "criteria": ["a", "b", "c"]}}},
]

source_fixture = json.load(open(requests_path))
requests = [{"id": case["id"], **case["request"]} for case in source_fixture["cases"]] + EXTRA_CASES

cases = []
image_seed = 0
for item in requests:
    request = {key: value for key, value in item.items() if key != "id"}
    if request.get("images"):
        real = []
        for _ in request["images"]:
            width, height = SIZES[image_seed % len(SIZES)]
            real.append(png(width, height, image_seed))
            image_seed += 1
        request["images"] = real
    plan = encoder.plan(request, None, "separate")
    grids = [] if plan[0].image_grid_thw is None else [[int(v) for v in row] for row in plan[0].image_grid_thw.tolist()]
    cases.append({
        "id": item["id"],
        "request": request,
        "context_text": encoder.context_text(request),
        "question_texts": [encoder.question_text(question) for question in request["questions"].values()],
        "grids": grids,
        "sequences": [
            {"keys": list(encoded.keys), "counts": list(encoded.counts), "input_ids": encoded.input_ids.tolist(),
             "markers": encoded.markers.tolist(), "segments": list(encoded.segments)}
            for encoded in plan
        ],
    })

out = {
    "protocol": protocol.PROTOCOL,
    "system_prompt": protocol.SYSTEM_PROMPT,
    "escapes": encoder._escapes,
    "codes": decision["codes"],
    "marker_id": encoder.marker_id,
    "separator_id": encoder.separator_id,
    "pad_id": encoder.pad_id,
    "image_pad_id": tokenizer.convert_tokens_to_ids("<|image_pad|>"),
    "merge_size": processor.image_processor.merge_size,
    "cases": cases,
}
json.dump(out, sys.stdout, ensure_ascii=False)
print(f"{len(cases)} cases, {sum(len(c['sequences']) for c in cases)} sequences", file=sys.stderr)
