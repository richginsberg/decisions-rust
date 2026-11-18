#!/usr/bin/env python3
"""Record what a v1.3 checkpoint's own Locate code produces, so the Rust port can be checked against it.

Usage: make_locate_fixture.py <checkpoint dir> > fixtures/locate.json

The checkpoint's `source/src/pplx_decider/locate.py` is loaded and its `LocateAdapter` is driven as `Decider.decide`
drives it (validate, _plan, _encode, _forward, answer), with the checkpoint's tokenizer and the real `SpanHead`
class. Only the model is replaced: the backbone returns recorded random hidden states of a small width, and the
span head has that width and seeded random weights, so the fixture stays small and the whole CPU side (the prompt,
token offsets, the head's scoring in FP32, the span clean-up) is the reference's own. Needs transformers, torch
(CPU), torchvision and Pillow, and from the checkpoint: the tokenizer and processor files, decision_config.json,
locate-span/locate_config.json and source/.
"""

import copy
import importlib.util
import json
import sys
import types
import typing
from pathlib import Path
from types import SimpleNamespace

try:
    typing.NotRequired
except AttributeError:
    import typing_extensions

    typing.NotRequired = typing_extensions.NotRequired

import torch
from transformers import AutoProcessor

checkpoint = Path(sys.argv[1])
source = checkpoint / "source" / "src" / "pplx_decider"
package = types.ModuleType("pplx_decider")
package.__path__ = [str(source)]
sys.modules["pplx_decider"] = package
# locate.py imports the Decider only for its type and the isolated layout, neither of which this script runs.
model_stub = types.ModuleType("pplx_decider.model")
model_stub.Decider = object
model_stub.question_layout = lambda chunks, padding: None
sys.modules["pplx_decider.model"] = model_stub
for name in ("types", "protocol", "locate"):
    spec = importlib.util.spec_from_file_location(f"pplx_decider.{name}", source / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
protocol, locate = sys.modules["pplx_decider.protocol"], sys.modules["pplx_decider.locate"]

config = json.loads((checkpoint / "locate-span" / "locate_config.json").read_text())
decision = json.loads((checkpoint / "decision_config.json").read_text())
HIDDEN, WIDTH, PROPOSALS = 16, 8, 6

# The adapter's tokenizer and encoder, built as LocateAdapter.__init__ builds them.
adapter = locate.LocateAdapter.__new__(locate.LocateAdapter)
processor = copy.deepcopy(AutoProcessor.from_pretrained(checkpoint))
tokenizer = processor.tokenizer
if locate.LOCATE_MARKER not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"extra_special_tokens": [locate.LOCATE_MARKER]}, replace_extra_special_tokens=False)
adapter.tokenizer = tokenizer
adapter.encoder = protocol.Encoder(processor, decision["codes"])
adapter.marker_id = adapter.encoder._single_token(locate.LOCATE_MARKER)
if adapter.marker_id != int(config["marker_id"]):
    sys.exit(f"The Locate marker got id {adapter.marker_id}, not the trained {config['marker_id']}")
if locate.LOCATE_MARKER not in adapter.encoder._escapes:
    sys.exit("User text could produce the Locate marker")
adapter.config = config
adapter.proposals = PROPOSALS
adapter.cutoff = float(config.get("thresholds", {}).get("default", 0.5))
adapter.embedding = SimpleNamespace(enabled=False)
adapter.adapters = []

generator = torch.Generator().manual_seed(20261118)
head = locate.SpanHead(HIDDEN, WIDTH, int(config["head"]["max_span"]))
with torch.no_grad():
    for parameter in head.parameters():
        parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.6)
    head.thresholds.bias[2] = -1.5  # so that a fair share of the candidate spans pass the cutoff
adapter.head = head.requires_grad_(False).eval()

hidden_states: list[list[float]] = []


def backbone(**inputs):
    length = inputs["input_ids"].shape[1]
    hidden = torch.randn((1, length, HIDDEN), generator=generator)
    hidden_states.append([round(v, 6) for v in hidden.flatten().tolist()])
    return SimpleNamespace(last_hidden_state=torch.tensor(hidden_states[-1]).view(1, length, HIDDEN))


adapter.decider = SimpleNamespace(max_length=None, device=torch.device("cpu"), isolation=SimpleNamespace(layout=None), backbone=backbone)

plans: list = []
original_plan = adapter._plan


def recording_plan(text, instructions):
    result = original_plan(text, instructions)
    plans.append((instructions, result))  # nested calls record first; the request's own plan is the last entry
    return result


LONG = ("Invoice 4471 from Acme Corp is due on March 3. Contact Jane Doe (jane@acme.example) or Bob Li at +1 555 0100. "
        "Ship to 12 Harbour Road, Leith; previous invoice 4460 was paid on 2026-01-17 by Globex GmbH. ")

CASES = [
    {"id": "simple", "state": LONG, "questions": {"company": {"type": "locate"}, "people": {"type": "locate", "instructions": "Person names"}}},
    {"id": "key-as-label", "state": "Alice met Bob in Paris.", "questions": {
        "city": {"type": "locate", "instructions": None}, "who": {"type": "locate", "instructions": ""},
        "spaces": {"type": "locate", "instructions": "  \n\t"}, "padded": {"type": "locate", "instructions": "  first names  "}}},
    {"id": "mixed-with-decisions", "state": LONG, "questions": {
        "kind": {"type": "noul", "instructions": "Is this an invoice?"}, "dates": {"type": "locate", "instructions": "Dates"},
        "urgency": {"type": "score", "criteria": ["low", "high"]}, "amounts": {"type": "locate"}}},
    {"id": "unicode", "state": "Zoë Müller (東京支社) paid €1.250,00 — naïve café ☕ and 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 🙂 at Ærøskøbing.",
     "questions": {"person": {"type": "locate"}, "place": {"type": "locate", "instructions": "Orte / 場所"}}},
    {"id": "leading-whitespace", "state": "\n\n  \tleading whitespace, then text.  \n", "questions": {"q": {"type": "locate", "instructions": "text"}}},
    {"id": "whitespace-only", "state": " \n ", "questions": {"q": {"type": "locate"}}},
    {"id": "single-char", "state": "x", "questions": {"q": {"type": "locate"}}},
    {"id": "special-tokens", "state": "Literal <|locate|> and <|fim_suffix|> and <|fim_middle|> and <|im_end|> stay text.",
     "questions": {"q": {"type": "locate", "instructions": "the <|locate|> marker"}}},
    {"id": "many-questions", "state": LONG * 2, "questions": {f"field_{i}": {"type": "locate", "instructions": f"Entity kind {i}"} for i in range(9)}},
    {"id": "long-state", "state": LONG * 12, "questions": {"company": {"type": "locate"}, "phone": {"type": "locate", "instructions": "Phone numbers"}}},
    {"id": "split-in-halves", "state": LONG, "max_length": 190,
     "questions": {f"f{i}": {"type": "locate", "instructions": f"Things of kind number {i} in the text"} for i in range(7)}},
    {"id": "too-long-alone", "state": LONG * 3, "max_length": 60, "questions": {"a": {"type": "locate"}, "b": {"type": "locate"}}},
    # Request rules.
    {"id": "error-images", "state": "text", "images": ["x"], "questions": {"q": {"type": "locate"}}},
    {"id": "error-json-state", "state": {"a": 1}, "questions": {"q": {"type": "locate"}}},
    {"id": "error-list-state", "state": ["a"], "questions": {"q": {"type": "locate"}}},
    {"id": "error-empty-state", "state": "", "questions": {"q": {"type": "locate"}}},
    {"id": "error-instructions-type", "state": "text", "questions": {"it's": {"type": "locate", "instructions": {"a": 1}}}},
    {"id": "error-instructions-list", "state": "text", "questions": {"q": {"type": "locate", "instructions": ["a"]}}},
    {"id": "error-too-many", "state": "text",
     "questions": {**{f"d{i}": {"type": "noul"} for i in range(100)}, **{f"l{i}": {"type": "locate"} for i in range(29)}}},
    {"id": "most-allowed", "state": "Alice and Bob.", "questions": {f"l{i}": {"type": "locate", "instructions": f"name {i}"} for i in range(128)}},
]

cases = []
for case in CASES:
    request = {key: value for key, value in case.items() if key not in ("id", "max_length")}
    questions = {key: question for key, question in request["questions"].items() if question.get("type") == "locate"}
    adapter.decider.max_length = case.get("max_length")
    plans.clear()
    hidden_states.clear()
    adapter._plan = recording_plan
    record = {"id": case["id"], "request": request, "max_length": case.get("max_length")}
    try:
        labels = adapter.validate(request, questions)
        result = adapter.answer(request, questions)
    except protocol.LengthError as error:
        record["length_error"] = str(error)
    except protocol.RequestError as error:
        record["request_error"] = str(error)
    else:
        record["labels"] = labels
        record["plans"] = [
            {"keys_from": sum(len(p[0].keys) for p in plans[-1][1][:i]), "input_ids": encoded.input_ids.tolist(),
             "markers": encoded.markers.tolist(), "segments": list(encoded.segments), "positions": positions,
             "offsets": [list(pair) for pair in offsets], "hidden": hidden}
            for i, ((encoded, positions, offsets), hidden) in enumerate(zip(plans[-1][1], hidden_states, strict=True))
        ]
        record["answers"] = result["answers"]
        record["usage"] = result["usage"]
    cases.append(record)

state = {name: [round(v, 8) for v in tensor.flatten().tolist()] for name, tensor in head.state_dict().items()}
out = {
    "format": config["format"],
    "marker_id": adapter.marker_id,
    "separator_id": adapter.encoder.separator_id,
    "pad_id": adapter.encoder.pad_id,
    "escapes": adapter.encoder._escapes,
    "codes": decision["codes"],
    "head": {"hidden": HIDDEN, "width": WIDTH, "max_span": head.max_span, "proposals": PROPOSALS, "cutoff": adapter.cutoff, "state": state},
    "cases": cases,
}
json.dump(out, sys.stdout, ensure_ascii=False)
spans = sum(len(a["spans"]) for c in cases for a in c.get("answers", {}).values())
print(f"{len(cases)} cases, {sum(len(c.get('plans', [])) for c in cases)} sequences, {spans} spans", file=sys.stderr)
