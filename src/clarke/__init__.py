"""clarke-flash: a System One extraction model built on Qwen3.5.

Single forward pass over a state + typed schema; answers are values read
from the input (as grounded spans), computed from it, or predicted inside
declared bounds — never generated tokens. See the README and
https://blog.cloudflare.com/clairic-extraction-model/ for the concept.
"""

from .encoding import EncodedRecord, collate_records, encode_record
from .head import JointExtractionHead
from .model import ClarkeModel, decode_answer, load_backbone, load_head, save_head
from .schema import FieldSpec

__all__ = [
    "ClarkeModel",
    "EncodedRecord",
    "FieldSpec",
    "JointExtractionHead",
    "collate_records",
    "decode_answer",
    "encode_record",
    "load_backbone",
    "load_head",
    "save_head",
]
