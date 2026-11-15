"""Validates constrained-decoding samples with Python's `jsonschema` (ADR-020).

Each input line is {"schema": ..., "text": ...}; every text must parse with `json` and
validate under Draft 2020-12. Used on the walks of `qubivex-grammar`'s schema test
(QUBIVEX_GRAMMAR_SAMPLES=<file>) and on model outputs.

    ~/.qubivex/oracle/bin/python fixtures/oracle/validate_json.py samples.jsonl
"""

import json
import sys

from jsonschema import Draft202012Validator


def main(path: str) -> int:
    n = bad = 0
    validators = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = json.dumps(row["schema"], sort_keys=True)
            v = validators.get(key)
            if v is None:
                Draft202012Validator.check_schema(row["schema"])
                v = validators[key] = Draft202012Validator(row["schema"])
            n += 1
            try:
                value = json.loads(row["text"])
            except ValueError as e:
                bad += 1
                print(f"does not parse: {row['text']!r}: {e}")
                continue
            errs = list(v.iter_errors(value))
            if errs:
                bad += 1
                print(f"invalid: {row['text']!r}: {errs[0].message}")
    print(f"{n} texts over {len(validators)} schemas, {bad} failures")
    return 1 if bad or n == 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
