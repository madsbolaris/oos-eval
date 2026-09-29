#!/usr/bin/env python3
"""Convert one CLINC150 domain into deterministic, labelled Kev partitions."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "model.json"
DEFAULT_OUTPUT = ROOT / "generated" / "kev"
PARTITIONS = ("train", "calibration", "development", "test")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_state(state):
    return " ".join(state.casefold().split())


def state_hash(state):
    return hashlib.sha256(normalized_state(state).encode()).hexdigest()


def option_description(label):
    if label == "oos":
        return "The request is outside the supported travel intents"
    return "Requests about " + label.replace("_", " ")


def question(spec, labels, label, source_split):
    question_spec = spec["question"]
    return {
        "type": question_spec["type"],
        "instructions": question_spec["instructions"],
        "criteria": {name: option_description(name) for name in labels},
        "label": label,
    }


def record(spec, labels, state, label, source_split, source_index):
    return {
        "state": state,
        "questions": {
            spec["question"]["id"]: question(spec, labels, label, source_split)
        },
        "_meta": {
            "id": f"clinc150/{source_split}/{source_index}",
            "source": "clinc150",
            "source_revision": spec["source"]["revision"],
            "domain": spec["domain"],
            "original_split": source_split,
            "text_sha256": state_hash(state),
        },
    }


def split_validation(rows):
    """Split each label evenly so calibration and development remain balanced."""
    by_label = defaultdict(list)
    for row in rows:
        label = next(iter(row["questions"].values()))["label"]
        by_label[label].append(row)

    calibration, development = [], []
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda row: row["_meta"]["text_sha256"])
        midpoint = len(ordered) // 2
        calibration.extend(ordered[:midpoint])
        development.extend(ordered[midpoint:])
    return calibration, development


def build(spec_path=DEFAULT_SPEC):
    spec = read_json(spec_path)
    source = read_json(ROOT / spec["source"]["data_file"])
    domains = read_json(ROOT / spec["source"]["domain_file"])
    domain_labels = domains[spec["domain"]]
    labels = [*domain_labels, spec["question"]["out_of_scope"]]
    allowed = set(labels)

    def convert(source_split, oos_split):
        selected = []
        label_counts = Counter()
        for index, (state, label) in enumerate(source[source_split]):
            if label in allowed and label != spec["question"]["out_of_scope"]:
                selected.append(record(spec, labels, state, label, source_split, index))
                label_counts[label] += 1
        counts = set(label_counts.values())
        if set(label_counts) != set(domain_labels) or len(counts) != 1:
            raise ValueError(f"{source_split} is not balanced across the selected domain")
        oos_limit = counts.pop()
        offset = len(source[source_split])
        for index, (state, label) in enumerate(source[oos_split][:oos_limit], offset):
            if label != spec["question"]["out_of_scope"]:
                raise ValueError(f"unexpected label {label!r} in {oos_split}")
            selected.append(record(spec, labels, state, label, oos_split, index))
        return selected

    train = convert("train", "oos_train")
    validation = convert("val", "oos_val")
    calibration, development = split_validation(validation)
    test = convert("test", "oos_test")
    partitions = {
        "train": train,
        "calibration": calibration,
        "development": development,
        "test": test,
    }
    validate(partitions, labels, spec["question"]["id"])
    return spec, partitions


def validate(partitions, labels, question_id):
    seen = {}
    expected = set(labels)
    for partition, rows in partitions.items():
        if not rows:
            raise ValueError(f"{partition} is empty")
        present = Counter()
        for row in rows:
            state = row.get("state")
            if not isinstance(state, str) or not state.strip():
                raise ValueError(f"{partition} contains an empty state")
            key = state_hash(state)
            if key in seen:
                raise ValueError(f"state appears in both {seen[key]} and {partition}: {state!r}")
            seen[key] = partition
            question_data = row.get("questions", {}).get(question_id)
            if not question_data or set(question_data["criteria"]) != expected:
                raise ValueError(f"{partition} record has inconsistent criteria")
            if question_data["label"] not in expected:
                raise ValueError(f"{partition} record has unknown label {question_data['label']!r}")
            present[question_data["label"]] += 1
        missing = expected - set(present)
        if missing:
            raise ValueError(f"{partition} is missing labels: {sorted(missing)}")


def encode_jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def sha256(content):
    return hashlib.sha256(content.encode()).hexdigest()


def render(spec, partitions):
    files = {f"{name}.jsonl": encode_jsonl(partitions[name]) for name in PARTITIONS}
    question_id = spec["question"]["id"]
    manifest = {
        "format": "kev-labelled-requests-v1",
        "source": spec["source"],
        "domain": spec["domain"],
        "locked": ["test"],
        "sampling": "All selected-domain rows; out-of-scope rows capped to one class per native split in source order.",
        "partitions": {},
    }
    for name in PARTITIONS:
        labels = Counter(
            row["questions"][question_id]["label"] for row in partitions[name]
        )
        manifest["partitions"][name] = {
            "records": len(partitions[name]),
            "labels": dict(sorted(labels.items())),
            "sha256": sha256(files[f"{name}.jsonl"]),
        }
    files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return files


def write(output, files):
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8", newline="\n")


def check(output, files):
    problems = []
    for name, expected in files.items():
        path = output / name
        if not path.exists():
            problems.append(f"missing {path}")
        elif path.read_text(encoding="utf-8") != expected:
            problems.append(f"stale {path}")
    extras = sorted(path.name for path in output.glob("*.json*") if path.name not in files)
    problems.extend(f"unexpected {output / name}" for name in extras)
    if problems:
        raise SystemExit("generated data is not current:\n  " + "\n  ".join(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    spec, partitions = build(args.spec)
    files = render(spec, partitions)
    if args.check:
        check(args.output, files)
    else:
        write(args.output, files)
    counts = ", ".join(f"{name}={len(partitions[name])}" for name in PARTITIONS)
    print(f"ok: {counts}")


if __name__ == "__main__":
    main()