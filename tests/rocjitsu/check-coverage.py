#!/usr/bin/env python3
"""Enforce LLVM host line coverage for Gunrock's first-party host scope."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEVICE_ONLY_PREFIX = Path("include/gunrock/framework/operators")
DEVICE_ONLY_FILES = {
    Path("include/gunrock/util/load_store.hxx"),
    Path("include/gunrock/util/math.hxx"),
}
HOST_OPERATOR_FILES = {
    Path("include/gunrock/framework/operators/for/for.hxx"),
}
VENDORED_PREFIX = Path("include/gunrock/io/detail")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvm-cov", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--minimum", type=float, default=90.0)
    parser.add_argument("--output")
    return parser.parse_args()


def relative_to_root(filename, source_root):
    try:
        return Path(filename).resolve().relative_to(source_root)
    except ValueError:
        return None


def in_host_scope(relative_path):
    if relative_path is None:
        return False
    if not relative_path.is_relative_to(Path("include/gunrock")):
        return False
    if relative_path.is_relative_to(VENDORED_PREFIX):
        return False
    if relative_path in DEVICE_ONLY_FILES:
        return False
    if (
        relative_path.is_relative_to(DEVICE_ONLY_PREFIX)
        and relative_path not in HOST_OPERATOR_FILES
    ):
        return False
    return True


def main():
    args = parse_args()
    if args.minimum < 90.0:
        print("coverage error: minimum cannot be lower than 90%", file=sys.stderr)
        return 2
    source_root = Path(args.source_root).resolve()
    command = [
        args.llvm_cov,
        "export",
        "-summary-only",
        f"-instr-profile={args.profile}",
        args.binary,
    ]
    completed = subprocess.run(
        command, check=True, stdout=subprocess.PIPE, text=True
    )
    export = json.loads(completed.stdout)

    files = []
    for entry in export["data"][0]["files"]:
        relative_path = relative_to_root(entry["filename"], source_root)
        if not in_host_scope(relative_path):
            continue
        lines = entry["summary"]["lines"]
        if lines["count"] == 0:
            continue
        files.append(
            {
                "path": relative_path.as_posix(),
                "covered": lines["covered"],
                "count": lines["count"],
                "percent": lines["percent"],
            }
        )

    covered = sum(entry["covered"] for entry in files)
    count = sum(entry["count"] for entry in files)
    if count == 0:
        print("coverage error: no first-party host lines were mapped", file=sys.stderr)
        return 2

    percent = covered * 100.0 / count
    report = {
        "scope": "first-party LLVM host mappings",
        "covered_lines": covered,
        "mapped_lines": count,
        "line_coverage_percent": round(percent, 2),
        "minimum_percent": args.minimum,
        "files": sorted(files, key=lambda entry: entry["path"]),
        "excluded": [
            "tests, examples, benchmarks, generated files, and external dependencies",
            "vendored NIST Matrix Market helpers under include/gunrock/io/detail",
            "AMD device-only operators and device primitives without host counters",
        ],
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"host line coverage: {covered}/{count} = {percent:.2f}% "
        f"(minimum {args.minimum:.2f}%)"
    )
    if percent + 1.0e-9 < args.minimum:
        print("coverage threshold not met", file=sys.stderr)
        return 1
    print("coverage threshold met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
