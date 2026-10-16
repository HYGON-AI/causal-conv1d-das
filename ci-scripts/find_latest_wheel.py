#!/usr/bin/env python3
"""Resolve the latest matching wheel requirement from the configured devpi simple index."""

import argparse
import html
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser
from urllib.parse import unquote

from packaging.version import InvalidVersion, Version

# 跟随当前解释器解析 ABI 标签，使同一脚本可用于 py3.10 / py3.12 并行 job。
PYTHON_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def build_timestamp(version_string: str) -> int:
    """Normalize supported build timestamp formats to YYYYMMDDHHMM."""
    version = Version(version_string)
    local = version.local or ""

    for part in local.split("."):
        if not part.isdigit():
            continue

        # Old format: 202604021232 -> 202604021232
        if len(part) == 12 and part.startswith("20"):
            return int(part)

        # New format: 2608070902 -> 202608070902
        if len(part) == 10:
            return int("20" + part)

    return 0


def wheel_versions(package: str, index_root: str):
    index = index_root.rstrip("/") + f"/{package}/"
    request = urllib.request.Request(
        index,
        headers={"User-Agent": "causal-conv1d-ci"},
    )
    page = urllib.request.urlopen(request, timeout=120).read().decode("utf-8")

    parser = LinkParser()
    parser.feed(page)

    pattern = re.compile(
        rf"{re.escape(package)}-([^-]+)-{PYTHON_TAG}-{PYTHON_TAG}-[^/]+\.whl$"
    )

    versions = set()
    for href in parser.hrefs:
        name = unquote(html.unescape(href.rsplit("/", 1)[-1].split("#", 1)[0]))
        match = pattern.match(name)
        if not match:
            continue

        try:
            version = Version(match.group(1))
        except InvalidVersion:
            continue

        versions.add(str(version))

    return index, versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", choices=("torch", "triton"))
    parser.add_argument("--public-version")
    parser.add_argument("--torch-version")
    parser.add_argument("--sort", choices=("build-time", "pep440"), required=True)
    args = parser.parse_args()

    index_root = os.environ["PIP_INDEX_URL"]
    index, candidates = wheel_versions(args.package, index_root)

    if args.package == "torch":
        if not args.public_version:
            parser.error("torch requires --public-version")
        candidates = {
            value for value in candidates if Version(value).public == args.public_version
        }
        if not candidates:
            raise SystemExit(
                f"No torch {args.public_version} {PYTHON_TAG} wheel found in {index}"
            )

    if args.package == "triton":
        if not args.torch_version:
            parser.error("triton requires --torch-version")
        torch_version = args.torch_version.split("+", 1)[0]
        torch_tag = re.sub(r"[^0-9]", "", torch_version)
        candidates = {
            value for value in candidates if f"torch{torch_tag}" in value
        }
        if not candidates:
            raise SystemExit(
                f"No Triton {PYTHON_TAG} wheel matching torch{torch_tag} "
                f"for TORCH_VERSION={torch_version} found in {index}"
            )

    if args.sort == "build-time":
        latest = max(
            candidates,
            key=lambda value: (build_timestamp(value), Version(value)),
        )
    else:
        latest = max(candidates, key=Version)

    print(f"{args.package}=={latest}")


if __name__ == "__main__":
    main()