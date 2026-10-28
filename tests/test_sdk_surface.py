"""Every Platform SDK method this repo calls must exist on the installed SDK.

On 2026-08-16 the 訓練 tab returned 500 in production:

    AttributeError: 'AsyncAccount' object has no attribute 'retrieve_summary'

Nothing was wrong with the code. `requirements.txt` said
`ultralytics-platform>=0.1.2`, the image happened to build after 0.1.5 shipped,
and 0.1.5 renamed six of the eleven methods this repo calls — retrieve_export
to export, list_images to images, retrieve_files to files, retrieve_training to
training, retrieve_file_url to signed_url, retrieve_summary to summary. Local
dev still had 0.1.2, so everything passed here and broke there.

The version is pinned now, but a pin only stops the accident; it does not tell
anyone when the assumption breaks. This does: it reads the call sites out of the
source and checks each one against whatever SDK is actually installed. Bump the
pin and this fails loudly, in CI, naming the methods to migrate.

No network, no credentials.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FAILS: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILS.append(label)


try:
    import ultralytics_platform
    from ultralytics_platform import AsyncPlatform, Platform
except ImportError as exc:  # pragma: no cover
    print("  FAIL  ultralytics-platform is not installed: %s" % exc)
    sys.exit(1)

import importlib.metadata as md  # noqa: E402

version = md.version("ultralytics-platform")
print("installed ultralytics-platform %s\n" % version)

# The resources the repo reaches for, and which client each caller uses:
# backend/services/platform.py is async (it runs inside FastAPI),
# tests/fetch_platform.py is the sync CLI.
SOURCES = {
    REPO / "backend" / "services" / "platform.py": "async",
    REPO / "backend" / "main.py": "async",
    REPO / "tests" / "fetch_platform.py": "sync",
}
CALL = re.compile(r"\.(account|datasets|projects|models|upload)\.([a-z_]+)\s*\(")

wanted: dict[str, set[tuple[str, str]]] = {"async": set(), "sync": set()}
for path, flavour in SOURCES.items():
    if not path.exists():
        continue
    for resource, method in CALL.findall(path.read_text()):
        wanted[flavour].add((resource, method))

print("call sites found: %d async, %d sync"
      % (len(wanted["async"]), len(wanted["sync"])))
check(bool(wanted["async"] or wanted["sync"]),
      "the source actually contains SDK calls — a regex that matches nothing "
      "would make this test vacuously green")


# The client attaches its resources in __init__ (`self.datasets = Datasets(...)`),
# so they are instance attributes and cannot be reached from the class. Read the
# assignment out of the source and look the class up by name instead — that needs
# no credentials and no network.
_ASSIGN = re.compile(r"self\.(\w+)\s*=\s*(\w+)\(")


def _resource_classes(client_cls) -> dict[str, type]:
    out = {}
    try:
        src = inspect.getsource(client_cls.__init__)
    except (OSError, TypeError):
        return out
    for attr, cls_name in _ASSIGN.findall(src):
        cls = getattr(ultralytics_platform.resources, cls_name, None)
        if cls is None:
            for mod in vars(ultralytics_platform.resources).values():
                cls = getattr(mod, cls_name, None)
                if inspect.isclass(cls):
                    break
                cls = None
        if inspect.isclass(cls):
            out[attr] = cls
    return out


def surface(client_cls, resource: str) -> set[str]:
    """Method names on a client's resource, without instantiating the client."""
    cls = _resource_classes(client_cls).get(resource)
    if cls is None:
        return set()
    return {n for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not n.startswith("_")}


for flavour, client in (("async", AsyncPlatform), ("sync", Platform)):
    if not wanted[flavour]:
        continue
    print("\n%s client (%s)" % (flavour, client.__name__))
    by_resource: dict[str, set[str]] = {}
    for resource, method in sorted(wanted[flavour]):
        by_resource.setdefault(resource, set()).add(method)
    for resource, methods in sorted(by_resource.items()):
        have = surface(client, resource)
        # A resource we cannot introspect is a FAILURE, not a skip: the first
        # version of this test silently skipped everything and reported ALL PASS
        # while checking nothing.
        check(bool(have), "%s.%s is introspectable" % (flavour, resource))
        if not have:
            continue
        missing = sorted(m for m in methods if m not in have)
        check(not missing,
              "%s.%s provides %s%s" % (
                  flavour, resource, ", ".join(sorted(methods)),
                  "" if not missing else "  — MISSING: %s  (have: %s)" % (
                      ", ".join(missing), ", ".join(sorted(have)))))

print("\n" + ("ALL PASS" if not FAILS
              else "FAILED: %d\n  - %s" % (len(FAILS), "\n  - ".join(FAILS))))
sys.exit(1 if FAILS else 0)
