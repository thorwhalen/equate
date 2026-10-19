"""Where downloaded benchmark data lives: a store (a ``MutableMapping``), never the package.

Benchmark loaders read their raw files through :func:`fetch`, which takes the bytes from a
*store* when they are there and otherwise downloads them once, checks them against a
pinned SHA-256, and writes them to the store. The store is any ``MutableMapping[str,
bytes]`` — a ``dict`` in tests, a ``dol`` store for S3 or a database — and defaults to
:class:`FileStore` over :func:`data_dir` (``$EQUATE_DATA_DIR``, else
``$XDG_DATA_HOME/equate/benchmarks``, else ``~/.local/share/equate/benchmarks``).

Downloads need nothing beyond the standard library; the ``equate[benchmarks]`` extra is
declared so that ``pip install 'equate[benchmarks]'`` stays the stable spelling as
loaders gain optional dependencies.
"""

import hashlib
import logging
import os
import urllib.request
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

__all__ = ["Source", "FileStore", "data_dir", "default_store", "fetch", "http_get"]

logger = logging.getLogger("equate.benchmarks")

DFLT_TIMEOUT = 120  # seconds, per download
_USER_AGENT = "equate-benchmarks (https://github.com/thorwhalen/equate)"


@dataclass(frozen=True)
class Source:
    """A remote file: where it is, and the SHA-256 its bytes must have.

    ``sha256=None`` skips the integrity check (for a file whose publisher re-issues it).
    """

    url: str
    sha256: Optional[str] = None


def data_dir() -> Path:
    """The default root of the benchmark cache (not created until something is written)."""
    if os.environ.get("EQUATE_DATA_DIR"):
        return Path(os.environ["EQUATE_DATA_DIR"]).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base).expanduser() / "equate" / "benchmarks"


class FileStore(MutableMapping):
    """A ``MutableMapping[str, bytes]`` over the files under ``root`` (dol-style).

    Keys are relative POSIX paths (``'wdc_products/80multi.zip'``); a key cannot escape
    ``root``. Writes are atomic (a temporary file, then a rename), so an interrupted
    download never leaves a truncated file that a later read would trust.

    >>> import tempfile
    >>> s = FileStore(tempfile.mkdtemp())
    >>> s['suite/file.csv'] = b'a,b'
    >>> list(s), s['suite/file.csv']
    (['suite/file.csv'], b'a,b')
    """

    def __init__(self, root):
        self.root = Path(root).expanduser()

    def _path(self, key) -> Path:
        rel = PurePosixPath(key)
        if not key or rel.is_absolute() or ".." in rel.parts:
            raise KeyError(
                f"invalid store key {key!r}: use a relative path inside the store"
            )
        return self.root.joinpath(*rel.parts)

    def __getitem__(self, key) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def __setitem__(self, key, value: bytes):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(value)
        os.replace(tmp, path)

    def __delitem__(self, key):
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        path.unlink()

    def __iter__(self) -> Iterator[str]:
        if not self.root.is_dir():
            return
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and not p.name.endswith(".part"):
                yield p.relative_to(self.root).as_posix()

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key):
        try:
            return self._path(key).is_file()
        except KeyError:
            return False

    def __repr__(self):
        return f"{type(self).__name__}({str(self.root)!r})"


def default_store() -> FileStore:
    """The default benchmark cache: a :class:`FileStore` over :func:`data_dir`."""
    return FileStore(data_dir())


def http_get(url: str, *, timeout: float = DFLT_TIMEOUT) -> bytes:
    """Download ``url`` (stdlib ``urllib``) and return its bytes."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def fetch(
    key: str,
    source: Source,
    *,
    store: Optional[MutableMapping] = None,
    download: bool = True,
    get: Optional[Callable[[str], bytes]] = None,
) -> bytes:
    """The bytes of ``source``, from ``store`` if cached, else downloaded and cached.

    A download is checked against ``source.sha256`` before it is stored: a mismatch
    raises ``ValueError`` and nothing is written (the publisher changed the file, or the
    download was corrupted). With ``download=False`` a missing key raises
    ``FileNotFoundError`` instead of reaching the network. ``get(url) -> bytes`` is the
    downloader (default :func:`http_get`).
    """
    store = default_store() if store is None else store
    if key in store:
        return store[key]
    if not download:
        raise FileNotFoundError(
            f"{key!r} is not in the benchmark cache {store!r} and download=False; "
            f"fetch it from {source.url}"
        )
    logger.info("downloading %s -> %s", source.url, key)
    data = (get or http_get)(source.url)
    if source.sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest != source.sha256:
            raise ValueError(
                f"{source.url} has SHA-256 {digest}, expected {source.sha256}: the "
                f"publisher changed the file or the download is corrupt; nothing was cached"
            )
    store[key] = data
    return data
