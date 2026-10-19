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
import tempfile
import time
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

__all__ = ["Source", "FileStore", "data_dir", "default_store", "fetch", "http_get"]

logger = logging.getLogger("equate.benchmarks")

DFLT_TIMEOUT = 120  # seconds, per download
_PART = ".part"  # suffix of in-progress writes (never a key)
_REPLACE_ATTEMPTS = 8  # os.replace retries (Windows sharing violations), ~1.3 s at most
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

    Keys are relative POSIX paths (``'wdc_products/80multi.zip'``) of plain names: no
    ``..``/``.``/empty segment, backslash, drive or ``.part`` suffix, so a key cannot escape
    ``root`` on any platform. Writes are atomic even with concurrent writers (a unique
    temporary file in the same directory, then a rename), so a reader never sees a
    truncated file.

    >>> import tempfile
    >>> s = FileStore(tempfile.mkdtemp())
    >>> s['suite/file.csv'] = b'a,b'
    >>> list(s), s['suite/file.csv']
    (['suite/file.csv'], b'a,b')
    """

    def __init__(self, root):
        self.root = Path(root).expanduser()

    def _path(self, key) -> Path:
        parts = key.split("/") if isinstance(key, str) else None
        if (
            not parts
            or any(p in ("", ".", "..") for p in parts)
            or "\\" in key
            or ":" in key
            or key.endswith(_PART)
            or PurePosixPath(key).is_absolute()
        ):
            raise KeyError(
                f"invalid store key {key!r}: use a relative path of plain names "
                f"inside the store"
            )
        return self.root.joinpath(*parts)

    def __getitem__(self, key) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def __setitem__(self, key, value: bytes):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=path.parent, prefix=path.name + ".", suffix=_PART
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(value)
            _replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def __delitem__(self, key):
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        path.unlink()

    def __iter__(self) -> Iterator[str]:
        if not self.root.is_dir():
            return
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and not p.name.endswith(_PART):
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


def _replace(tmp, path, *, attempts=_REPLACE_ATTEMPTS):
    """``os.replace`` that tolerates Windows' transient sharing violations.

    On Windows, replacing a file while another writer is replacing (or a reader holds)
    the same path raises ``PermissionError``. Retry with a short backoff; if it still
    fails while the file exists, a concurrent writer won — a valid last-writer-wins
    order — so drop this write rather than fail.
    """
    for k in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == attempts - 1:
                if Path(path).is_file():
                    Path(tmp).unlink(missing_ok=True)
                    return
                raise
            time.sleep(0.005 * 2**k)


def default_store() -> FileStore:
    """The default benchmark cache: a :class:`FileStore` over :func:`data_dir`."""
    return FileStore(data_dir())


def http_get(url: str, *, timeout: float = DFLT_TIMEOUT) -> bytes:
    """Download ``url`` (stdlib ``urllib``) and return its bytes."""
    import urllib.request  # here, not at import time: importing equate.evaluate stays cheap

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

    The bytes are checked against ``source.sha256`` **every time**, cached or not: a
    cached copy that no longer matches (corrupted on disk, or a pin updated for a
    re-issued file) is downloaded again; a download that does not match raises
    ``ValueError`` and nothing is written. With ``download=False`` a missing or stale
    key raises ``FileNotFoundError`` instead of reaching the network. ``get(url) ->
    bytes`` is the downloader (default :func:`http_get`).
    """
    store = default_store() if store is None else store
    if key in store:
        data = store[key]
        if _matches(data, source):
            return data
        logger.warning("cached %s does not match its pinned SHA-256; re-fetching", key)
    if not download:
        raise FileNotFoundError(
            f"{key!r} is not (validly) in the benchmark cache {store!r} and "
            f"download=False; fetch it from {source.url}"
        )
    logger.info("downloading %s -> %s", source.url, key)
    data = (get or http_get)(source.url)
    if not _matches(data, source):
        raise ValueError(
            f"{source.url} has SHA-256 {hashlib.sha256(data).hexdigest()}, expected "
            f"{source.sha256}: the publisher changed the file or the download is "
            f"corrupt; nothing was cached"
        )
    store[key] = data
    return data


def _matches(data: bytes, source: Source) -> bool:
    return source.sha256 is None or hashlib.sha256(data).hexdigest() == source.sha256
