# equate.evaluate.store

Where downloaded benchmark data lives: a store (a `MutableMapping`), never the package.

Benchmark loaders read their raw files through [`fetch()`](#equate.evaluate.store.fetch), which takes the bytes from a
*store* when they are there and otherwise downloads them once, checks them against a
pinned SHA-256, and writes them to the store. The store is any `MutableMapping[str,
bytes]` — a `dict` in tests, a `dol` store for S3 or a database — and defaults to
[`FileStore`](#equate.evaluate.store.FileStore) over [`data_dir()`](#equate.evaluate.store.data_dir) (`$EQUATE_DATA_DIR`, else
`$XDG_DATA_HOME/equate/benchmarks`, else `~/.local/share/equate/benchmarks`).

Downloads need nothing beyond the standard library; the `equate[benchmarks]` extra is
declared so that `pip install 'equate[benchmarks]'` stays the stable spelling as
loaders gain optional dependencies.

### Functions

| [`data_dir`](#equate.evaluate.store.data_dir)()                                     | The default root of the benchmark cache (not created until something is written).                                                                           |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`default_store`](#equate.evaluate.store.default_store)()                                | The default benchmark cache: a [`FileStore`](#equate.evaluate.store.FileStore) over [`data_dir()`](#equate.evaluate.store.data_dir). |
| [`fetch`](#equate.evaluate.store.fetch)(key, source, \*[, store, download, get]) | The bytes of `source`, from `store` if cached, else downloaded and cached.                                                                                  |
| [`http_get`](#equate.evaluate.store.http_get)(url, \*[, timeout])                   | Download `url` (stdlib `urllib`) and return its bytes.                                                                                                      |

### Classes

| [`Source`](#equate.evaluate.store.Source)(url[, sha256])   | A remote file: where it is, and the SHA-256 its bytes must have.        |
|--------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`FileStore`](#equate.evaluate.store.FileStore)(root)         | A `MutableMapping[str, bytes]` over the files under `root` (dol-style). |

### *class* equate.evaluate.store.FileStore(root)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

A `MutableMapping[str, bytes]` over the files under `root` (dol-style).

Keys are relative POSIX paths (`'wdc_products/80multi.zip'`) of plain names: no
`..`/`.`/empty segment, backslash, drive or `.part` suffix, so a key cannot escape
`root` on any platform. Writes are atomic even with concurrent writers (a unique
temporary file in the same directory, then a rename), so a reader never sees a
truncated file.

```pycon
>>> import tempfile
>>> s = FileStore(tempfile.mkdtemp())
>>> s['suite/file.csv'] = b'a,b'
>>> list(s), s['suite/file.csv']
(['suite/file.csv'], b'a,b')
```

### *class* equate.evaluate.store.Source(url, sha256=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A remote file: where it is, and the SHA-256 its bytes must have.

`sha256=None` skips the integrity check (for a file whose publisher re-issues it).

### equate.evaluate.store.data_dir()

The default root of the benchmark cache (not created until something is written).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### equate.evaluate.store.default_store()

The default benchmark cache: a [`FileStore`](#equate.evaluate.store.FileStore) over [`data_dir()`](#equate.evaluate.store.data_dir).

* **Return type:**
  [`FileStore`](#equate.evaluate.store.FileStore)

### equate.evaluate.store.fetch(key, source, , store=None, download=True, get=None)

The bytes of `source`, from `store` if cached, else downloaded and cached.

The bytes are checked against `source.sha256` **every time**, cached or not: a
cached copy that no longer matches (corrupted on disk, or a pin updated for a
re-issued file) is downloaded again; a download that does not match raises
`ValueError` and nothing is written. With `download=False` a missing or stale
key raises `FileNotFoundError` instead of reaching the network. `get(url) ->
bytes` is the downloader (default [`http_get()`](#equate.evaluate.store.http_get)).

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### equate.evaluate.store.http_get(url, , timeout=120)

Download `url` (stdlib `urllib`) and return its bytes.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)
