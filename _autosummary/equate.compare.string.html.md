# equate.compare.string

String comparators (stage ②): pairwise similarity between two strings.

Core comparators use only the standard library (`difflib` and a pure-python edit
distance); faster/richer ones (rapidfuzz, jellyfish, py_stringmatching) are optional
extras imported lazily via `equate._dependencies.require()`. Each carries a
[`equate.base.ComparatorMeta`](equate.base.html.md#equate.base.ComparatorMeta) declaring its polarity, boundedness, whether it is a
metric, and whether it is symmetric — so the framework can adapt (never treat a
non-metric as a metric, keep directional comparators directional).

### Functions

| [`ratio`](#equate.compare.string.ratio)(a, b)                            | `difflib.SequenceMatcher` ratio — a bounded [0, 1] similarity (stdlib, core).   |
|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`levenshtein_distance`](#equate.compare.string.levenshtein_distance)(a, b)             | Levenshtein edit distance (pure python, core) — a true metric, unbounded.       |
| [`levenshtein`](#equate.compare.string.levenshtein)(a, b)                      | Length-normalized Levenshtein *similarity* in [0, 1] (core).                    |
| [`monge_elkan`](#equate.compare.string.monge_elkan)(a, b, \*[, sim, tokenize]) | Monge-Elkan hybrid token similarity — **directional** (`ME(a,b) != ME(b,a)`).   |
| [`jaro_winkler`](#equate.compare.string.jaro_winkler)(a, b)                     | Jaro-Winkler similarity via `rapidfuzz` — requires `equate[fuzzy]`.             |
| [`phonetic_match`](#equate.compare.string.phonetic_match)(a, b, \*[, algorithm])  | Boolean phonetic-code agreement via `jellyfish` — requires `equate[phonetic]`.  |

### equate.compare.string.jaro_winkler(a, b)

Jaro-Winkler similarity via `rapidfuzz` — requires `equate[fuzzy]`.

### equate.compare.string.levenshtein(a, b)

Length-normalized Levenshtein *similarity* in [0, 1] (core).

### equate.compare.string.levenshtein_distance(a, b)

Levenshtein edit distance (pure python, core) — a true metric, unbounded.

### equate.compare.string.monge_elkan(a, b, \*, sim=<function ratio>, tokenize=<method 'split' of 'str' objects>)

Monge-Elkan hybrid token similarity — **directional** (`ME(a,b) != ME(b,a)`).

Each token of `a` is scored by its best match among the tokens of `b` (via the
inner `sim`), then averaged over `a`’s tokens. Kept directional on purpose (see
decision register D3); symmetrize only at the matcher boundary if a scalar is needed.

### equate.compare.string.phonetic_match(a, b, , algorithm='metaphone')

Boolean phonetic-code agreement via `jellyfish` — requires `equate[phonetic]`.

Returns 1.0 when the two strings share a phonetic code (`metaphone` by default,
or `soundex` / `nysiis`), else 0.0.

### equate.compare.string.ratio(a, b)

`difflib.SequenceMatcher` ratio — a bounded [0, 1] similarity (stdlib, core).
