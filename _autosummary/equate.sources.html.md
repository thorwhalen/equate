# equate.sources

Align several sources of the same items: one group per item, at most one record per source.

Several sources often describe the same items, each partially and each with its own kind
of error, and no shared identifier: a recording split into utterances (good timing, noisy
transcript), a printed sheet (exact text, no timing), an OCR or PDF extraction (exact
text, broken layout). [`align_sources()`](#equate.sources.align_sources) decides which records of the sources are the
same item; [`equate.fusion.fuse()`](equate.fusion.html.md#equate.fusion.fuse) then merges each item field by field and reports
where the sources disagree.

The alignment is **progressive** (the classic multiple-sequence-alignment strategy): the
first source is the backbone, and each further source is aligned against the groups built
so far. One step is a two-collection matching problem over a *gain* matrix,
`gain[g, j] = similarity(group g, record j) - threshold` (a record’s similarity to a
group is its best similarity to any member), solved by the `order` strategy:

- `'monotone'` (default): the sources list the items in the same order, gaps allowed.
  Solved by equate’s [`needleman_wunsch_alignment()`](equate.compare.sequence.html.md#equate.compare.sequence.needleman_wunsch_alignment) with the
  gain as substitution score and free gaps, so a record missing from one source becomes a
  gap instead of shifting everything after it. Records left over on both sides within
  `window` positions of each other are then paired by assignment when they match, which
  is how a **local order swap** (two items read in the opposite order) is recovered rather
  than cascading; such groups are marked `reordered`.
- `'free'`: no order; an optimal assignment (equate’s `'assign'` matcher). Any other
  matcher name from `equate.matchers` (`'greedy'`, `'stable'`) works the same way.
- any callable `gain ScoreMatrix -> iterable of (g, j)`: your own strategy.

Records are compared on the fields they share (`compare`), with equate’s comparators or
any callable, so domain knowledge (a phonetic reading, a unit normaliser) is injected, not
built in. A group’s `margin` is its score minus the best score a nearby rival record
would get in its place: low (or negative) where the alignment is unsure.

```pycon
>>> sheet = [{'text': 'bonjour'}, {'text': 'merci'}, {'text': 'au revoir'}]
>>> audio = [{'text': 'bonjou', 'start': 0.0}, {'text': 'au revoar', 'start': 2.1}]
>>> al = align_sources({'sheet': sheet, 'audio': audio})
>>> [g.members for g in al]
[{'sheet': 0, 'audio': 0}, {'sheet': 1}, {'sheet': 2, 'audio': 1}]
```

### Module Attributes

| [`DFLT_ALIGN_THRESHOLD`](#equate.sources.DFLT_ALIGN_THRESHOLD)   | minimum record similarity for two records to be the same item (strictly above it)   |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`DFLT_ORDER_WINDOW`](#equate.sources.DFLT_ORDER_WINDOW)      | how far (in positions) a `'monotone'` alignment looks for swapped items and rivals  |
| [`DFLT_FIELD_COMPARATOR`](#equate.sources.DFLT_FIELD_COMPARATOR)  | the comparator used for a shared string field when `compare` names no comparator    |

### Functions

| [`align_sources`](#equate.sources.align_sources)(sources, \*[, compare, order, ...])   | Align N sources of records into groups, one per item, at most one record per source.   |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`record_similarity`](#equate.sources.record_similarity)([compare, combine, missing])      | Build a record comparator `(a, b) -> float` from per-field comparators.                |
| [`monotone_order`](#equate.sources.monotone_order)(gain, \*[, window])                  | Order-preserving pairing of `gain`'s rows and columns, gaps free.                      |
| [`free_order`](#equate.sources.free_order)(gain)                                    | Pair rows and columns of `gain` by optimal assignment, keeping positive gains.         |

### Classes

| [`SourceGroup`](#equate.sources.SourceGroup)(members, records[, scores, ...])   | One item: the records the sources hold for it, at most one per source.                                     |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| [`SourceAlignment`](#equate.sources.SourceAlignment)(groups, sources[, order, ...]) | The result of [`align_sources()`](#equate.sources.align_sources): groups in alignment order. |

### equate.sources.DFLT_ALIGN_THRESHOLD *= 0.5*

minimum record similarity for two records to be the same item (strictly above it)

### equate.sources.DFLT_FIELD_COMPARATOR *= 'ratio'*

the comparator used for a shared string field when `compare` names no comparator

### equate.sources.DFLT_ORDER_WINDOW *= 2*

how far (in positions) a `'monotone'` alignment looks for swapped items and rivals

### *class* equate.sources.SourceAlignment(groups, sources, order='monotone', threshold=0.5)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of [`align_sources()`](#equate.sources.align_sources): groups in alignment order.

Iterates as its [`SourceGroup`](#equate.sources.SourceGroup) items; `sources` names every input source, in
order (so a consumer can tell which sources a group lacks).

#### missing(group)

The sources that have no record for `group`.

### *class* equate.sources.SourceGroup(members, records, scores=list(), score=None, margin=None, reordered=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One item: the records the sources hold for it, at most one per source.

- `members`: `{source: index}` into that source’s sequence, in source order;
- `records`: `{source: record}`, the records themselves;
- `scores`: the pairwise record similarities, as `(source_a, source_b, score)`;
- `score`: their mean (`None` for a single-source group);
- `margin`: `score` minus the best a nearby rival record would score in a member’s
  place, over all members (low or negative = unsure; `None` for a single record);
- `reordered`: the group crosses another one, i.e. some source lists this item out of
  the order the others do (a local swap).

JSON-able when the records are.

#### *property* sources

The sources holding this item.

### equate.sources.align_sources(sources, , compare=None, order='monotone', threshold=0.5, window=2)

Align N sources of records into groups, one per item, at most one record per source.

`sources` maps a source name to a sequence of records (dicts, or objects with
attributes). The first source is the backbone of the progressive alignment: put the
most complete one first.

- `compare`: how two records are scored (see [`record_similarity()`](#equate.sources.record_similarity)): `None`
  (shared string fields, `ratio`), `{field: comparator}` with equate’s comparator
  names or any callables, or one record callable `(a, b) -> float`. Similarities,
  higher = more alike, roughly in `[0, 1]`.
- `order`: `'monotone'` (same order, gaps allowed; default), `'free'`
  (optimal assignment), any matcher name from `equate.matchers` (`'greedy'`,
  `'stable'`, …), or a callable taking the gains as a
  [`ScoreMatrix`](equate.base.html.md#equate.base.ScoreMatrix) (`-inf` where two records share no compared
  > field) and returning `(group, record)` pairs; pairs without a positive gain are
  > dropped.
- `threshold`: two records are aligned only if their similarity is strictly above it.
  A record joins a group through its best similarity to the group’s members, so one
  noisy member does not keep it out.
- `window`: for `'monotone'`, how many positions away a swapped item is looked for;
  for `'monotone'` also how many positions away rival records are looked for when
  computing `margin` (other orders look everywhere). `None`: everywhere; `0`: no
  swap repair.

Returns a [`SourceAlignment`](#equate.sources.SourceAlignment) (iterate it for [`SourceGroup`](#equate.sources.SourceGroup) items).

A group holds **at most one record per source**: segmentation is the caller’s job. When
one source merges two items into a record (an utterance covering two lines) or splits
one over two, split or merge those records before aligning, or the extra one is left
as its own group.

With two sources and the items in the same order, this is `equate.match()` on the
shared fields, keeping pairs above `threshold`.

### equate.sources.free_order(gain)

Pair rows and columns of `gain` by optimal assignment, keeping positive gains.

```pycon
>>> free_order([[-1, 1], [1, -1]])
[(0, 1), (1, 0)]
```

### equate.sources.monotone_order(gain, , window=2)

Order-preserving pairing of `gain`’s rows and columns, gaps free.

A Needleman-Wunsch global alignment with `gain` as substitution score and zero gap
cost: the maximum-gain set of non-crossing pairs. Rows and columns it leaves unpaired
are then paired by assignment when they have positive gain and lie within `window`
positions of the gap they sit in (a local order swap); those pairs cross the others.
`window=0` keeps the alignment strictly monotone; `None` repairs at any distance.

```pycon
>>> monotone_order([[1, -1], [-1, 1]])
[(0, 0), (1, 1)]
>>> sorted(monotone_order([[-1, 1], [1, -1]]))  # a swap, recovered
[(0, 1), (1, 0)]
>>> monotone_order([[-1, 1], [1, -1]], window=0)
[(0, 1)]
```

### equate.sources.record_similarity(compare=None, \*, combine=<function mean>, missing=0.0)

Build a record comparator `(a, b) -> float` from per-field comparators.

`compare` is:

- `None` (default): every field both records hold as a string, scored with
  [`DFLT_FIELD_COMPARATOR`](#equate.sources.DFLT_FIELD_COMPARATOR) (two plain strings are compared directly);
- a mapping `{field: comparator}` (a registered name or a callable): those fields;
- a callable `(a, b) -> float`: returned as it is.

The per-field scores are a [`comparison_vector()`](equate.compare.vectorize.html.md#equate.compare.vectorize.comparison_vector) over
the fields **both records actually have** (a field missing or `None` on either side
is left out, not scored zero), reduced by `combine`: a combiner of
[`equate.compare.vectorize`](equate.compare.vectorize.html.md#module-equate.compare.vectorize), which takes the `{field: score}` mapping (the
default `mean`, `weighted_sum`, `fellegi_sunter`), or a reducer of numbers
(`max`, `min`, `sum`, `statistics.median`, …), which gets the scores. So a
source with extra fields is compared on what it shares. Records sharing no compared
field score `missing`.

```pycon
>>> sim = record_similarity({'fr': 'ratio', 'zh': 'ratio'}, combine=max)
>>> sim({'fr': 'merci', 'zh': '谢谢'}, {'fr': 'merci', 'zh': '鞋'})  # the better field
1.0
```

```pycon
>>> sim = record_similarity({'text': 'ratio'})
>>> sim({'text': 'merci'}, {'text': 'merci', 'pinyin': 'xiexie'})
1.0
>>> sim({'start': 1.0}, {'text': 'merci'})
0.0
```
