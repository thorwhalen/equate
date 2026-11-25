# equate.fusion

Fuse aligned records field by field, weighting sources by trust, and report conflicts.

The stage after `equate.align_sources()` (data fusion, or “truth discovery” when the
source weights are learned): for each item, each field’s candidate values are grouped
into **votes** (values that agree), each vote weighs the summed `trust` of the sources
behind it, and the heaviest vote wins. The value kept is the one from the most trusted
source in the winning vote, and its source is recorded as the field’s provenance.

A field is a **conflict** when a losing vote weighs at least `conflict_ratio` of the
winning one: trusted sources disagree (a field no source is trusted for never is).
Numbers agree only when equal by default: give timing fields a tolerance, e.g.
`agree={'start': lambda a, b: abs(a - b) < 0.1}`. A noisy, low-trust source contradicting a trusted
one is expected and not reported; two trusted sources disagreeing is exactly what a
person should check.

```pycon
>>> from equate import align_sources
>>> sheet = [{'text': 'bonjour', 'fr': 'hello'}, {'text': 'merci', 'fr': 'thanks'}]
>>> ocr = [{'text': 'bonjour'}, {'text': 'mercl'}]
>>> audio = [{'text': 'bonjou', 'start': 0.0}, {'text': 'merci', 'start': 1.5}]
>>> al = align_sources({'sheet': sheet, 'ocr': ocr, 'audio': audio})
>>> fused = fuse(al, trust={'audio': {'text': 0.3}})
>>> [item.values for item in fused]
[{'text': 'bonjour', 'fr': 'hello', 'start': 0.0},
 {'text': 'merci', 'fr': 'thanks', 'start': 1.5}]
>>> [(c.field, c.votes[1].sources) for c in fused[1].conflicts]
[('text', ['ocr'])]
```

### Module Attributes

| [`DFLT_CONFLICT_RATIO`](#equate.fusion.DFLT_CONFLICT_RATIO)   | a losing vote weighing at least this fraction of the winner makes the field a conflict   |
|------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [`DFLT_TRUST`](#equate.fusion.DFLT_TRUST)            | the trust of a source (or a source's field) that `trust` does not mention                |

### Functions

| [`fuse`](#equate.fusion.fuse)(groups, \*[, trust, agree, ...])   | Fuse each aligned item's records into one, field by field.                                 |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`default_agree`](#equate.fusion.default_agree)(a, b)                     | Whether two field values agree: `==`, strings compared case- and whitespace-insensitively. |

### Classes

| [`Fusion`](#equate.fusion.Fusion)(items, sources)                          | The result of [`fuse()`](#equate.fusion.fuse): one [`FusedItem`](#equate.fusion.FusedItem) per aligned item, in order.   |
|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`FusedItem`](#equate.fusion.FusedItem)(values, provenance[, conflicts, ...]) | One item, fused.                                                                                                                                                    |
| [`FieldVote`](#equate.fusion.FieldVote)(value, sources, weight)               | A value some sources agree on, and the summed trust behind it.                                                                                                      |
| [`FieldConflict`](#equate.fusion.FieldConflict)(field, votes)                     | A field whose trusted sources disagree.                                                                                                                             |

### equate.fusion.DFLT_CONFLICT_RATIO *= 0.5*

a losing vote weighing at least this fraction of the winner makes the field a conflict

### equate.fusion.DFLT_TRUST *= 1.0*

the trust of a source (or a source’s field) that `trust` does not mention

### *class* equate.fusion.FieldConflict(field, votes)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A field whose trusted sources disagree. `votes` is heaviest first; the first is
the value kept.

#### *property* value

The value kept.

### *class* equate.fusion.FieldVote(value, sources, weight)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A value some sources agree on, and the summed trust behind it.

### *class* equate.fusion.FusedItem(values, provenance, conflicts=list(), votes=dict(), members=dict(), missing=list(), margin=None, reordered=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One item, fused.

- `values`: `{field: value}`;
- `provenance`: `{field: source}`, the source each value was taken from;
- `conflicts`: the [`FieldConflict`](#equate.fusion.FieldConflict) fields;
- `votes`: `{field: [FieldVote, ...]}` for every field, heaviest first (what a
  trust estimator or a review screen reads);
- `members`: `{source: index}` of the records fused (from the alignment);
- `missing`: the sources that have no record for this item;
- `margin` / `reordered`: carried over from the alignment (low margin = unsure).

#### *property* needs_review

Whether the item has a conflict or is out of order (margins are the caller’s
to threshold: their scale depends on the comparators).

### *class* equate.fusion.Fusion(items, sources)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of [`fuse()`](#equate.fusion.fuse): one [`FusedItem`](#equate.fusion.FusedItem) per aligned item, in order.

#### *property* conflicts

`(item_index, FieldConflict)` for every conflict, in item order.

#### *property* unmatched

The items only one source has (the others lack them), when there are several
sources.

### equate.fusion.default_agree(a, b)

Whether two field values agree: `==`, strings compared case- and
whitespace-insensitively.

```pycon
>>> default_agree(' Merci  beaucoup', 'merci beaucoup')
True
```

### equate.fusion.fuse(groups, , trust=None, agree=None, conflict_ratio=0.5, fields=None)

Fuse each aligned item’s records into one, field by field.

`groups` is a [`SourceAlignment`](equate.sources.html.md#equate.sources.SourceAlignment) (from
`align_sources()`) or any iterable of its
[`SourceGroup`](equate.sources.html.md#equate.sources.SourceGroup) items.

- `trust`: `{source: weight}` or `{source: {field: weight}}`; anything unstated
  weighs [`DFLT_TRUST`](#equate.fusion.DFLT_TRUST) (so the default is equal trust). A weight of `0` keeps a
  source’s value only when no other source has the field.
- `agree`: when two values of a field agree: `None` ([`default_agree()`](#equate.fusion.default_agree)), a
  predicate `(a, b) -> bool`, or `{field: predicate}`.
- `conflict_ratio`: a field is a conflict when a losing vote weighs at least this
  fraction of the winning one.
- `fields`: the fields to fuse (default: every field of every record, in the order
  first seen).

Returns a [`Fusion`](#equate.fusion.Fusion): iterate it for [`FusedItem`](#equate.fusion.FusedItem) items; `.conflicts` and
`.unmatched` list what a person should look at.
