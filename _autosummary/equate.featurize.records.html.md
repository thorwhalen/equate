# equate.featurize.records

The text of an object, for text featurizers: a record (mapping) is its joined values.

Text featurizers (TF-IDF, embedders) need a string. A record — a dict from JSON, a CSV row
— has none of its own, so it is read as its non-empty values joined by spaces; anything
else as `str(obj)`. This is what lets `match(records_a, records_b)` work on the
default vector route.

### Functions

| [`record_text`](#equate.featurize.records.record_text)(obj)   | Text of `obj`: a mapping's non-empty values joined by spaces, else `str(obj)`.   |
|---------------------------------------------------------------------|----------------------------------------------------------------------------------|

### equate.featurize.records.record_text(obj)

Text of `obj`: a mapping’s non-empty values joined by spaces, else `str(obj)`.

```pycon
>>> record_text({'name': 'Jon Smith', 'city': 'Paris', 'zip': None})
'Jon Smith Paris'
>>> record_text(42)
'42'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
