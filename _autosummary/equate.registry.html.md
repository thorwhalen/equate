# equate.registry

A tiny name-keyed strategy registry with lazy factories (open-closed dispatch).

Each pipeline stage (featurize, compare, block, match, cluster) owns a `Registry`
mapping a string `name` to a lazy factory that builds the strategy on demand, so a
heavy optional import fires only when that strategy is actually used. Users select a
strategy by name (`featurize='tfidf'`) or pass a callable directly — [`Registry.resolve()`](#equate.registry.Registry.resolve)
accepts both, and users can register their own strategies without subclassing.

### Classes

| [`Registry`](#equate.registry.Registry)(kind)   | A `name -> lazy factory` registry for one pipeline stage's strategies.   |
|-------------------------------------------------------------------|--------------------------------------------------------------------------|

### Exceptions

| [`UnknownStrategyError`](#equate.registry.UnknownStrategyError)   | No strategy is registered under that name (a `KeyError`, for back-compat).   |
|-------------------------------------------------------------------------|------------------------------------------------------------------------------|

### *class* equate.registry.Registry(kind)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A `name -> lazy factory` registry for one pipeline stage’s strategies.

```pycon
>>> reg = Registry('demo')
>>> @reg.register('double')
... def _double_factory():
...     return lambda x: x * 2
>>> reg.names()
['double']
>>> reg.create('double')(21)          # look up + build + call
42
>>> reg.resolve(str.upper)('hi')       # a callable passes through unchanged
'HI'
>>> 'double' in reg
True
```

#### create(name, \*args, \*\*kwargs)

Build the strategy registered under `name` (calls its lazy factory).

#### factory(name)

Return the (uncalled) factory registered under `name`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

#### meta(name)

Return the declared metadata for `name` (`None` if none was declared).

Raises `KeyError` for an unknown name, consistent with [`factory()`](#equate.registry.Registry.factory) — so
an unknown name is never silently conflated with a known name that has no meta.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

#### names()

Sorted list of registered strategy names.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

#### register(name, factory=None, , meta=None)

Register `factory` under `name`. Usable as a decorator or directly.

The factory is called lazily (by [`create()`](#equate.registry.Registry.create)/[`resolve()`](#equate.registry.Registry.resolve)), so any heavy
import inside it fires only on use. `meta` is optional declared metadata
(e.g. a `FeaturizerMeta`/`ComparatorMeta`).

#### resolve(spec, \*args, \*\*kwargs)

Resolve `spec` to a strategy.

A callable `spec` passes through unchanged; a `str` is looked up and its
factory is called with `(*args, **kwargs)`.

### *exception* equate.registry.UnknownStrategyError

Bases: [`KeyError`](https://docs.python.org/3/builtins/exceptions.html#KeyError)

No strategy is registered under that name (a `KeyError`, for back-compat).

Its own type lets a surface tell a caller’s unknown name (an HTTP 422) from a
`KeyError` raised by a bug.
