"""A tiny name-keyed strategy registry with lazy factories (open-closed dispatch).

Each pipeline stage (featurize, compare, block, match, cluster) owns a ``Registry``
mapping a string ``name`` to a lazy factory that builds the strategy on demand, so a
heavy optional import fires only when that strategy is actually used. Users select a
strategy by name (``featurize='tfidf'``) or pass a callable directly — :meth:`Registry.resolve`
accepts both, and users can register their own strategies without subclassing.
"""

from collections.abc import Callable
from typing import Any, Optional

__all__ = ['Registry']


class Registry:
    """A ``name -> lazy factory`` registry for one pipeline stage's strategies.

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
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict = {}  # name -> (factory, meta)

    def register(self, name: str, factory: Optional[Callable] = None, *, meta: Any = None):
        """Register ``factory`` under ``name``. Usable as a decorator or directly.

        The factory is called lazily (by :meth:`create`/:meth:`resolve`), so any heavy
        import inside it fires only on use. ``meta`` is optional declared metadata
        (e.g. a ``FeaturizerMeta``/``ComparatorMeta``).
        """
        def deco(f):
            self._entries[name] = (f, meta)
            return f

        return deco(factory) if factory is not None else deco

    def factory(self, name: str) -> Callable:
        """Return the (uncalled) factory registered under ``name``."""
        try:
            return self._entries[name][0]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {self.names()}"
            ) from None

    def meta(self, name: str) -> Any:
        """Return the declared metadata for ``name`` (``None`` if none was declared).

        Raises ``KeyError`` for an unknown name, consistent with :meth:`factory` — so
        an unknown name is never silently conflated with a known name that has no meta.
        """
        if name not in self._entries:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {self.names()}")
        return self._entries[name][1]

    def create(self, name: str, *args, **kwargs):
        """Build the strategy registered under ``name`` (calls its lazy factory)."""
        return self.factory(name)(*args, **kwargs)

    def resolve(self, spec, *args, **kwargs):
        """Resolve ``spec`` to a strategy.

        A callable ``spec`` passes through unchanged; a ``str`` is looked up and its
        factory is called with ``(*args, **kwargs)``.
        """
        if callable(spec):
            return spec
        if isinstance(spec, str):
            return self.create(spec, *args, **kwargs)
        raise TypeError(
            f"{self.kind} spec must be a registered name (str) or a callable, "
            f"got {type(spec).__name__}"
        )

    def names(self) -> list:
        """Sorted list of registered strategy names."""
        return sorted(self._entries)

    def __contains__(self, name) -> bool:
        return name in self._entries
