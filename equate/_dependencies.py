"""Optional-dependency helpers: import heavy libraries lazily, with a friendly error.

equate keeps its core install light (``numpy``/``scipy``) and draws every heavy
capability — fast string similarity, embeddings, ANN indexes, optimal transport,
graph algorithms, LLMs — behind a `pip` *extra* imported only on use. Strategy code
calls :func:`require` **inside** its factory/first-use so that a missing optional
dependency raises an actionable :class:`MissingDependencyError` naming the extra to
install, never a bare ``ImportError`` from a top-level import that would break
``import equate`` itself. See the ``equate-dev-add-strategy`` skill.
"""

import importlib
import importlib.util

__all__ = ["MissingDependencyError", "require", "have"]


class MissingDependencyError(ImportError):
    """An optional dependency needed by a strategy is not installed.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers still
    catch it, while carrying an actionable install hint in its message.
    """


def require(name, *, extra=None, purpose=None):
    """Import and return the module ``name``, or raise :class:`MissingDependencyError`.

    Use this instead of a bare ``import`` for any optional dependency, so a missing
    package produces an actionable message instead of a raw ``ImportError``.

    Args:
        name: the importable module name (e.g. ``'rapidfuzz'``, ``'networkx'``).
        extra: the equate extra that bundles it (``pip install 'equate[<extra>]'``).
        purpose: a short phrase naming what needs it, woven into the error message.

    Only a genuinely *absent* package becomes a :class:`MissingDependencyError`. If the
    package is present but fails to import for another reason (a missing transitive
    dependency, a broken C-extension), that real error is re-raised unchanged — the
    user is not misdirected to a useless reinstall.

    >>> require('numpy').__name__          # a present dependency imports normally
    'numpy'
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        # Convert to "not installed" only when the requested package itself is the
        # missing module — not when a *present* package's import fails because one of
        # ITS dependencies is absent (e.name would be that deeper module).
        if e.name not in (None, name, name.split(".")[0]):
            raise
        hint = f"pip install 'equate[{extra}]'" if extra else f"pip install {name}"
        because = f" for {purpose}" if purpose else ""
        raise MissingDependencyError(
            f"equate needs the optional package {name!r}{because}, which is not "
            f"installed. Install it with:  {hint}"
        ) from e


def have(name) -> bool:
    """Return whether optional dependency ``name`` is importable, without importing it.

    A cheap, side-effect-free capability probe for choosing among installed backends
    (e.g. prefer ``rapidfuzz`` over stdlib ``difflib`` when present). It checks the
    **top-level package** of ``name`` (a dotted name is reduced to its first
    component) *precisely so that no package code runs* — ``importlib.util.find_spec``
    on a dotted name would import the parent package to locate the submodule, which
    would defeat the light-import goal (``have('sklearn.metrics.pairwise')`` must not
    import sklearn).

    >>> have('numpy')
    True
    >>> have('no_such_pkg_xyz')
    False
    """
    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False
