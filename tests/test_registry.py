"""Tests for the generic name-keyed strategy Registry."""

import pytest

from equate.registry import Registry


def make_registry():
    reg = Registry("demo")

    @reg.register("double", meta={"note": "x2"})
    def _double_factory():
        return lambda x: x * 2

    reg.register("triple", lambda: (lambda x: x * 3))
    return reg


def test_register_create_and_names():
    reg = make_registry()
    assert reg.names() == ["double", "triple"]
    assert reg.create("double")(21) == 42
    assert reg.create("triple")(2) == 6


def test_meta_and_contains():
    reg = make_registry()
    assert "double" in reg and "nope" not in reg
    assert reg.meta("double") == {"note": "x2"}
    assert reg.meta("nope") is None


def test_resolve_callable_passthrough():
    reg = make_registry()
    assert reg.resolve(str.upper)("hi") == "HI"


def test_resolve_name_builds_via_factory():
    reg = make_registry()
    assert reg.resolve("double")(4) == 8


def test_unknown_name_raises_keyerror_listing_available():
    reg = make_registry()
    with pytest.raises(KeyError) as ei:
        reg.factory("missing")
    assert "double" in str(ei.value) and "triple" in str(ei.value)


def test_resolve_bad_type_raises_typeerror():
    reg = make_registry()
    with pytest.raises(TypeError):
        reg.resolve(123)


def test_factory_is_lazy_not_called_until_create():
    reg = Registry("demo")
    calls = []

    @reg.register("boom")
    def _factory():
        calls.append(1)
        return lambda x: x

    assert calls == []  # registering does not call the factory
    reg.create("boom")
    assert calls == [1]  # only create() calls it
