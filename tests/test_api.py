"""
Test deprecating renamed functions and paramters.
"""

import warnings

import pytest

from lsmtool.api import deprecated
from lsmtool.io import load


# ---------------------------------------------------------------------------- #
# Fixtures
@pytest.fixture
def example_deprecate_renamed_function():
    """
    Fixture providing an example deprecated function.
    """

    @deprecated(replacement="new_function_name")
    def example_deprecate_renamed_function():
        """Example deprecated function."""
        return "test string from example_deprecate_renamed_function"

    # Return the decorated function as the fixture value
    return example_deprecate_renamed_function


class Example:
    """
    Example class with a deprecated method and a replacement method.
    """

    @deprecated(replacement="new_method_name")
    def example_deprecate_renamed_method(self):
        """Example deprecated method."""
        return "test string from Example.example_deprecate_renamed_method"

    def new_method_name(self):
        """Example replacement method."""
        return "test string from Example.new_method_name"


# ---------------------------------------------------------------------------- #
# Tests


def test_deprecated_renamed_function(example_deprecate_renamed_function):
    """
    Test that a deprecated function emits a deprecation warning.
    """

    with pytest.deprecated_call(
        match=(
            "The function 'example_deprecate_renamed_function' is deprecated. "
            "Please use the new function name 'new_function_name' instead."
        ),
    ):
        assert (
            example_deprecate_renamed_function()
            == "test string from example_deprecate_renamed_function"
        )


def test_deprecated_renamed_method():
    """
    Test that a deprecated function emits a deprecation warning.
    """

    with pytest.deprecated_call(
        match=(
            "The function 'example_deprecate_renamed_method' is deprecated."
            " Please use the new function name 'new_method_name' instead."
        ),
    ):
        assert (
            Example().example_deprecate_renamed_method()
            == "test string from Example.example_deprecate_renamed_method"
        )


def test_deprecation_emits_once_only(example_deprecate_renamed_function):
    """
    Test that a deprecated function emits a deprecation warning only once.
    """

    with pytest.deprecated_call():
        example_deprecate_renamed_function()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            example_deprecate_renamed_function()
            == "test string from example_deprecate_renamed_function"
        )


# ---------------------------------------------------------------------------- #


class TestAttributeDeprecation:
    """
    Test the deprecation of class attributes.
    """

    EXPECTED_MESSAGE = (
        "The attribute 'deprecatedAttribute' of 'ExampleDeprecateAttribute' is "
        "deprecated. Please use the new attribute name 'new_attribute' instead."
    )

    @pytest.fixture
    def example_deprecate_attribute(self):
        """
        Fixture providing an example class with a deprecated attribute.
        """

        class ExampleDeprecateAttribute:
            """Example class with a deprecated attribute."""

            deprecatedAttribute = deprecated("new_attribute")  # noqa
            new_attribute = "new value"

        # Return an instance of the example class with the deprecated attribute.
        return ExampleDeprecateAttribute()

    def test_get_deprecated_attribute(self, example_deprecate_attribute):
        """
        Test that accessing a deprecated attribute emits a deprecation warning.
        """
        with pytest.deprecated_call(match=self.EXPECTED_MESSAGE):
            result = example_deprecate_attribute.deprecatedAttribute

        assert result == "new value"

    def test_set_deprecated_attribute(self, example_deprecate_attribute):
        """
        Test that accessing a deprecated attribute emits a deprecation warning.
        """
        with pytest.deprecated_call(match=self.EXPECTED_MESSAGE):
            example_deprecate_attribute.deprecatedAttribute = "test value"

        assert example_deprecate_attribute.new_attribute == "test value"

    def test_deprecated_attribute_class_lookup(
        self, example_deprecate_attribute
    ):
        """
        Test that accessing a deprecated attribute emits a deprecation warning.
        """
        with pytest.deprecated_call(match=self.EXPECTED_MESSAGE):
            assert (
                example_deprecate_attribute.__class__.deprecatedAttribute
                == "new value"
            )


# ---------------------------------------------------------------------------- #


class TestDeprecatedParameters:
    """
    Test the deprecation of renamed parameters.
    """

    renamed_parameters = {
        "fileName": "filename",
        "beamMS": "beam_ms",
        "checkDup": "check_dup",
        "VOPosition": "vo_position",
        "VORadius": "vo_radius",
    }

    @pytest.fixture
    def example_deprecate_renamed_parameters(self):
        """
        Fixture providing an example function with deprecated parameter names.
        """

        @deprecated(
            renamed_parameters=TestDeprecatedParameters.renamed_parameters,
            target_version="1.x",
            warn_once=False,
        )
        def _example_deprecate_renamed_parameters(
            filename,
            beam_ms=None,
            check_dup=False,
            vo_position=None,
            vo_radius=None,
        ):
            """
            Example demonstrating parameter name deprecation.
            """

            # return the local namespace so we can check that the values were
            # correctly propagated
            return locals()

        # return the example function from the fixture
        return _example_deprecate_renamed_parameters

    def test_nominal_call(self, example_deprecate_renamed_parameters):
        """
        Test the nominal call of the function with the new parameter names.
        """
        result = example_deprecate_renamed_parameters(
            filename="filename",
            beam_ms="beam",
            check_dup=False,
            vo_position=None,
            vo_radius=1,
        )
        assert result == {
            "filename": "filename",
            "beam_ms": "beam",
            "check_dup": False,
            "vo_position": None,
            "vo_radius": 1,
        }

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                {
                    "fileName": "filename",
                    "beamMS": "beam",
                    "checkDup": False,
                    "VOPosition": None,
                    "VORadius": 1,
                },
                id="old spec",
            ),
            pytest.param(
                {
                    "fileName": "filename",
                    "beam_ms": "beam",
                    "check_dup": False,
                    "VOPosition": None,
                    "VORadius": 1,
                },
                id="mixed spec",
            ),
        ],
    )
    def test_deprecated_renamed_parameters(
        self, example_deprecate_renamed_parameters, params
    ):
        """
        Test that a function with deprecated parameter names emits a deprecation
        warning.

        Check that the values of the deprecated parameters are correctly mapped
        to the new names.
        """

        with pytest.deprecated_call(
            match=(
                "The following parameters of "
                "'_example_deprecate_renamed_parameters' have been renamed:"
                + "".join(
                    f"\n    {old} -> {new}"
                    for old, new in self.renamed_parameters.items()
                    if old in params
                )
                + "\nThis message will become an error in lsmtool version 1.x."
            )
        ):
            result = example_deprecate_renamed_parameters(**params)
            assert result == {
                "filename": "filename",
                "beam_ms": "beam",
                "check_dup": False,
                "vo_position": None,
                "vo_radius": 1,
            }

    def test_overspecified_raises(self, example_deprecate_renamed_parameters):
        """
        Test that providing both old and new parameter names raises a
        ValueError.
        """
        with pytest.raises(ValueError):
            example_deprecate_renamed_parameters(
                fileName="filename",
                filename="filename",
            )


# ---------------------------------------------------------------------------- #
# Test SkyModel deprecations


def test_skymodel_deprecations(pytestconfig):
    """
    Test that deprecated SkyModel methods emit deprecation warnings.
    """
    skymodel = load(pytestconfig.resource_dir / "to_patched.sky")

    with pytest.deprecated_call():
        result = skymodel.getPatchPositions("Patch1")

    # Call the new method should produce the same result without a deprecation
    # warning.
    new = skymodel.get_patch_positions("Patch1")

    # Check that the results are identical
    assert result == new
