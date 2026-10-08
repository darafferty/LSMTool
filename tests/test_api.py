"""
Test deprecating renamed functions and paramters.
"""

import warnings

import pytest

from lsmtool.api import deprecated

# ---------------------------------------------------------------------------- #
# Fixtures


@deprecated(replacement="new_function_name")
def example_deprecate_renamed_function():
    """Example deprecated function."""


class Example:
    @deprecated(replacement="new_method_name")
    def example_deprecate_renamed_method(self):
        """Example deprecated method."""

    def new_method_name(self):
        """Example replacement method."""


@deprecated(
    renamed_parameters={
        "fileName": "filename",
        "beamMS": "beam_ms",
        "checkDup": "check_dup",
        "VOPosition": "vo_position",
        "VORadius": "vo_radius",
    },
    target_version="1.9.0",
    warn_once=False,
)
def example_deprecate_renamed_parameters(
    filename,
    beam_ms=None,
    check_dup=False,
    vo_position=None,
    vo_radius=None,
):
    """
    Example demonstrating parameter name deprecation.
    """

    # return the local namespace so we can check that the values were correctly
    # propagated
    return locals()


# ---------------------------------------------------------------------------- #
# Tests


def test_deprecated_renamed_function():
    """
    Test that a deprecated function emits a deprecation warning.
    """

    with pytest.deprecated_call(
        match=(
            "The function 'example_deprecate_renamed_function' is deprecated. "
            "Please use the new function name 'new_function_name' instead."
        ),
    ):
        example_deprecate_renamed_function()


def test_deprecated_renamed_method():
    """
    Test that a deprecated function emits a deprecation warning.
    """

    with pytest.deprecated_call(
        match=(
            "The function 'example_deprecate_renamed_method' is deprecated. "
            "Please use the new function name 'new_method_name' instead."
        ),
    ):
        Example().example_deprecate_renamed_method()


def test_deprecation_emits_once_only():
    """
    Test that a deprecated function emits a deprecation warning only once.
    """

    @deprecated(replacement="new_function_name")
    def example_deprecate_renamed_function():
        pass

    with pytest.deprecated_call():
        example_deprecate_renamed_function()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        example_deprecate_renamed_function()


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
def test_deprecated_renamed_parameters(params):
    """
    Test that a function with deprecated parameter names emits a deprecation
    warning. Check that the values of the deprecated parameters are correctly
    mapped to the new names.
    """
    with pytest.deprecated_call(
        match=(
            "The following parameters of 'example_deprecate_renamed_parameters'"
            " have been renamed:"
            +("\n    fileName -> filename" if "fileName" in params else "")
            +("\n    beamMS -> beam_ms" if "beamMS" in params else "")
            +("\n    checkDup -> check_dup" if "checkDup" in params else "")
            +("\n    VOPosition -> vo_position" if "VOPosition" in params else "")
            +("\n    VORadius -> vo_radius" if "VORadius" in params else "")
            +"\nThis message will become an error in lsmtool version 1.9.0."
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


# ---------------------------------------------------------------------------- #


class TestAttributeDeprecation:
    """
    Test the deprecation of class attributes.
    """

    EXPECTED_MESSAGE = (
        "The attribute 'deprecatedAttribute'ExampleDeprecateAttribute of "
        "'ExampleDeprecateAttribute' is deprecated. Please use the new "
        "attribute name 'new_attribute' instead."
    )

    @pytest.fixture
    def example_deprecate_attribute(self):

        class ExampleDeprecateAttribute:
            deprecatedAttribute = deprecated("new_attribute")
            new_attribute = "new value"

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
