"""
Test deprecating renamed functions and paramters.
"""

import warnings

import pytest

from lsmtool.api import deprecated, deprecated_attribute


# ---------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture
def example_deprecate_renamed_function():
    @deprecated(replacement="new_function_name")
    def example_deprecate_renamed_function():
        """
        An example function that has been deprecated in favor of a new function.
        """
        pass

    return example_deprecate_renamed_function


@pytest.fixture
def example_deprecate_renamed_parameters():
    """
    Fixture defining an example function that has deprecated parameter names.
    """

    @deprecated(
        renamed_parameters={
            "fileName": "filename",
            "beamMS": "beam_ms",
            "checkDup": "check_dup",
            "VOPosition": "vo_position",
            "VORadius": "vo_radius",
        },
        target_version="1.9.0",
    )
    def example_deprecate_renamed_parameters(
        filename,
        beam_ms=None,
        check_dup=False,
        vo_position=None,
        vo_radius=None,
    ):
        """
        An example function that has deprecated parameter names.
        """

        # return the local namespace so we can check that the values were correctly
        # propagated
        return locals()

    return example_deprecate_renamed_parameters


# ---------------------------------------------------------------------------- #
# Tests


def test_api_deprecated_renamed_function(example_deprecate_renamed_function):
    """
    Test that a deprecated function emits a deprecation warning.
    """
    with pytest.deprecated_call(
        match=(
            "The function 'example_deprecate_renamed_function' is deprecated in"
            " favour of 'new_function_name', please update your code to use the"
            " new function name."
        ),
    ):
        example_deprecate_renamed_function()


def test_api_deprecation_emits_once_only(example_deprecate_renamed_function):
    """
    Test that a deprecated function emits a deprecation warning only once.
    """
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
        pytest.param(
            {
                "filename": "filename",
                "beam_ms": "beam",
                "check_dup": False,
                "vo_position": None,
                "vo_radius": 1,
            },
            id="new spec",
        ),
    ],
)
def test_api_deprecated_renamed_parameters(
    example_deprecate_renamed_parameters, params
):
    """
    Test that a function with deprecated parameter names emits a deprecation
    warning. Check that the values of the deprecated parameters are correctly
    mapped to the new names.
    """
    with pytest.deprecated_call(
        match=(
            "The following parameters of 'example_deprecate_renamed_parameters'"
            " have been renamed:\n"
            "    fileName -> filename\n"
            if params.get("fileName")
            else "    beamMS -> beam_ms\n"
            if params.get("beamMS")
            else "    checkDup -> check_dup\n"
            if params.get("checkDup")
            else "    VOPosition -> vo_position\n"
            if params.get("VOPosition")
            else "    VORadius -> vo_radius\n"
            if params.get("VORadius")
            else "This message will become an error in lsmtool version 1.9.0."
        ),
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
        "The 'deprecatedAttribute' attribute of 'ExampleDeprecateAttribute'"
        " is deprecated. Please use the new attribute name 'new_attribute' "
        "instead."
    )

    @pytest.fixture
    def example_deprecate_attribute(self):

        class ExampleDeprecateAttribute:
            deprecatedAttribute = deprecated_attribute("new_attribute")
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
