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
    """
    An example function that has been deprecated in favor of a new function.
    """
    pass

@deprecated(replacement="another_new_function_name")
def another_example_deprecate_renamed_function():
    """
    An example function that has been deprecated in favor of a new function.
    """
    pass


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

# ---------------------------------------------------------------------------- #


def test_api_deprecated_renamed_function():
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The function 'example_deprecate_renamed_function' is deprecated in"
            " favour of 'new_function_name', please update your code to use the"
            " new function name."
        ),
    ):
        example_deprecate_renamed_function()


def test_api_deprecation_emits_once_only():
    with pytest.warns(DeprecationWarning):
        another_example_deprecate_renamed_function()
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        another_example_deprecate_renamed_function()

def test_api_deprecated_renamed_parameters():
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The following parameters of 'example_deprecate_renamed_parameters'"
            " have been renamed:\n"
            "    fileName -> filename\n"
            "    beamMS -> beam_ms\n"
            "    checkDup -> check_dup\n"
            "    VOPosition -> vo_position\n"
            "    VORadius -> vo_radius\n"
            "This message will become an error in lsmtool version 1.9.0."
        ),
    ):
        result = example_deprecate_renamed_parameters(
            fileName="filename",
            beamMS="beam",
            checkDup=False,
            VOPosition=None,
            VORadius=1,
        )
        assert result == {
            "filename": "filename",
            "beam_ms": "beam",
            "check_dup": False,
            "vo_position": None,
            "vo_radius": 1,
        }
