from lsmtool.api import deprecated

import pytest


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
    pass


@deprecated(replacement="new_function_name")
def example_deprecate_renamed_function():
    pass


def test_api_deprecated_renamed_parameters():
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The following parameters of 'example_deprecate_renamed_parameters'"
            " have been renamed:\n"
            "    beamMS -> beam_ms\n"
            "    checkDup -> check_dup\n"
            "    VOPosition -> vo_position\n"
            "    VORadius -> vo_radius\n"
            "This message will become an error in lsmtool version 1.9.0."
        ),
    ):
        example_deprecate_renamed_parameters(
            "filename",
            beamMS=None,
            checkDup=False,
            VOPosition=None,
            VORadius=None,
        )


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
