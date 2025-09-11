"""
Filter and group a sky model based on image data.

This module provides functionality to filter and refine sky models using image
data and source finders like PyBDSF or SoFiA-2. It allows for the creation of
both apparent and true sky models, taking into account primary beam correction.

There are two submodules: :py:mod:`.bdsf` and :py:mod:`.sofia`, implementing
the sky model filtering with `pyBDSF
<https://pybdsf.readthedocs.io/en/stable/>`_ and `SoFiA-2
<https://gitlab.com/SoFiA-Admin/SoFiA-2/-/wikis/documents/SoFiA-2_User_Manual.pdf>`_
respectively.
"""

import contextlib as ctx

from ..io import (
    PathLike,
    PathLikeOptional,
    PathLikeOrListOptional,
)
from . import bdsf

# Module variable tracking available source finders
KNOWN_SOURCE_FINDERS = {"bdsf": bdsf}

with ctx.suppress(ModuleNotFoundError):
    from . import sofia

    KNOWN_SOURCE_FINDERS["sofia"] = sofia


def filter_skymodel(
    flat_noise_image: PathLike,
    true_sky_image: PathLikeOptional,
    input_true_skymodel: PathLikeOptional,
    input_apparent_skymodel: PathLikeOptional,
    output_apparent_sky: PathLike,
    output_true_sky: PathLike,
    beam_ms: PathLikeOrListOptional,
    input_bright_skymodel: PathLikeOptional = None,
    *,
    source_finder: str = "bdsf",
    **kws,
):
    """
    Filters a sky model based on a source finder.

    This function filters a sky model using either SoFiA-2 or PyBDSF, based on
    the `source_finder` parameter.  It applies the chosen source finder to
    generate a filtered sky model.

    Parameters
    ----------
    flat_noise_image : str or pathlib.Path
        Filename of input image to use to detect sources for filtering. It
        should be a flat-noise / apparent sky image (without primary-beam
        correction).
    true_sky_image : str or pathlib.Path or None
        Filename of input image to use to determine the true flux of sources.
        It should be a true flux image (with primary-beam correction).

        - If `beam_ms` is given and exists, this parameter is ignored and the
          `flat_noise_image` is used instead.
        - If beam_ms is None or an empty string, this argument must be
          supplied.

    input_true_skymodel : str or pathlib.Path or None
        Filename of input makesourcedb sky model, with primary-beam correction.

        - If this file exists, and `input_bright_skymodel` exists, they are
          concatenated and used as the `input_true_skymodel`.
        - If this file does not exist, and the `input_bright_skymodel` does not
          exist either, source filtering is still done on the
          `input_apparent_skymodel` if that exists.
    input_apparent_skymodel : str or pathlib.Path or None
        Filename of input makesourcedb sky model, without primary-beam
        correction.

        - If this file exists, and `input_true_skymodel` exists, it is filtered
          and grouped to match the sources and patches of the
          `input_true_skymodel`.
        - If this file does not exist it is generated from the
          `input_true_skymodel` by applying the beam attenuation. If `beam_ms`
          is not given, the generated `output_apparent_sky` file will be
          identical to the `output_true_sky` file.

    output_apparent_sky: str or pathlib.Path or None
        Output file name for the generated apparent sky model, without
        primary-beam correction.

        - If this file exists, it will be overwritten.

    output_true_sky : str or pathlib.Path
        Output file name for the generated true sky model, with primary-beam
        correction.

        - If this file exists, it will be overwritten.

    beam_ms : str or pathlib.Path or list of str or list of Path or None, default None
        The filename of the MS for deriving the beam attenuation and
        theoretical image noise.

        - If None (the default), or an empty string, the generated apparent and
          true sky models will be equal.

    input_bright_skymodel : str or pathlib.Path, optional
        Filename of input makesourcedb sky model of bright sources only. This
        parameter can be used to add back bright sources to the sky model that
        were peeled before imaging. This should be a true sky model, with
        primary-beam correction.

        - If `input_true_skymodel` exists, sources in `input_bright_skymodel`
          will be added to the sky model from `input_true_skymodel`.
        - If `input_true_skymodel` does not exist, the
          `input_bright_skymodel` will be used as the `input_true_skymodel`.

    Other Parameters
    ----------------
    source_finder : str, optional
        The source finder to use, either "sofia" or "bdsf". Defaults to "bdsf".
    **kws
        Additional keyword arguments to pass to the source finder function.

    Returns
    -------
    n_sources : int or None
        The number of sources detected (only returned when source_finder is
        "bdsf"; otherwise None is returned).

    See Also
    --------
    lsmtool.filter_skymodel.bdsf.filter_skymodel :
        Skymodel filtering implementation using pyBDSF.
    lsmtool.filter_skymodel.sofia.filter_skymodel :
        Skymodel filtering implementation using SoFia-2.

    """

    source_finder = resolve_source_finder(source_finder)
    runner = KNOWN_SOURCE_FINDERS[source_finder].filter_skymodel
    return runner(
        flat_noise_image,
        true_sky_image,
        input_true_skymodel,
        input_apparent_skymodel,
        output_apparent_sky,
        output_true_sky,
        beam_ms=beam_ms,
        input_bright_skymodel=input_bright_skymodel,
        **kws,
    )


def resolve_source_finder(name: str) -> str:
    """
    Resolve which source finder to use based on input string.

    This function checks the given source finder name against valid options,
    Currently supported options are: "bdsf" and "sofia" (if installed). The
    currently supported source finders are listed in the module variable:
    `KNOWN_SOURCE_FINDERS`.

    Parameters
    ----------
    name : str
        The source finder name to resolve. Input is converted to lower case. If
        the result is not in the list of known source finders, an exception is
        raised.

    Returns
    -------
    str
        The resolved source finder name.

    Raises
    ------
    ValueError
        If the input `name` is not a string matching one of the known source
        finders.
    """

    additional_info = ""
    if isinstance(name, str):
        source_finder = name.lower()
        if source_finder in KNOWN_SOURCE_FINDERS:
            return source_finder

        if source_finder == "sofia" and "sofia" not in KNOWN_SOURCE_FINDERS:
            additional_info = (
                " You have requested 'sofia' as a source finder, but it appears "
                "that SoFia-2 is not installed on your system. You may install "
                "sofia by running the following command: "
                ">>> python -m pip install lsmtool[sofia]"
            )

    raise ValueError(
        f"{name!r} is not a valid value for 'source_finder'. Valid "
        f"options are {set(KNOWN_SOURCE_FINDERS.keys())}.{additional_info}"
    )
