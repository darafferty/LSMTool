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

from typing import Union

from ..io import (
    PathLike,
    PathLikeOptional,
    PathLikeOrListOptional,
)
from . import bdsf, sofia

KNOWN_SOURCE_FINDERS = {"sofia": sofia, "bdsf": bdsf}


def filter_skymodel(
    flat_noise_image: PathLike,
    true_sky_image: PathLikeOptional,
    input_true_skymodel: PathLikeOptional,
    input_apparent_skymodel: PathLikeOptional,
    output_apparent_sky: PathLike,
    output_true_sky: PathLike,
    beam_ms: PathLikeOrListOptional,
    /,
    source_finder: str = "bdsf",
    **kws,
):
    """
    Filters a sky model based on a source finder.

    This function filters a sky model using either SoFiA-2 or PyBDSF,
    based on the `source_finder` parameter.  It applies the chosen
    source finder to generate a filtered sky model.

    Parameters
    ----------
    flat_noise_image : str or Path
        Filename of input image to use to detect sources for filtering.
        It should be a flat-noise / apparent sky image (without primary-beam
        correction).
    true_sky_image : str or Path or None
        Filename of input image to use to determine the true flux of sources.
        It should be a true flux image (with primary-beam correction).
        - If `beam_ms` is given and exists, this parameter is ignored and
        the `flat_noise_image` is used instead.
        - If beam_ms is None or an empty string, this argument must be
        supplied.
    input_true_skymodel : str or Path or None
        Filename of input makesourcedb sky model, with primary-beam correction.
        - If this file does not exist, and `input_bright_skymodel` exists,
        `input_bright_skymodel` will be used as the `input_true_skymodel`.
        - If this file does not exist, and the `input_bright_skymodel` does not
        exist, the steps related to the input sky model are skipped but all
        other processing is still done.
    input_apparent_skymodel : str or Path or None
        Filename of input makesourcedb sky model, without primary-beam
        correction.
        - If this file exists, and input_true_skymodel exists, it is filtered
        and grouped to match the sources and patches of the
        `input_true_skymodel`.
        - If this file does not exist, and `input_true_skymodel` exists, it is
        generated from the `input_true_skymodel` by applying the beam
        attenuation. In this case, if `beam_ms` is not given, or is None, the
        generated `output_apparent_sky` will be identical to the
        `output_true_sky` files.
        - If `input_apparent_skymodel` does not exist, and neither
        `input_true_skymodel` nor `input_bright_skymodel` exist, an exception
        is raised.
    output_apparent_sky: str or Path or None
        Output file name for the generated apparent sky model, without
        primary-beam correction.
        - If this file exists, it will be overwritten.
    output_true_sky : str or Path
        Output file name for the generated true sky model, with
        primary-beam correction.
        - If this file exists, it will be overwritten.
    beam_ms : str or Path or list of str or list of Path or None, default None
        The filename of the MS for deriving the beam attenuation and
        theoretical image noise.
        - If None (the default), or an empty string, the generated
        apparent and true sky models will be equal.
    source_finder : str, optional
        The source finder to use, either "sofia" or "bdsf". Defaults to "bdsf".
    **kws
        Additional keyword arguments to pass to the source finder function.

    """

    source_finder = resolve_source_finder(source_finder)
    runner = KNOWN_SOURCE_FINDERS[source_finder].filter_skymodel
    runner(
        flat_noise_image,
        true_sky_image,
        input_true_skymodel,
        input_apparent_skymodel,
        output_apparent_sky,
        output_true_sky,
        beam_ms=beam_ms,
        **kws,
    )


def resolve_source_finder(
    name: Union[None, bool, str], fallback: str = "bdsf"
) -> Union[None, str]:
    """
    Resolve which source finder to use.

    This function checks the given source finder name against valid options
    ("sofia" or "bdsf"). If the name is invalid, it falls back to the
    default algorithm and emits a warning message. Custom message handling is
    possible by passing a callable as the `emit` parameter.

    Parameters
    ----------
    name : str or bool or None
        The source finder name to resolve.  If True or "on", the fallback
        is used.  If None, False, "off", or "none", None is returned. If an
        invalid string is passed, this is reported by the `emit` function, and
        the fallback value is returned (if emit does not raise an exception).
    fallback : str, optional
        The default source finder algorithm to use if the given name is
        invalid. Defaults to "bdsf".

    Returns
    -------
    str or None
        The resolved source finder name, or None if no source finder should
        be used.

    Raises
    ------
    ValueError
        If the input `name` is not boolean, or None, or not a string matching
        one of the known source finders.
    """

    if name in {None, False, "off", "none"}:
        return None

    if name in {True, "on"}:
        name = fallback

    if isinstance(name, str):
        source_finder = name.lower()
        if source_finder in KNOWN_SOURCE_FINDERS:
            return source_finder

    raise ValueError(
        f"{name!r} is not a valid value for 'source_finder'. Valid "
        f"options are {set(KNOWN_SOURCE_FINDERS.keys())}."
    )
