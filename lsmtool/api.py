"""
Tools for managing the LSMTool API.
"""

import functools as ftl
import warnings


class Deprecated:
    """
    A descriptor class for marking attributes as deprecated.
    """

    def __init__(
        self,
        replacement: str = None,
        renamed_parameters: dict = None,
        target_version: str = None,
        warn_once: bool = True,
    ):
        """
        Mark an attribute as deprecated and provide a replacement.

        Value lookup and assignments are redirected to the replacement attribute
        and a deprecation warning is emitted. By default the warning is emitted
        only on the first lookup or assignment and silenced thereafter.

        Parameters
        ----------
        replacement : str
            New attribute name to use as a replacement.
        renamed_parameters : dict, optional
            A dictionary mapping old parameter names to new parameter names.
        target_version : str, optional
            The version of the package in which the deprecation will become an
            error.
        warn_once : bool, optional
            If True, the deprecation warning will be emitted only on the first
            time the lookup occurs.
        """
        if not replacement and not renamed_parameters:
            raise ValueError(
                "Either 'replacement' or 'renamed_parameters' must be provided."
            )

        self.replacement = replacement
        self.renamed_parameters = renamed_parameters or {}
        self.target_version = target_version
        self.warn_once = warn_once
        self.attribute_name = None

    def __call__(self, func):

        if not callable(func):
            raise TypeError(
                "The {self.__class__.__name__} decorator can only be applied to"
                "callable objects."
            )

        @ftl.wraps(func)
        def wrapper(*args, **kws):

            kws, replaced_kws = self._rename_parameters(kws)
            if replaced_kws or self.replacement:
                self.emit("", func, replaced_kws)

            return func(*args, **kws)

        return wrapper

    def _rename_parameters(self, kws):
        replaced = []
        if rename_needed := set(self.renamed_parameters).intersection(kws):
            for old in rename_needed:
                new = self.renamed_parameters[old]
                kws[new] = kws.pop(old)
                replaced.append(old)

        return kws, replaced

    def emit(self, origin, *args):
        """
        Emit a deprecation warning for the given function and keyword arguments.
        """
        message = self._get_message(origin, *args)
        warnings.warn(message, DeprecationWarning, stacklevel=3)

        if self.warn_once:
            self.emit = self.emit_noop

    def emit_noop(self, *_, **__):
        return

    def __set_name__(self, owner, name):
        self.attribute_name = name

    def __get__(self, instance, owner=None):
        self.emit(owner.__name__)
        return getattr(instance or owner, self.replacement)

    def __set__(self, instance, value):
        self.emit(instance.__class__.__name__)
        setattr(instance, self.replacement, value)

    def _get_message(self, origin, func=None, renamed_kws=None):

        if func:
            name = func.__name__
            descriptor = "function"
        else:
            name = self.attribute_name
            descriptor = "attribute"

        if origin:
            origin = f" of {origin!r}"
        else:
            origin = ""

        if self.replacement:
            message = (
                f"The {descriptor} {name!r}{origin} is deprecated. Please use "
                f"the new {descriptor} name {self.replacement!r} instead."
            )

        if renamed_kws:
            message = (
                f"The following parameters of {func.__name__!r} have been "
                f"renamed:"
            )
            kws = sorted(
                renamed_kws,
                key=lambda s: func.__code__.co_varnames.index(
                    self.renamed_parameters[s]
                ),
            )
            for old in kws:
                new = self.renamed_parameters[old]
                message += f"\n    {old} -> {new}"

        if self.target_version:
            message += (
                f"\nThis message will become an error in {__package__} version "
                f"{self.target_version}."
            )

        return message


# alias
deprecated = Deprecated
