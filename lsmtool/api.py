"""
Tools for managing the LSMTool API.
"""

import functools as ftl
import warnings


class deprecated:
    """
    Decorator to mark functions as deprecated.
    """

    def __init__(
        self,
        replacement: str = None,
        renamed_parameters: dict = None,
        target_version: str = None,
        once: bool = True,
    ):
        """
        Mark a function as deprecated.

        Parameters
        ----------
        replacement : str, optional
            The new name of the function to use instead of the deprecated one.
        renamed_parameters : dict, optional
            A dictionary mapping old parameter names to new parameter names.
        target_version : str, optional
            The version of the package in which the deprecation will become an
            error.
        once : bool, optional
            If True, the deprecation warning will be emitted only once per
            function call site.
        """
        self.replacement = replacement
        self.renamed_parameters = renamed_parameters or {}
        self.target_version = target_version
        self.once = once

    def _get_message(self, func, kws):
        if self.replacement:
            yield (
                f"The function {func.__name__!r} is deprecated in favour of "
                f"{self.replacement!r}, please update your code to use the new "
                "function name."
            )

        if rename_needed := set(self.renamed_parameters).intersection(kws):
            yield (
                f"The following parameters of {func.__name__!r} have been "
                f"renamed:"
            )
            rename_needed = sorted(
                rename_needed,
                key=lambda s: func.__code__.co_varnames.index(
                    self.renamed_parameters[s]
                ),
            )

            for old in rename_needed:
                new = self.renamed_parameters[old]
                yield (f"    {old} -> {new}")
                kws[new] = kws.pop(old)

        if self.target_version:
            yield (
                f"This message will become an error in {__package__} version "
                f"{self.target_version}."
            )

    def emit(self, func, kws):
        """
        Emit a deprecation warning for the given function and keyword arguments.
        """
        message = "\n".join(self._get_message(func, kws))
        warnings.warn(message, DeprecationWarning)

        if self.once:
            self.emit = self.emit_noop

    def emit_noop(self, _, __):
        return

    def __call__(self, func):
        @ftl.wraps(func)
        def wrapper(*args, **kws):
            self.emit(func, kws)
            return func(*args, **kws)

        return wrapper


class deprecated_attribute:
    """
    A descriptor class for marking attributes as deprecated.
    """

    def __init__(
        self,
        replacement: str,
        target_version: str = None,
        once: bool = True,
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
        target_version : str, optional
            The version of the package in which the deprecation will become an
            error.
        once : bool, optional
            If True, the deprecation warning will be emitted only on the first
            time the lookup occurs.
        """
        self.replacement = replacement
        self.target_version = target_version
        self.once = once
        self.attribute_name = None
        self._emitted = False

    def __set_name__(self, owner, name):
        self.attribute_name = name

    def __get__(self, instance, owner=None):
        if not self._emitted:
            lookup_origin = (owner or instance.__class__).__name__
            message = self._get_message(lookup_origin)
            warnings.warn(message, DeprecationWarning)
            self._emitted = True

        return getattr(instance, self.replacement)

    def __set__(self, instance, value):
        if not self._emitted:
            lookup_origin = instance.__class__.__name__
            message = self._get_message(lookup_origin)
            warnings.warn(message, DeprecationWarning)
            self._emitted = True

        setattr(instance, self.replacement, value)

    def _get_message(self, origin):
        message = (
            f"The {self.attribute_name!r} attribute of {origin!r} is "
            "deprecated. Please use the new attribute name "
            f"{self.replacement!r} instead."
        )

        if self.target_version:
            message += (
                f" This message will become an error in {__package__} version "
                f"{self.target_version}."
            )

        return message
