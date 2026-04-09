"""Core package metadata for the retail forecasting engine."""

from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
	"""Resolve package version from installed metadata.

	Returns:
		Installed package version, or a local fallback.
	"""
	try:
		return version("retail-forecasting-engine")
	except PackageNotFoundError:
		return "0.1.0"


__version__ = _resolve_version()

__all__ = ["__version__"]
