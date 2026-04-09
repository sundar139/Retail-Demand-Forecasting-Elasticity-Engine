"""Path resolution helpers for repository-standard directories."""

from dataclasses import dataclass
import logging
from pathlib import Path

from retail_forecasting.config import Settings, get_settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved, absolute paths used throughout the repository."""

    project_root: Path
    data_raw_dir: Path
    data_interim_dir: Path
    data_processed_dir: Path
    artifacts_dir: Path
    notebooks_dir: Path
    reports_figures_dir: Path
    prompts_dir: Path
    scripts_dir: Path


def get_project_root() -> Path:
    """Return the project root inferred from the package location.

    Returns:
        Absolute project root path.
    """
    return Path(__file__).resolve().parents[2]


def resolve_from_root(project_root: Path, candidate: Path) -> Path:
    """Resolve a possibly-relative path against the project root.

    Args:
        project_root: Base project root.
        candidate: Relative or absolute path.

    Returns:
        Absolute resolved path.
    """
    return candidate if candidate.is_absolute() else (project_root / candidate)


def build_project_paths(settings: Settings | None = None) -> ProjectPaths:
    """Build resolved repository paths from settings and defaults.

    Args:
        settings: Optional explicit Settings object.

    Returns:
        Fully resolved ProjectPaths structure.
    """
    resolved_settings = settings if settings is not None else get_settings()
    project_root = get_project_root()

    paths = ProjectPaths(
        project_root=project_root,
        data_raw_dir=resolve_from_root(project_root, resolved_settings.data_raw_dir),
        data_interim_dir=resolve_from_root(project_root, Path("data/interim")),
        data_processed_dir=resolve_from_root(project_root, resolved_settings.data_processed_dir),
        artifacts_dir=resolve_from_root(project_root, resolved_settings.artifacts_dir),
        notebooks_dir=resolve_from_root(project_root, Path("notebooks")),
        reports_figures_dir=resolve_from_root(project_root, Path("reports/figures")),
        prompts_dir=resolve_from_root(project_root, Path("prompts")),
        scripts_dir=resolve_from_root(project_root, Path("scripts")),
    )
    LOGGER.debug("Resolved project root to %s", paths.project_root)
    return paths
