"""Abstract base class for rendering backends."""

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Base class for all rendering backends."""

    @abstractmethod
    def render_clips(self, clips: list, **kwargs: Any) -> Any:
        """Render a list of Clip objects.

        Args:
            clips: List of Clip instances to render.
            **kwargs: Backend-specific options.

        Returns:
            Backend-specific result.
        """
        ...
