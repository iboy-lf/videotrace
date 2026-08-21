"""VideoMemo package.

Keep the package import lightweight so that independent components such as the
Agent tool runtime can be exercised in a minimal serving environment.  The
full pipeline (and its retrieval dependencies) is imported only when callers
actually request :class:`VideoMemoPipeline`.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import VideoMemoPipeline

__all__ = ["VideoMemoPipeline"]


def __getattr__(name: str) -> Any:
    if name == "VideoMemoPipeline":
        from .pipeline import VideoMemoPipeline

        return VideoMemoPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
