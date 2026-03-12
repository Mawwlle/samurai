"""In-memory video repository backed by filesystem storage."""

import threading
from dataclasses import dataclass

from api.exceptions import VideoNotFoundError


@dataclass
class VideoRecord:
    """Metadata for a stored video.

    Attributes:
        id: Unique video identifier (UUID).
        filename: Original uploaded filename.
        video_path: Absolute filesystem path to the video file.
        poster_path: Absolute filesystem path to the poster image.
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_sec: Video duration in seconds.
    """

    id: str
    filename: str
    video_path: str
    poster_path: str
    width: int
    height: int
    duration_sec: float


class VideoRepository:
    """Thread-safe in-memory store for uploaded video metadata.

    Attributes:
        _videos: Map from video_id to ``VideoRecord``.
        _lock: Mutex protecting all mutations of ``_videos``.
    """

    def __init__(self) -> None:
        self._videos: dict[str, VideoRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: VideoRecord) -> None:
        """Store a new video record.

        Args:
            record: Fully initialised video record.
        """

        with self._lock:
            self._videos[record.id] = record

    def get(self, video_id: str) -> VideoRecord:
        """Return a video record by ID.

        Args:
            video_id: Unique video identifier.

        Returns:
            The matching ``VideoRecord``.

        Raises:
            VideoNotFoundError: If no video with this ID is stored.
        """

        with self._lock:
            record = self._videos.get(video_id)

        if record is None:
            raise VideoNotFoundError(video_id)

        return record

    def list_all(self) -> list[VideoRecord]:
        """Return all stored video records.

        Returns:
            Snapshot list of all ``VideoRecord`` instances.
        """

        with self._lock:
            return list(self._videos.values())

    def total(self) -> int:
        """Return the total number of stored videos.

        Returns:
            Count of stored video records.
        """

        with self._lock:
            return len(self._videos)
