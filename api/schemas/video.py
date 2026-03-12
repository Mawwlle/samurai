"""Video resource DTOs."""

from pydantic import BaseModel, Field


class VideoDTO(BaseModel):
    """Uploaded video metadata returned by the API.

    Attributes:
        id: Unique video identifier (UUID).
        filename: Original uploaded filename.
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_sec: Video duration in seconds.
        url: Relative URL to stream the video.
        poster_url: Relative URL to fetch the first-frame poster image.
    """

    id: str
    filename: str
    width: int
    height: int
    duration_sec: float
    url: str
    poster_url: str


class VideoListDTO(BaseModel):
    """Paginated list of videos.

    Attributes:
        items: Videos on the current page.
        total: Total number of videos stored.
    """

    items: list[VideoDTO]
    total: int


class VideoUploadResponse(BaseModel):
    """Response returned after a successful video upload.

    Attributes:
        video: Metadata of the newly created video.
    """

    video: VideoDTO


class VideoTrimParams(BaseModel):
    """Optional trim parameters for video upload.

    Attributes:
        start_sec: Start time offset in seconds (default: beginning of video).
        duration_sec: Duration to keep in seconds (default: full remaining video).
    """

    start_sec: float = Field(default=0.0, ge=0.0)
    duration_sec: float | None = Field(default=None, gt=0.0)
