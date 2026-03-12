"""Video upload and listing endpoints."""

import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.config import Settings
from api.dependencies import get_settings, get_video_repo
from api.exceptions import VideoProcessingError
from api.repositories.video_repo import VideoRecord, VideoRepository
from api.schemas.video import VideoDTO, VideoListDTO, VideoUploadResponse

router = APIRouter(prefix="/videos", tags=["Videos"])


def _video_to_dto(record: VideoRecord) -> VideoDTO:
    """Convert a ``VideoRecord`` to a ``VideoDTO``.

    Args:
        record: Stored video metadata.

    Returns:
        API-facing video DTO with URL fields populated.
    """

    return VideoDTO(
        id=record.id,
        filename=record.filename,
        width=record.width,
        height=record.height,
        duration_sec=record.duration_sec,
        url=f"/videos/{record.id}/stream",
        poster_url=f"/videos/{record.id}/poster",
    )


def _probe_video(video_path: Path) -> tuple[int, int, float]:
    """Extract width, height, and duration from a video file using ffprobe.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        Tuple of ``(width, height, duration_sec)``.

    Raises:
        VideoProcessingError: If ffprobe fails or returns unexpected output.
    """

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-of", "csv=p=0",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoProcessingError(
            f"ffprobe failed for '{video_path.name}': {exc.stderr.strip()}"
        ) from exc

    parts = result.stdout.strip().split(",")

    try:
        width, height, duration = int(parts[0]), int(parts[1]), float(parts[2])
    except (IndexError, ValueError) as exc:
        raise VideoProcessingError(
            f"Unexpected ffprobe output for '{video_path.name}': {result.stdout!r}"
        ) from exc

    return width, height, duration


def _extract_poster(video_path: Path, poster_path: Path) -> None:
    """Extract the first frame of a video as a JPEG poster image.

    Args:
        video_path: Source video file path.
        poster_path: Destination JPEG path.

    Raises:
        VideoProcessingError: If ffmpeg fails to extract the frame.
    """

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(poster_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoProcessingError(
            f"ffmpeg poster extraction failed for '{video_path.name}': {exc.stderr.decode().strip()}"
        ) from exc


def _trim_video(
    source: Path,
    dest: Path,
    start_sec: float,
    duration_sec: float | None,
    ffmpeg_threads: int,
) -> None:
    """Trim a video to the specified time range and re-encode to dest.

    Args:
        source: Input video path.
        dest: Output video path.
        start_sec: Start offset in seconds.
        duration_sec: Duration to keep; ``None`` means until end of file.
        ffmpeg_threads: Number of threads for ffmpeg.

    Raises:
        VideoProcessingError: If ffmpeg trimming fails.
    """

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(source),
        "-threads", str(ffmpeg_threads),
        "-c", "copy",
    ]

    if duration_sec is not None:
        cmd += ["-t", str(duration_sec)]

    cmd.append(str(dest))

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoProcessingError(
            f"ffmpeg trim failed for '{source.name}': {exc.stderr.decode().strip()}"
        ) from exc


@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    status_code=201,
    summary="Upload a video",
    description=(
        "Upload an MP4 video for tracking. "
        "Optional ``start_sec`` and ``duration_sec`` trim the video before storing."
    ),
)
async def upload_video(
    file: UploadFile,
    start_sec: float = Form(default=0.0, ge=0.0),
    duration_sec: float | None = Form(default=None, gt=0.0),
    settings: Settings = Depends(get_settings),
    video_repo: VideoRepository = Depends(get_video_repo),
) -> VideoUploadResponse:
    """Accept a video upload, optionally trim it, and store metadata."""

    if file.content_type not in ("video/mp4", "video/quicktime"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload an MP4 video.",
        )

    video_id = str(uuid.uuid4())
    upload_dir: Path = settings.data_path / video_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    raw_path = upload_dir / "raw.mp4"
    final_path = upload_dir / "video.mp4"
    poster_path = upload_dir / "poster.jpg"

    try:
        with raw_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except OSError as exc:
        raise VideoProcessingError(
            f"Failed to save uploaded file '{file.filename}': {exc}"
        ) from exc

    needs_trim = start_sec > 0.0 or duration_sec is not None
    if needs_trim:
        _trim_video(raw_path, final_path, start_sec, duration_sec, settings.ffmpeg_threads)
        raw_path.unlink()
    else:
        raw_path.rename(final_path)

    width, height, actual_duration = _probe_video(final_path)

    if actual_duration > settings.max_upload_duration_sec:
        shutil.rmtree(upload_dir)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Video duration {actual_duration:.1f}s exceeds the "
                f"limit of {settings.max_upload_duration_sec:.0f}s."
            ),
        )

    _extract_poster(final_path, poster_path)

    record = VideoRecord(
        id=video_id,
        filename=file.filename or "video.mp4",
        video_path=str(final_path),
        poster_path=str(poster_path),
        width=width,
        height=height,
        duration_sec=actual_duration,
    )
    video_repo.add(record)

    return VideoUploadResponse(video=_video_to_dto(record))


@router.get(
    "",
    response_model=VideoListDTO,
    summary="List all videos",
    description="Return metadata for every uploaded video.",
)
def list_videos(
    video_repo: VideoRepository = Depends(get_video_repo),
) -> VideoListDTO:
    """Return all stored video records."""

    records = video_repo.list_all()

    return VideoListDTO(
        items=[_video_to_dto(r) for r in records],
        total=len(records),
    )


@router.get(
    "/{video_id}",
    response_model=VideoDTO,
    summary="Get video metadata",
    description="Return metadata for a single video by ID.",
)
def get_video(
    video_id: str,
    video_repo: VideoRepository = Depends(get_video_repo),
) -> VideoDTO:
    """Return metadata for the requested video."""

    record = video_repo.get(video_id)
    return _video_to_dto(record)


@router.get(
    "/{video_id}/stream",
    summary="Stream video file",
    description="Serve the raw video file for playback.",
    response_class=FileResponse,
)
def stream_video(
    video_id: str,
    video_repo: VideoRepository = Depends(get_video_repo),
) -> FileResponse:
    """Serve the video file for the requested ID."""

    record = video_repo.get(video_id)
    return FileResponse(record.video_path, media_type="video/mp4")


@router.get(
    "/{video_id}/poster",
    summary="Get poster image",
    description="Serve the first-frame poster JPEG for a video.",
    response_class=FileResponse,
)
def get_poster(
    video_id: str,
    video_repo: VideoRepository = Depends(get_video_repo),
) -> FileResponse:
    """Serve the poster image for the requested video."""

    record = video_repo.get(video_id)
    return FileResponse(record.poster_path, media_type="image/jpeg")
