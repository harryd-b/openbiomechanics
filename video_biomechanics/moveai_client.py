"""
Move.ai API Client for motion capture processing.

Provides integration with Move.ai's GraphQL API to:
- Upload videos and create files
- Create single-camera and multi-camera takes
- Submit processing jobs
- Poll job status
- Download motion capture outputs (FBX, BVH, JSON, etc.)

Usage:
    from moveai_client import MoveAiClient, MocapModels, OutputFormats

    client = MoveAiClient(api_key="your_api_key")

    # Single camera
    result = await client.process_single_camera("video.mp4")

    # Multi camera
    result = await client.process_multi_camera(
        videos=["cam1.mp4", "cam2.mp4"],
        human_height=1.77,
        lens="goprohero10-fhd"
    )
"""

import os
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

MOVEAI_GRAPHQL_ENDPOINT = "https://api.move.ai/ugc/graphql"


class MocapModels:
    """Available motion capture models."""
    # Single camera models
    S1 = "S1"
    S2 = "S2"
    S2_LIGHT = "S2-LIGHT"
    # Multi camera models
    M1 = "M1"
    M2 = "M2"
    M2_LIGHT = "M2-LIGHT"
    M2_XL = "M2-XL"
    # Real-time models
    RT1 = "RT1"
    RT2 = "RT2"


class OutputFormats:
    """Available output formats for motion capture."""
    MAIN_FBX = "MAIN_FBX"
    MAIN_BVH = "MAIN_BVH"
    MAIN_USDC = "MAIN_USDC"
    MAIN_USDZ = "MAIN_USDZ"
    MAIN_GLB = "MAIN_GLB"
    MAIN_BLEND = "MAIN_BLEND"
    MAIN_C3D = "MAIN_C3D"
    MAIN_JSON = "MAIN_JSON"
    MAIN_CSV = "MAIN_CSV"
    RENDER_VIDEO = "RENDER_VIDEO"


class JobState(Enum):
    """Job processing states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class FileInfo:
    """Information about an uploaded file."""
    id: str
    presigned_url: str


@dataclass
class TakeInfo:
    """Information about a created take."""
    id: str
    created: str


@dataclass
class JobInfo:
    """Information about a processing job."""
    id: str
    state: str
    percentage_complete: float
    outputs: List[Dict[str, Any]]


@dataclass
class ProcessingResult:
    """Result of a complete processing workflow."""
    job_id: str
    take_id: str
    file_ids: List[str]
    outputs: Dict[str, str]  # output_key -> local_file_path


class MoveAiClient:
    """
    Async client for Move.ai GraphQL API.

    Supports both single-camera and multi-camera motion capture workflows.
    """

    def __init__(self, api_key: str, endpoint: str = MOVEAI_GRAPHQL_ENDPOINT):
        """
        Initialize the Move.ai client.

        Args:
            api_key: Move.ai API key
            endpoint: GraphQL endpoint URL (default: production)
        """
        if not api_key:
            raise ValueError("API key is required")
        self.api_key = api_key
        self.endpoint = endpoint
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _execute_query(self, query: str, variables: Dict = None) -> Dict:
        """
        Execute a GraphQL query/mutation.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Response data dictionary
        """
        session = await self._get_session()

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key
        }

        async with session.post(self.endpoint, json=payload, headers=headers) as response:
            result = await response.json()

            if "errors" in result:
                error_msg = ", ".join(e.get("message", str(e)) for e in result["errors"])
                raise Exception(f"GraphQL Error: {error_msg}")

            return result.get("data", {})

    async def test_connection(self) -> Dict:
        """
        Test API connection by retrieving client info.

        Returns:
            Client information dictionary
        """
        query = """
            query Client {
                client {
                    created
                    id
                    metadata
                    name
                    portal
                }
            }
        """
        return await self._execute_query(query)

    async def create_file(self, file_type: str = "mp4") -> FileInfo:
        """
        Create a file entity and get presigned upload URL.

        Args:
            file_type: File extension (e.g., "mp4", "mov")

        Returns:
            FileInfo with id and presigned_url
        """
        query = """
            mutation CreateFile($type: String!) {
                file: createFile(type: $type) {
                    id
                    presignedUrl
                }
            }
        """
        result = await self._execute_query(query, {"type": file_type})
        file_data = result.get("file", {})
        return FileInfo(
            id=file_data.get("id"),
            presigned_url=file_data.get("presignedUrl")
        )

    async def upload_file(self, presigned_url: str, file_path: str,
                          progress_callback: Callable[[int, int], None] = None) -> None:
        """
        Upload a video file to the presigned URL.

        Args:
            presigned_url: S3 presigned URL from createFile
            file_path: Local path to video file
            progress_callback: Optional callback(bytes_sent, total_bytes)
        """
        path = Path(file_path)
        file_size = path.stat().st_size

        content_type_map = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".webm": "video/webm"
        }
        content_type = content_type_map.get(path.suffix.lower(), "video/mp4")

        session = await self._get_session()

        async with aiofiles.open(file_path, 'rb') as f:
            data = await f.read()

        headers = {"Content-Type": content_type}

        async with session.put(presigned_url, data=data, headers=headers) as response:
            if response.status not in (200, 201):
                raise Exception(f"Upload failed with status {response.status}")

        if progress_callback:
            progress_callback(file_size, file_size)

    async def create_single_cam_take(self, file_id: str,
                                      device_label: str = "cam01",
                                      format: str = "MP4") -> TakeInfo:
        """
        Create a single-camera take.

        Args:
            file_id: File ID from createFile
            device_label: Human-readable camera label
            format: Video format (MP4, MOV, etc.)

        Returns:
            TakeInfo with id and created timestamp
        """
        query = """
            mutation CreateSingleCamTake($sources: [SingleCamSourceInput!]!) {
                take: createSingleCamTake(sources: $sources) {
                    id
                    created
                }
            }
        """
        variables = {
            "sources": [{
                "deviceLabel": device_label,
                "fileId": file_id,
                "format": format.upper()
            }]
        }
        result = await self._execute_query(query, variables)
        take_data = result.get("take", {})
        return TakeInfo(
            id=take_data.get("id"),
            created=take_data.get("created")
        )

    async def create_single_cam_job(self, take_id: str,
                                     model: str = MocapModels.S2,
                                     track_fingers: bool = False,
                                     outputs: List[str] = None) -> JobInfo:
        """
        Create a single-camera processing job.

        Args:
            take_id: Take ID from createSingleCamTake
            model: Mocap model (S1, S2, S2-LIGHT)
            track_fingers: Enable finger tracking
            outputs: List of output formats

        Returns:
            JobInfo with initial status
        """
        outputs = outputs or [OutputFormats.MAIN_FBX, OutputFormats.MAIN_BVH, OutputFormats.MAIN_JSON]
        outputs_str = ", ".join(outputs)

        query = f"""
            mutation CreateSingleCamJob {{
                job: createSingleCamJob(
                    takeId: "{take_id}",
                    options: {{mocapModel: "{model}", trackFingers: {str(track_fingers).lower()}}},
                    outputs: [{outputs_str}]
                ) {{
                    id
                    created
                    progress {{
                        state
                        percentageComplete
                    }}
                }}
            }}
        """
        result = await self._execute_query(query)
        job_data = result.get("job", {})
        progress = job_data.get("progress", {})
        return JobInfo(
            id=job_data.get("id"),
            state=progress.get("state", "PENDING"),
            percentage_complete=progress.get("percentageComplete", 0),
            outputs=[]
        )

    async def create_volume_with_human(self,
                                        sources: List[Dict],
                                        human_height: float = 1.77,
                                        clip_start: float = 0.1,
                                        clip_end: float = None,
                                        clap_start: float = None,
                                        clap_end: float = None) -> str:
        """
        Create a calibration volume for multi-camera setup.

        Args:
            sources: List of camera sources with fileId, deviceLabel, lens, format
            human_height: Height of human subject in meters
            clip_start: Start time for calibration clip
            clip_end: End time for calibration clip
            clap_start: Start time for audio sync clap
            clap_end: End time for audio sync clap

        Returns:
            Volume ID
        """
        # Build sources input
        sources_input = []
        for src in sources:
            source_obj = {
                "deviceLabel": src.get("deviceLabel", f"cam{len(sources_input)+1:02d}"),
                "fileId": src["fileId"],
                "format": src.get("format", "MP4"),
                "cameraSettings": {
                    "lens": src.get("lens", "iphone15promax-4k")
                }
            }
            sources_input.append(source_obj)

        # Build sync method
        sync_method = {}
        if clap_start is not None and clap_end is not None:
            sync_method = {
                "clapWindow": {
                    "startTime": clap_start,
                    "endTime": clap_end
                }
            }

        # Build clip window
        clip_window = {"startTime": clip_start}
        if clip_end is not None:
            clip_window["endTime"] = clip_end

        query = """
            mutation CreateVolume($sources: [MultiCamSourceInput!]!, $humanHeight: Float!,
                                   $clipWindow: ClipWindowInput, $syncMethod: SyncMethodInput) {
                volume: createVolumeWithHuman(
                    sources: $sources,
                    humanHeight: $humanHeight,
                    clipWindow: $clipWindow,
                    syncMethod: $syncMethod,
                    areaType: NORMAL
                ) {
                    id
                    state
                    humanHeight
                }
            }
        """

        variables = {
            "sources": sources_input,
            "humanHeight": human_height,
            "clipWindow": clip_window
        }
        if sync_method:
            variables["syncMethod"] = sync_method

        result = await self._execute_query(query, variables)
        return result.get("volume", {}).get("id")

    async def get_volume(self, volume_id: str) -> Dict:
        """
        Get volume/calibration status.

        Args:
            volume_id: Volume ID

        Returns:
            Volume status dictionary
        """
        query = """
            query GetVolume($volumeId: ID!) {
                volume: getVolume(volumeId: $volumeId) {
                    id
                    state
                    humanHeight
                    created
                }
            }
        """
        result = await self._execute_query(query, {"volumeId": volume_id})
        return result.get("volume", {})

    async def create_multi_cam_take(self, volume_id: str, sources: List[Dict]) -> TakeInfo:
        """
        Create a multi-camera take using a calibrated volume.

        Args:
            volume_id: Calibrated volume ID
            sources: List of camera sources

        Returns:
            TakeInfo with id and created timestamp
        """
        query = """
            mutation CreateMultiCamTake($volumeId: ID!, $sources: [MultiCamSourceInput!]!) {
                take: createMultiCamTake(volumeId: $volumeId, sources: $sources) {
                    id
                    created
                }
            }
        """
        variables = {
            "volumeId": volume_id,
            "sources": sources
        }
        result = await self._execute_query(query, variables)
        take_data = result.get("take", {})
        return TakeInfo(
            id=take_data.get("id"),
            created=take_data.get("created")
        )

    async def create_multi_cam_job(self, take_id: str,
                                    number_of_actors: int = 1,
                                    outputs: List[str] = None,
                                    rig: str = None) -> JobInfo:
        """
        Create a multi-camera processing job.

        Args:
            take_id: Take ID from createMultiCamTake
            number_of_actors: Number of people in the capture
            outputs: List of output formats
            rig: Optional rig name (e.g., "move_ve")

        Returns:
            JobInfo with initial status
        """
        outputs = outputs or [OutputFormats.MAIN_FBX, OutputFormats.MAIN_BVH, OutputFormats.MAIN_JSON]
        outputs_str = ", ".join(outputs)

        rig_param = f', rig: "{rig}"' if rig else ""

        query = f"""
            mutation CreateMultiCamJob {{
                job: createMultiCamJob(
                    takeId: "{take_id}",
                    numberOfActors: {number_of_actors},
                    outputs: [{outputs_str}]{rig_param}
                ) {{
                    id
                    state
                    created
                }}
            }}
        """
        result = await self._execute_query(query)
        job_data = result.get("job", {})
        return JobInfo(
            id=job_data.get("id"),
            state=job_data.get("state", "PENDING"),
            percentage_complete=0,
            outputs=[]
        )

    async def get_job(self, job_id: str) -> JobInfo:
        """
        Get job status and outputs.

        Args:
            job_id: Job ID

        Returns:
            JobInfo with current status and outputs
        """
        query = """
            query GetJob($jobId: ID!) {
                job: getJob(jobId: $jobId) {
                    id
                    name
                    created
                    progress {
                        state
                        percentageComplete
                    }
                    outputs {
                        key
                        file {
                            id
                            presignedUrl
                            created
                        }
                    }
                }
            }
        """
        result = await self._execute_query(query, {"jobId": job_id})
        job_data = result.get("job", {})
        progress = job_data.get("progress", {})

        outputs = []
        for output in job_data.get("outputs", []) or []:
            if output.get("file"):
                outputs.append({
                    "key": output.get("key"),
                    "file_id": output["file"].get("id"),
                    "presigned_url": output["file"].get("presignedUrl")
                })

        return JobInfo(
            id=job_data.get("id"),
            state=progress.get("state", "UNKNOWN"),
            percentage_complete=progress.get("percentageComplete", 0),
            outputs=outputs
        )

    async def wait_for_job(self, job_id: str,
                           poll_interval: float = 10.0,
                           timeout: float = 3600.0,
                           progress_callback: Callable[[str, float], None] = None) -> JobInfo:
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID to monitor
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds
            progress_callback: Optional callback(state, percentage)

        Returns:
            Final JobInfo with outputs
        """
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            job = await self.get_job(job_id)

            if progress_callback:
                progress_callback(job.state, job.percentage_complete)

            if job.state == "FINISHED":
                return job
            elif job.state in ("FAILED", "CANCELLED"):
                raise Exception(f"Job {job.state}: {job_id}")

            await asyncio.sleep(poll_interval)

        raise Exception(f"Job timed out after {timeout} seconds")

    async def download_file(self, presigned_url: str, output_path: str) -> str:
        """
        Download a file from presigned URL.

        Args:
            presigned_url: S3 presigned download URL
            output_path: Local path to save file

        Returns:
            Path to downloaded file
        """
        session = await self._get_session()

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        async with session.get(presigned_url) as response:
            if response.status != 200:
                raise Exception(f"Download failed with status {response.status}")

            async with aiofiles.open(output_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)

        return output_path

    async def process_single_camera(self,
                                     video_path: str,
                                     output_dir: str = "./moveai_output",
                                     model: str = MocapModels.S2,
                                     track_fingers: bool = False,
                                     outputs: List[str] = None,
                                     progress_callback: Callable[[str, float], None] = None) -> ProcessingResult:
        """
        Complete single-camera processing workflow.

        Args:
            video_path: Path to video file
            output_dir: Directory for downloaded outputs
            model: Mocap model to use
            track_fingers: Enable finger tracking
            outputs: Output formats to generate
            progress_callback: Progress callback(stage, percentage)

        Returns:
            ProcessingResult with all output file paths
        """
        outputs = outputs or [OutputFormats.MAIN_FBX, OutputFormats.MAIN_BVH, OutputFormats.MAIN_JSON]

        if progress_callback:
            progress_callback("Creating file", 0)

        # Create file
        path = Path(video_path)
        file_type = path.suffix.lstrip('.').lower()
        file_info = await self.create_file(file_type)

        if progress_callback:
            progress_callback("Uploading video", 5)

        # Upload
        await self.upload_file(file_info.presigned_url, video_path)

        if progress_callback:
            progress_callback("Creating take", 15)

        # Create take
        take = await self.create_single_cam_take(
            file_info.id,
            device_label="camera-01",
            format=file_type.upper()
        )

        if progress_callback:
            progress_callback("Starting job", 20)

        # Create job
        job = await self.create_single_cam_job(
            take.id,
            model=model,
            track_fingers=track_fingers,
            outputs=outputs
        )

        # Wait for completion
        def job_progress(state, pct):
            if progress_callback:
                # Scale to 20-90% range
                scaled_pct = 20 + (pct * 0.7)
                progress_callback(f"Processing ({state})", scaled_pct)

        completed_job = await self.wait_for_job(job.id, progress_callback=job_progress)

        if progress_callback:
            progress_callback("Downloading outputs", 90)

        # Download outputs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = {}
        for output in completed_job.outputs:
            key = output["key"]
            url = output["presigned_url"]
            if url:
                ext = self._get_extension(key)
                output_path = output_dir / f"{job.id}_{key}.{ext}"
                await self.download_file(url, str(output_path))
                downloaded[key] = str(output_path)

        if progress_callback:
            progress_callback("Complete", 100)

        return ProcessingResult(
            job_id=job.id,
            take_id=take.id,
            file_ids=[file_info.id],
            outputs=downloaded
        )

    async def process_multi_camera(self,
                                    video_paths: List[str],
                                    output_dir: str = "./moveai_output",
                                    human_height: float = 1.77,
                                    lens: str = "iphone15promax-4k",
                                    clap_window: tuple = None,
                                    number_of_actors: int = 1,
                                    outputs: List[str] = None,
                                    progress_callback: Callable[[str, float], None] = None) -> ProcessingResult:
        """
        Complete multi-camera processing workflow.

        Args:
            video_paths: List of paths to video files
            output_dir: Directory for downloaded outputs
            human_height: Subject height in meters
            lens: Camera lens preset
            clap_window: Optional (start, end) times for audio sync
            number_of_actors: Number of people in capture
            outputs: Output formats to generate
            progress_callback: Progress callback(stage, percentage)

        Returns:
            ProcessingResult with all output file paths
        """
        outputs = outputs or [OutputFormats.MAIN_FBX, OutputFormats.MAIN_BVH, OutputFormats.MAIN_JSON]

        if progress_callback:
            progress_callback("Creating files", 0)

        # Create and upload files
        file_infos = []
        sources = []

        for i, video_path in enumerate(video_paths):
            path = Path(video_path)
            file_type = path.suffix.lstrip('.').lower()

            file_info = await self.create_file(file_type)
            file_infos.append(file_info)

            if progress_callback:
                pct = (i + 1) / len(video_paths) * 10
                progress_callback(f"Uploading video {i+1}/{len(video_paths)}", pct)

            await self.upload_file(file_info.presigned_url, video_path)

            sources.append({
                "deviceLabel": f"cam{i+1:02d}",
                "fileId": file_info.id,
                "format": file_type.upper(),
                "lens": lens
            })

        if progress_callback:
            progress_callback("Creating calibration volume", 15)

        # Create volume for calibration
        clap_start, clap_end = clap_window if clap_window else (None, None)
        volume_id = await self.create_volume_with_human(
            sources=sources,
            human_height=human_height,
            clap_start=clap_start,
            clap_end=clap_end
        )

        # Wait for calibration
        if progress_callback:
            progress_callback("Calibrating cameras", 20)

        for _ in range(60):  # 10 minute timeout
            volume = await self.get_volume(volume_id)
            if volume.get("state") == "FINISHED":
                break
            elif volume.get("state") == "FAILED":
                raise Exception("Volume calibration failed")
            await asyncio.sleep(10)

        if progress_callback:
            progress_callback("Creating take", 25)

        # Create multi-cam take
        take = await self.create_multi_cam_take(volume_id, sources)

        if progress_callback:
            progress_callback("Starting job", 30)

        # Create job
        job = await self.create_multi_cam_job(
            take.id,
            number_of_actors=number_of_actors,
            outputs=outputs
        )

        # Wait for completion
        def job_progress(state, pct):
            if progress_callback:
                scaled_pct = 30 + (pct * 0.6)
                progress_callback(f"Processing ({state})", scaled_pct)

        completed_job = await self.wait_for_job(job.id, progress_callback=job_progress)

        if progress_callback:
            progress_callback("Downloading outputs", 90)

        # Download outputs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = {}
        for output in completed_job.outputs:
            key = output["key"]
            url = output["presigned_url"]
            if url:
                ext = self._get_extension(key)
                output_path = output_dir / f"{job.id}_{key}.{ext}"
                await self.download_file(url, str(output_path))
                downloaded[key] = str(output_path)

        if progress_callback:
            progress_callback("Complete", 100)

        return ProcessingResult(
            job_id=job.id,
            take_id=take.id,
            file_ids=[f.id for f in file_infos],
            outputs=downloaded
        )

    def _get_extension(self, output_key: str) -> str:
        """Get file extension for output key."""
        extensions = {
            "MAIN_FBX": "fbx",
            "MAIN_BVH": "bvh",
            "MAIN_USDC": "usdc",
            "MAIN_USDZ": "usdz",
            "MAIN_GLB": "glb",
            "MAIN_BLEND": "blend",
            "MAIN_C3D": "c3d",
            "MAIN_JSON": "json",
            "MAIN_CSV": "csv",
            "RENDER_VIDEO": "mp4"
        }
        return extensions.get(output_key, "bin")

    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None


# Synchronous wrapper for easier use in Dash callbacks
class MoveAiClientSync:
    """
    Synchronous wrapper for MoveAiClient.

    Useful for integration with synchronous code like Dash callbacks.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_or_create_loop(self):
        """Get existing event loop or create new one."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def test_connection(self) -> Dict:
        """Test API connection."""
        async def _test():
            async with MoveAiClient(self.api_key) as client:
                return await client.test_connection()

        loop = self._get_or_create_loop()
        return loop.run_until_complete(_test())

    def process_single_camera(self, video_path: str, **kwargs) -> ProcessingResult:
        """Process single camera video."""
        async def _process():
            async with MoveAiClient(self.api_key) as client:
                return await client.process_single_camera(video_path, **kwargs)

        loop = self._get_or_create_loop()
        return loop.run_until_complete(_process())

    def process_multi_camera(self, video_paths: List[str], **kwargs) -> ProcessingResult:
        """Process multi camera videos."""
        async def _process():
            async with MoveAiClient(self.api_key) as client:
                return await client.process_multi_camera(video_paths, **kwargs)

        loop = self._get_or_create_loop()
        return loop.run_until_complete(_process())


# Available camera lens presets
CAMERA_LENSES = {
    # GoPro
    "goprohero10-fhd": "GoPro Hero 10 (1080p)",
    "goprohero10-4k": "GoPro Hero 10 (4K)",
    "goprohero11-fhd": "GoPro Hero 11 (1080p)",
    "goprohero11-4k": "GoPro Hero 11 (4K)",
    # iPhone
    "iphone13pro-4k": "iPhone 13 Pro (4K)",
    "iphone14pro-4k": "iPhone 14 Pro (4K)",
    "iphone15promax-4k": "iPhone 15 Pro Max (4K)",
    # Generic
    "generic-fhd": "Generic 1080p",
    "generic-4k": "Generic 4K"
}


def load_api_key(env_path: str = None) -> Optional[str]:
    """
    Load Move.ai API key from environment or .env file.

    Args:
        env_path: Optional path to .env file

    Returns:
        API key string or None
    """
    # Try environment variable first
    api_key = os.environ.get("MOVEAI_API_KEY") or os.environ.get("test_api_key")
    if api_key:
        return api_key

    # Try .env file
    env_paths = [env_path] if env_path else [".env", "../.env", "~/.moveai/.env"]

    for path in env_paths:
        if path is None:
            continue
        expanded = Path(path).expanduser()
        if expanded.exists():
            try:
                with open(expanded) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, _, value = line.partition('=')
                            if key.strip() in ("MOVEAI_API_KEY", "test_api_key"):
                                return value.strip()
            except Exception:
                pass

    return None
