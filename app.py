# ruff: noqa: E501
"""
This script provides a comprehensive solution for indexing a local image gallery,
generating descriptive metadata and vector embeddings using the Google Gemini API,
and performing semantic searches on the indexed data.

**Features:**
- **Database Backend:** Uses PostgreSQL with the pgvector extension for efficient similarity searches.
- **ORM:** Leverages SQLAlchemy 2.0 with async support for database interactions.
- **AI-Powered Indexing:**
    - Generates rich captions and relevant tags for each image using a Gemini model.
    - Creates vector embeddings from the generated text for semantic understanding.
    - Handles various image formats, including automatic sprite sheet conversion for GIFs.
- **Robust API Interaction:**
    - Implements an exponential backoff retry mechanism for handling transient API errors.
    - Supports both API Key and OAuth 2.0 authentication methods.
    - Intelligently falls back to different AI models if a preferred model fails.
- **Efficient & Asynchronous:**
    - Built with asyncio for high-concurrency processing of images.
    - Uses a semaphore to control concurrent API requests and respect rate limits.
- **Configuration:** Managed via a Pydantic Settings class, allowing for easy setup through environment variables.
- **Semantic Search:** Provides a function to find semantically similar images based on a natural language query.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Coroutine, Literal, Optional, TypeVar, Union
from zoneinfo import ZoneInfo

import colorlog
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.auth.credentials import TokenState
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.genai import Client, types, errors
from google.genai.types import Part
from pgvector.sqlalchemy import Vector
from PIL import Image as PILImage
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    FilePath,
    DirectoryPath,
    AnyUrl,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    select,
    text,
    or_,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, relationship

# ==============================================================================
# 1. Configuration
# ==============================================================================

load_dotenv()


class Settings(BaseSettings):
    """Manages application configuration using Pydantic for validation and type safety."""

    # --- Project & Authentication ---
    PROJECT_ID: Optional[str] = None
    LOCATION: str = "us-central1"
    CLIENT_SECRETS_FILE: FilePath = Field(
        default=Path(".secrets/client_secret.json"),
        description="Path to the Google OAuth client secrets JSON file.",
    )
    TOKEN_DIR: DirectoryPath = Field(
        default=Path(".secrets"),
        description="Directory to store the OAuth token file.",
    )
    REDIRECT_URI: str = "urn:ietf:wg:oauth:2.0:oob"
    SCOPES: list[str] = [
        "openid",
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]
    GOOGLE_API_KEY: Optional[str] = None
    CRED_TYPE: Literal["API", "OAUTH"] = "API"

    # --- Database ---
    DB_URL: AnyUrl = Field(
        default="postgresql+asyncpg://user:password@localhost/rao-search",
        description="Async PostgreSQL database URL with pgvector installed.",
    )

    # --- Gemini AI Models ---
    AI_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "text-embedding-004"
    VECTOR_DIMENSION: int = 768

    # --- File & Directory Paths ---
    IMAGE_GALLERY: DirectoryPath = Field(
        default=Path.home() / "pictures",
        description="Directory containing images to be indexed.",
    )
    TEMP_DIRECTORY: Path = Field(default=Path("temp"), description="Directory for temporary files (e.g., GIF sprites).")
    LOG_DIR: Path = Field(default=Path("logs"), description="Directory for log files.")

    # --- Indexing & Concurrency ---
    SEMAPHORE_LIMIT: int = 3
    INDEX_TASK_DELAY: int = 5
    NO_BROWSER: bool = False
    NO_RETRY_DELAY: bool = False
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 2.0

    # --- File Handling ---
    IMAGE_EXTENSIONS: set[str] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".heic",
        ".heif",
        ".gif",
    }

    @property
    def TOKEN_FILE(self) -> Path:
        """Constructs the full path to the token file."""
        return self.TOKEN_DIR / "token.json"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Instantiate settings early
settings = Settings()


# ==============================================================================
# 2. Logging Setup
# ==============================================================================

# Create a logger instance for the application
logger = logging.getLogger("rao_search")


def setup_logging():
    """Configures colorized logging for console output and file output."""
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent duplicate logs in root logger

    # Console Handler
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    stream_handler = colorlog.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # File Handler
    settings.LOG_DIR.mkdir(exist_ok=True)
    log_file_name = f"app_{datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d_%H-%M-%S')}.log"
    file_handler = logging.FileHandler(settings.LOG_DIR / log_file_name)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.INFO)


# ==============================================================================
# 3. Custom Exceptions
# ==============================================================================


class RaoSearchError(Exception):
    """Base exception for this application."""


class PermanentAnalysisError(RaoSearchError):
    """For errors that should not be retried (e.g., content blocks, parsing failures)."""


class DatabaseError(RaoSearchError):
    """For database connection or query failures."""


class ConfigurationError(RaoSearchError):
    """For misconfigurations or setup problems."""


class FileProcessingError(RaoSearchError):
    """For errors during file I/O or processing (e.g., GIF conversion)."""


# ==============================================================================
# 4. Database Models (SQLAlchemy ORM)
# ==============================================================================

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    user_id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    images = relationship("Image", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.user_id}, username='{self.username}')>"


class Image(Base):
    __tablename__ = "images"
    image_id = Column(BigInteger, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, unique=True)
    local_file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer)
    width_px = Column(Integer)
    height_px = Column(Integer)
    mime_type = Column(String(50))
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    capture_date = Column(DateTime)
    ai_caption = Column(String(500))
    user_id = Column(BigInteger, ForeignKey("users.user_id"))
    tags = relationship("ImageTagXref", back_populates="image", cascade="all, delete-orphan")
    vector = relationship("FeatureVector", uselist=False, back_populates="image", cascade="all, delete-orphan")
    user = relationship("User", back_populates="images")

    def __repr__(self):
        return f"<Image(id={self.image_id}, filename='{self.filename}')>"


class Tag(Base):
    __tablename__ = "tags"
    tag_id = Column(BigInteger, primary_key=True, autoincrement=True)
    tag_name = Column(String(100), nullable=False, unique=False)
    tag_category = Column(String(50))
    images = relationship("ImageTagXref", back_populates="tag")

    def __repr__(self):
        return f"<Tag(id={self.tag_id}, name='{self.tag_name}')>"


class ImageTagXref(Base):
    __tablename__ = "image_tag_xref"
    image_tag_id = Column(BigInteger, primary_key=True, autoincrement=True)
    image_id = Column(BigInteger, ForeignKey("images.image_id"), nullable=False)
    tag_id = Column(BigInteger, ForeignKey("tags.tag_id"), nullable=False)
    confidence = Column(Float)
    image = relationship("Image", back_populates="tags")
    tag = relationship("Tag", back_populates="images")
    __table_args__ = (UniqueConstraint("image_id", "tag_id", name="_image_tag_uc"),)

    def __repr__(self):
        return f"<ImageTagXref(image_id={self.image_id}, tag_id={self.tag_id})>"


class FeatureVector(Base):
    __tablename__ = "feature_vectors"
    image_id = Column(BigInteger, ForeignKey("images.image_id"), primary_key=True)
    vector_embedding = Column(Vector(settings.VECTOR_DIMENSION), nullable=False)
    model_version = Column(String(50))
    image = relationship("Image", back_populates="vector")

    def __repr__(self):
        return f"<FeatureVector(image_id={self.image_id}, model='{self.model_version}')>"


# ==============================================================================
# 5. Pydantic Models for API Responses
# ==============================================================================


class ObjectDetection(BaseModel):
    """Defines the structure for a single detected object."""

    box_2d: list[int] = Field(description="Normalized coordinates [y0, x0, y1, x1] (0-1000).")
    label: str = Field(description="A descriptive label for the object.")


class ImageAnalysis(BaseModel):
    """Defines the complete structured output from the Gemini analysis call."""

    caption: str = Field(description="A short, concise description about the image.")
    tags: list[str] = Field(description="A list of 3-5 keywords that describe the image.")
    objects: list[ObjectDetection] = Field(
        default_factory=list,
        description="A list of up to 10 objects detected in the image.",
        max_length=10,
    )


# ==============================================================================
# 6. Generic Retry Decorator
# ==============================================================================

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])


def async_retry(
    max_retries: int = settings.MAX_RETRIES,
    backoff_factor: float = settings.BACKOFF_FACTOR,
    permanent_exceptions: tuple[type[Exception], ...] = (PermanentAnalysisError,),
) -> Callable[[F], F]:
    """
    A decorator for retrying an async function with exponential backoff.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except permanent_exceptions as e:
                    logger.critical(f"Permanent failure in '{func.__name__}': {e}. No more retries.")
                    raise  # Re-raise the permanent error
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} for '{func.__name__}' failed.")

                    sleep_time = 0.0
                    if not settings.NO_RETRY_DELAY:
                        sleep_time = backoff_factor**attempt
                        if isinstance(e, errors.GoogleAPIError) and "429" in str(e):
                            logger.warning("[429 Error] Resource exhausted. Using exponential backoff.")
                        else:
                            logger.warning(f"Transient error: {e}. Using exponential backoff.")

                    if attempt + 1 < max_retries:
                        if sleep_time > 0:
                            logger.info(f"Sleeping {sleep_time:.2f} seconds before retry...")
                            await asyncio.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_retries} retries for '{func.__name__}' failed.")
                        raise  # Re-raise the last exception

        return wrapper  # type: ignore

    return decorator


# ==============================================================================
# 7. Gemini Service
# ==============================================================================


class GeminiService:
    """Encapsulates all interactions with the Google Gemini API."""

    def __init__(self, cred_type: str):
        self.client: Client = self._initialize_client(cred_type)
        self.cred_type = cred_type
        self._model_cache: Optional[tuple[list[str], list[str]]] = None
        self.ai_model = settings.AI_MODEL
        self.embedding_model = settings.EMBEDDING_MODEL

    def _load_oauth_creds(self) -> Credentials:
        """Handles the OAuth 2.0 authentication flow."""
        creds = None
        if settings.TOKEN_FILE.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(settings.TOKEN_FILE), settings.SCOPES)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load token file, will re-authenticate: {e}")

        if not creds or not creds.token_state(TokenState.FRESH):

            if creds and creds.token_state(TokenState.INVALID or TokenState.STALE) and creds.refresh_token:
                logger.info("Refreshing expired OAuth credentials...")
                creds.refresh(Request())
            else:
                logger.info("Performing OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.CLIENT_SECRETS_FILE, settings.SCOPES, redirect_uri=settings.REDIRECT_URI
                )
                if settings.NO_BROWSER:
                    auth_url, _ = flow.authorization_url(prompt="consent")
                    logger.info(f"Please visit this URL to authorize: {auth_url}")
                    code = input("Enter the authorization code: ")
                    flow.fetch_token(code=code)
                    creds = flow.credentials
                else:
                    creds = flow.run_local_server(open_browser=True)

            with open(settings.TOKEN_FILE, "w") as token_file:
                token_file.write(creds.to_json())
            logger.info("OAuth credentials saved.")
        return creds

    def _initialize_client(self, cred_type: str) -> Client:
        """Initializes the Gemini client using either API Key or OAuth."""
        try:
            if cred_type == "OAUTH":
                logger.info("Initializing Gemini client with OAuth credentials...")
                creds = self._load_oauth_creds()
                return Client(credentials=creds, vertexai=True, location=settings.LOCATION, project_id=settings.PROJECT_ID)
            elif cred_type == "API" and settings.GOOGLE_API_KEY:
                logger.info("Initializing Gemini client with API Key...")
                return Client(api_key=settings.GOOGLE_API_KEY)
            else:
                raise ConfigurationError("Gemini client could not be initialized. Check CRED_TYPE and GOOGLE_API_KEY.")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Gemini client: {e}")

    async def get_available_models(self) -> tuple[list[str], list[str]]:
        """Fetches and caches a sorted list of available AI and embedding models."""
        if self._model_cache:
            return self._model_cache

        logger.info("Fetching available models from the API...")
        ai_models, embed_models = [], []
        try:
            model_list = await self.client.aio.models.list()
            for model in model_list:
                name = model.name.replace("models/", "")
                if "embedContent" in model.supported_actions:
                    embed_models.append(name)
                if "generateContent" in model.supported_actions:
                    ai_models.append(name)
        except Exception as e:
            raise RaoSearchError(f"Could not retrieve model list: {e}")

        ai_models.sort(reverse=True)
        embed_models.sort(reverse=True)
        self._model_cache = (ai_models, embed_models)
        logger.debug(f"Found {len(ai_models)} AI models and {len(embed_models)} embedding models.")
        return ai_models, embed_models

    async def _ensure_models_are_available(self):
        """Ensures the configured models are available, falling back if necessary."""
        if self._model_cache:
            return
        available_ai, available_embed = await self.get_available_models()
        if self.ai_model not in available_ai:
            logger.warning(f"Configured AI model '{self.ai_model}' not found. Using '{available_ai[0]}'.")
            self.ai_model = available_ai[0]
        if self.embedding_model not in available_embed:
            logger.warning(f"Configured embedding model '{self.embedding_model}' not found. Using '{available_embed[0]}'.")
            self.embedding_model = available_embed[0]

    @async_retry()
    async def get_analysis(self, file_resource: Union[types.File, dict], mime_type: str, is_sprite: bool) -> ImageAnalysis:
        """Generates image analysis using the Gemini model."""
        await self._ensure_models_are_available()
        file_name = file_resource.name if hasattr(file_resource, "name") else file_resource.get("file_name", "bytes")
        logger.info(f"Generating analysis for '{file_name}' with model '{self.ai_model}'...")

        if is_sprite:
            prompt = "Analyze this sprite sheet of 5 frames from a GIF. Describe the full animation from start to finish. Do not mention that its a sprite sheet."
        else:
            prompt = "Analyze the image and provide the requested structured data."

        if self.cred_type == "API":
            contents = [Part.from_text(text=prompt), file_resource]
        else:  # OAUTH
            contents = [Part.from_text(text=prompt), Part.from_bytes(data=file_resource["bytes"], mime_type=mime_type)]

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=ImageAnalysis.model_json_schema(),
            thinking_config=types.ThinkingConfig(include_thoughts=False),
        )
        response = await self.client.aio.models.generate_content(model=self.ai_model, contents=contents, config=config)

        if not response or not response.text:
            raise PermanentAnalysisError("Gemini returned an empty response.")

        try:
            return ImageAnalysis.model_validate_json(response.text)
        except (ValidationError, IndexError, AttributeError) as e:
            logger.error(f"Failed to parse model's JSON response: {e}\nRaw response: {response.text}")
            raise PermanentAnalysisError("Failed to validate Gemini's JSON response.")

    @async_retry()
    async def get_embedding(self, text_content: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Generates a vector embedding for a list of text strings."""
        await self._ensure_models_are_available()
        logger.info(f"Generating embedding for {len(text_content)} strings with model '{self.embedding_model}'...")
        response = await self.client.aio.models.embed_content(
            model=self.embedding_model,
            contents=text_content,
            config=types.EmbedContentConfig(output_dimensionality=settings.VECTOR_DIMENSION, task_type=task_type),
        )
        if not response.embeddings:
            raise PermanentAnalysisError("API returned no embeddings.")
        return response.embeddings[0].values

    async def upload_file(self, path: Path, mime_type: str) -> Union[types.File, dict]:
        """Uploads a file to the Gemini API or reads it into memory for OAuth."""
        logger.info(f"Preparing '{path.name}' for processing (MIME: {mime_type}).")
        if self.cred_type == "API":
            try:
                file_resource = await self.client.aio.files.upload(file=path, config={"mime_type": mime_type})
                logger.info(f"API Upload successful: {file_resource.name}")
                return file_resource
            except Exception as e:
                raise FileProcessingError(f"SDK Upload Failed for {path.name}: {e}")
        else:  # OAUTH
            try:
                with open(path, "rb") as f:
                    return {"bytes": f.read(), "file_name": path.name}
            except IOError as e:
                raise FileProcessingError(f"I/O error reading {path.name}: {e}")

    async def delete_file(self, file_resource: Any):
        """Deletes a file from the Gemini API if applicable."""
        if self.cred_type == "API" and isinstance(file_resource, types.File):
            try:
                await self.client.aio.files.delete(name=file_resource.name)
                logger.info(f"API file deleted: {file_resource.name}")
            except Exception as e:
                logger.warning(f"Could not delete API file {file_resource.name}: {e}")


# ==============================================================================
# 8. Database Service
# ==============================================================================


class DatabaseService:
    """Manages all database interactions."""

    def __init__(self, db_url: AnyUrl):
        if not str(db_url).startswith("postgresql+asyncpg"):
            raise ConfigurationError("Database URL must use the 'postgresql+asyncpg' driver.")
        self.engine = create_async_engine(str(db_url), echo=False, pool_pre_ping=True)
        self.Session = async_sessionmaker(bind=self.engine, expire_on_commit=False, class_=AsyncSession)

    async def initialize(self):
        """Enables the vector extension and creates tables."""
        logger.info(f"Connecting to database at {self.engine.url.host}...")
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database initialization complete.")
        except Exception as e:
            raise DatabaseError(f"Could not initialize database: {e}")

    async def get_or_create_user(self, session: AsyncSession, username: str, email: str) -> User:
        """Retrieves a user by username or creates them if they don't exist."""
        result = await session.execute(select(User).filter_by(username=username))
        user = result.scalar_one_or_none()
        if not user:
            logger.info(f"Creating new user '{username}'...")
            user = User(username=username, email=email)
            session.add(user)
            await session.flush()  # Flush to get the ID, but don't commit
        return user

    async def get_new_and_incomplete_image_paths(self) -> list[Path]:
        """Scans the gallery and returns paths for new or incompletely indexed images."""
        gallery_dir = settings.IMAGE_GALLERY
        disk_paths = {p for p in gallery_dir.iterdir() if p.is_file() and p.suffix.lower() in settings.IMAGE_EXTENSIONS}
        disk_filenames = {p.name for p in disk_paths}

        async with self.Session() as session:
            # Find existing files
            stmt_existing = select(Image.filename).where(Image.filename.in_(disk_filenames))
            result_existing = await session.execute(stmt_existing)
            db_filenames = {row[0] for row in result_existing}

            # Find incomplete files
            subquery_tagged = select(ImageTagXref.image_id).join(Tag).where(Tag.tag_category == "AI_Keyword").distinct()
            stmt_incomplete = (
                select(Image)
                .outerjoin(Image.vector)
                .where(
                    or_(
                        Image.ai_caption.is_(None) | (Image.ai_caption == ""),
                        Image.image_id.notin_(subquery_tagged),
                        FeatureVector.image_id.is_(None),
                    )
                )
            )
            result_incomplete = await session.execute(stmt_incomplete)
            incomplete_images = result_incomplete.scalars().unique().all()

        new_filenames = disk_filenames - db_filenames
        paths_to_index = {p for p in disk_paths if p.name in new_filenames}

        if incomplete_images:
            logger.info(f"Found {len(incomplete_images)} incomplete entries to be re-indexed.")
            for image in incomplete_images:
                path = Path(image.local_file_path)
                if path.exists():
                    paths_to_index.add(path)

        num_skipped = len(disk_filenames) - len(new_filenames)
        if num_skipped > 0:
            logger.info(f"Skipped {num_skipped} already indexed images.")

        return list(paths_to_index)

    async def find_similar_images(self, query_vector: list[float], limit: int) -> list[dict]:
        """Performs a vector similarity search in the database."""
        async with self.Session() as session:
            distance = FeatureVector.vector_embedding.l2_distance(query_vector).label("distance")
            stmt = select(Image, distance).join(FeatureVector).order_by(distance).limit(limit)
            result = await session.execute(stmt)
            return [
                {
                    "image_id": img.image_id,
                    "filename": img.filename,
                    "ai_caption": img.ai_caption,
                    "distance": dist,
                }
                for img, dist in result.all()
            ]

    async def dispose(self):
        """Closes the database connection engine."""
        await self.engine.dispose()


# ==============================================================================
# 9. Image & File Utilities
# ==============================================================================


def _sync_convert_gif(gif_path: Path, temp_path: Path, num_frames: int) -> Optional[Path]:
    """Synchronous implementation of GIF to sprite sheet conversion."""
    try:
        with PILImage.open(gif_path) as img:
            total_frames = img.n_frames
            indices = sorted(
                list(set([int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)] if num_frames > 1 else [0]))
            )

            frames = []
            for i in indices:
                img.seek(i)
                frames.append(img.copy().convert("RGBA"))

            if not frames:
                return None

            sprite_sheet = PILImage.new("RGBA", (frames[0].width * len(frames), frames[0].height))
            for i, frame in enumerate(frames):
                sprite_sheet.paste(frame, (i * frames[0].width, 0))

            sprite_sheet.save(temp_path, format="PNG")
            logger.info(f"Saved {len(frames)}-frame sprite sheet: {temp_path.name}")
            return temp_path
    except Exception as e:
        raise FileProcessingError(f"Pillow failed to convert GIF {gif_path.name}: {e}")


async def convert_gif_to_sprite_sheet(gif_path: Path, temp_path: Path, num_frames: int = 5) -> Optional[Path]:
    """Asynchronously converts a GIF to a horizontal PNG sprite sheet."""
    return await asyncio.to_thread(_sync_convert_gif, gif_path, temp_path, num_frames)


def get_image_mime_type(extension: str) -> Optional[str]:
    """Returns the MIME type for a given file extension."""
    return f"image/{'jpeg' if extension == '.jpg' else extension.lstrip('.')}"


async def cleanup_temp_directory():
    """Deletes all files in the temporary directory."""
    settings.TEMP_DIRECTORY.mkdir(exist_ok=True)
    for item in settings.TEMP_DIRECTORY.iterdir():
        if item.is_file():
            try:
                item.unlink()
            except OSError as e:
                logger.warning(f"Error deleting temp file {item.name}: {e}")
    logger.info("Temporary directory cleaned up.")


# ==============================================================================
# 10. Main Application Logic
# ==============================================================================


async def process_and_index_image(
    session: AsyncSession,
    gemini: GeminiService,
    image_path: Path,
    user_id: int,
):
    """
    The core logic for processing a single image: upload, analyze, embed, and
    store in the database.
    """
    logger.info(f"[Indexing] Starting: {image_path.name}")
    file_resource = None
    temp_image_path = None
    is_sprite = False
    ext = image_path.suffix.lower()
    orig_mime = get_image_mime_type(ext)

    try:
        # --- 1. Prepare file for upload (handle GIFs) ---
        path_to_upload = image_path
        upload_mime = orig_mime
        if ext == ".gif":
            temp_name = image_path.with_suffix(".sprite.png").name
            temp_image_path = await convert_gif_to_sprite_sheet(image_path, settings.TEMP_DIRECTORY / temp_name)
            if not temp_image_path:
                raise FileProcessingError("GIF conversion failed.")
            path_to_upload = temp_image_path
            upload_mime = "image/png"
            is_sprite = True

        # --- 2. Upload and get AI analysis ---
        file_resource = await gemini.upload_file(path_to_upload, upload_mime)
        analysis = await gemini.get_analysis(file_resource, upload_mime, is_sprite)

        # --- 3. Get embedding ---
        embedding_content = [analysis.caption] + analysis.tags + [obj.label for obj in analysis.objects]
        vector = await gemini.get_embedding(embedding_content)

        # --- 4. Delete existing entry if it's a re-scan ---
        existing_img_stmt = select(Image).filter_by(filename=image_path.name)
        existing_img = (await session.execute(existing_img_stmt)).scalar_one_or_none()
        if existing_img:
            logger.info(f"Deleting incomplete entry for '{image_path.name}' before re-indexing.")
            await session.delete(existing_img)
            await session.flush()  # Ensure delete is processed before add

        # --- 5. Populate database ---
        with PILImage.open(image_path) as img:
            width, height = img.size

        new_image = Image(
            filename=image_path.name,
            local_file_path=str(image_path.resolve()),
            file_size_bytes=image_path.stat().st_size,
            width_px=width,
            height_px=height,
            mime_type=orig_mime,
            user_id=user_id,
            ai_caption=analysis.caption,
        )
        session.add(new_image)
        await session.flush()  # Flush to get new_image.image_id

        session.add(FeatureVector(image_id=new_image.image_id, vector_embedding=vector, model_version=gemini.embedding_model))

        # Add tags
        tags_to_add = {
            (analysis.caption, "AI_Caption", 1.0),
            *((tag, "AI_Keyword", 0.9) for tag in analysis.tags),
            *((obj.label, "AI_Object", 0.8) for obj in analysis.objects),
        }
        for tag_name, category, confidence in tags_to_add:
            tag_name = tag_name[:100]  # Enforce length limit
            tag_result = await session.execute(select(Tag).filter_by(tag_name=tag_name, tag_category=category))
            tag = tag_result.scalar_one_or_none()
            if not tag:
                tag = Tag(tag_name=tag_name, tag_category=category)
                session.add(tag)
                await session.flush()
            session.add(ImageTagXref(image_id=new_image.image_id, tag_id=tag.tag_id, confidence=confidence))

        logger.info(f"[Indexing] SUCCESS: {image_path.name}")

    except Exception as e:
        logger.error(f"[Indexing] FAILED for {image_path.name}: {e}")
        raise  # Re-raise to be caught by the task runner for rollback
    finally:
        if file_resource:
            await gemini.delete_file(file_resource)
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink()


async def indexing_task_runner(
    db: DatabaseService,
    gemini: GeminiService,
    user_id: int,
    image_paths: list[Path],
):
    """Manages concurrent execution of indexing tasks with a semaphore."""
    semaphore = asyncio.Semaphore(settings.SEMAPHORE_LIMIT)
    tasks = []

    async def task_wrapper(path: Path):
        async with semaphore:
            try:
                async with db.Session() as session:
                    async with session.begin():  # Use transaction
                        await process_and_index_image(session, gemini, path, user_id)
                # Add delay *after* successful completion of a task
                if settings.INDEX_TASK_DELAY > 0:
                    await asyncio.sleep(settings.INDEX_TASK_DELAY)
            except Exception:
                # Error is already logged in process_and_index_image
                # The transaction ensures a rollback on failure
                pass

    for path in image_paths:
        tasks.append(task_wrapper(path))

    await asyncio.gather(*tasks)


async def run_search_test(db: DatabaseService, gemini: GeminiService, query: str):
    """Runs a sample search query and prints the results."""
    logger.info(f"\n{'='*50}\nPERFORMING SEMANTIC SEARCH TEST\n{'='*50}")
    logger.info(f"Search Query: '{query}'")

    query_vector = await gemini.get_embedding([query], task_type="RETRIEVAL_QUERY")
    if not query_vector:
        logger.error("Could not generate query vector. Aborting search test.")
        return

    results = await db.find_similar_images(query_vector, limit=3)

    if results:
        for res in results:
            print(
                f"  - File: {res['filename']} (ID: {res['image_id']})\n"
                f"    Caption: {res['ai_caption']}\n"
                f"    Distance (L2): {res['distance']:.4f}\n"
            )
    else:
        logger.info("No search results found.")


async def main():
    """Main asynchronous function to orchestrate the application."""
    setup_logging()
    logger.info("--- Rao Search Application Starting ---")

    # --- Initialization ---
    db = DatabaseService(settings.DB_URL)
    await db.initialize()
    gemini = GeminiService(settings.CRED_TYPE)
    await cleanup_temp_directory()

    # --- Get User ---
    async with db.Session() as session:
        async with session.begin():
            user = await db.get_or_create_user(session, "image_indexer_bot", "indexer@example.com")

    # --- Indexing ---
    paths_to_index = await db.get_new_and_incomplete_image_paths()
    if paths_to_index:
        logger.info(f"Found {len(paths_to_index)} new or incomplete images to index.")
        await indexing_task_runner(db, gemini, user.user_id, paths_to_index)
    else:
        logger.info("Image gallery is fully indexed and up to date.")

    # --- Search Test ---
    await run_search_test(db, gemini, "A person smiling")

    # --- Cleanup ---
    await db.dispose()
    logger.info("--- Application Finished ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (ConfigurationError, DatabaseError, KeyboardInterrupt) as e:
        logger.critical(f"Application terminated: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected fatal error occurred: {e}", exc_info=True)
        sys.exit(1)
