import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google.auth.credentials import TokenState
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google import genai
from google.genai import types, errors
from google.genai.types import Part
import httpx
from sqlalchemy import (
    exc as sqlalchemy_exc,
    UniqueConstraint,
    Column,
    BigInteger,
    Integer,
    String,
    DateTime,
    Float,
    ForeignKey,
    text,
    select,
    or_,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from pgvector.sqlalchemy import Vector
from PIL import Image as PILImage
import asyncio
import logging
from pydantic import BaseModel, Field, ValidationError
from json import JSONDecodeError
import json
from typing import Optional, Union
import colorlog
from zoneinfo import ZoneInfo

load_dotenv()

CLIENT_SECRETS_FILE = os.getenv("CLIENT_SECRETS_FILE", ".secrets/client_secret.json")
TOKEN_DIR = os.getenv("TOKEN_DIR", ".secrets")
TOKEN_FILE = Path(TOKEN_DIR) / "token.json"
IMAGE_GALLERY = os.getenv("IMAGE_GALLERY", Path.home() / "pictures")
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
REDIRECT_URI = os.getenv("REDIRECT_URI", "urn:ietf:wg:oauth:2.0:oob")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
DB_URL = os.getenv("DB_URL", "sqlite+pysqlite://data.db")
VECTOR_DIMENSION = 768
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif"}
EMBEDDING_MODEL = "embedding-001"
AI_MODEL = "gemini-2.5-flash-lite"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TEMP_DIRECTORY = os.getenv("TEMP_DIRECTORY", "temp")
NO_BROWSER = os.getenv("NO_BROWSER", False)
SEMAPHORE_LIMIT = os.getenv("SEMAPHORE_LIMIT", 3)
INDEX_TASK_DELAY = os.getenv("INDEX_TASK_DELAY", 5)
CRED_TYPE = "API"
NO_RETRY_DELAY = os.getenv("NO_RETRY_DELAY", 0)

os.makedirs("logs", exist_ok=True)

central_tz = ZoneInfo("America/Chicago")
now = datetime.now()
log_file_name = f"app_{now.strftime('%Y-%m-%d_%X')}.log"
log_file_path = Path("logs") / log_file_name

# Get the root logger
logger = colorlog.getLogger()
logger.setLevel(logging.DEBUG)

# Create a colorized formatter
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s%(reset)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

# Create a handler for stream output and set the formatter
stream_handler = colorlog.StreamHandler()
stream_handler.setFormatter(formatter)

# Create a handler for file output
file_handler = logging.FileHandler(log_file_path)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M")
file_handler.setFormatter(file_formatter)

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add handlers to the logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def load_creds():
    creds = None
    token_path = Path(TOKEN_FILE)
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except ValueError as e:
            logging.error(f"Credentials file unsupported: {e}")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Error occured: {e}")
            sys.exit(0)
    else:
        logging.warning(f"Token file unavaliable at {token_path}")

    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES, redirect_uri=REDIRECT_URI)

    if not creds or creds.token_state == (TokenState.INVALID or TokenState.STALE):
        if NO_BROWSER:
            auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
            logging.info("Please visit this URL on another device:")
            logging.info(auth_url)

            code = input("Enter the authorization code here: ")
            flow.fetch_token(code=code)
            try:
                creds = flow.credentials
            except ValueError as e:
                logging.error(f"Credential Token Value Error: {e}")
                sys.exit(0)
            except Exception as e:
                logging.error(f"Credential Token Error: {e}")
                sys.exit(0)
        else:
            creds = flow.run_local_server(open_browser=True)

    # Save token to file after successful auth or refresh
    with open(token_path, "w") as token_file:
        token_file.write(creds.to_json())

    return creds


def get_gemini_client():
    creds = None

    global CRED_TYPE

    if CRED_TYPE == "OAUTH":
        if CLIENT_SECRETS_FILE:
            try:
                creds = load_creds()
                client = genai.Client(credentials=creds, vertexai=True, location=LOCATION, project=PROJECT_ID)
                logging.info("OAUTH Authentication was a success!")
                CRED_TYPE = "OAUTH"
                return client, CRED_TYPE
            except Exception as e:
                logging.error(f"OAuth Configuration failed: {e}")
                sys.exit(0)

    elif GOOGLE_API_KEY or CRED_TYPE == "API":
        try:
            client = genai.Client(api_key=str(GOOGLE_API_KEY))
            logging.info("Client initialized using GOOGLE_API_KEY for Google AI SDK (supporting file upload).")
            CRED_TYPE = "API"
            return client, CRED_TYPE
        except genai.errors.ClientError as e:
            logging.error(f"[Client Error] get_gemini_client: {e}")
            if hasattr(e, "response") and e.response and hasattr(e.response, "text") and e.response.text:
                logging.error(f"API Error Body: {e.response.text}")
            sys.exit(0)
        except errors.APIError as e:
            logging.error(f"[API Error] {e.code}: {e.message}")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Error initializing Google AI Client with API Key: {e}")
            sys.exit(0)


client, CRED_TYPE = get_gemini_client()

# Define the declarative base for all models
Base = declarative_base()

# class GeminiModelManager:


_model_cache = None


async def get_available_ai_models():
    global client, _model_cache

    if _model_cache:
        logging.debug("Returning cached model list.")
        return _model_cache

    logging.info("Fetching available models from the API...")
    """Gets a sorted list of available models that support content generation."""
    ai_models = []
    embed_models = []

    model_list = await client.aio.models.list()

    try:
        for model in model_list:
            if model.supported_actions and "embedContent" in model.supported_actions:
                embed_name = model.name.replace("models/", "")
                embed_models.append(embed_name)

            if (
                model.supported_actions
                and "generateContent" in model.supported_actions
                and "createCachedContent" in model.supported_actions
            ):
                model_name = model.name.replace("models/", "")
                ai_models.append(model_name)

    except Exception as e:
        logging.critical(f"Could not retrieve model list: {e}")
        sys.exit(0)

    embed_models.sort(reverse=True)
    ai_models.sort(reverse=True)

    _model_cache = (ai_models, embed_models)

    logging.debug(f"Number of available embedding models: {len(embed_models)}")
    logging.debug(f"Number of available AI models: {len(ai_models)}")

    return ai_models, embed_models


# 0. New Table: Users ---


class User(Base):
    """
    Table for storing user information.
    """

    __tablename__ = "users"

    user_id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now())

    # Relationships: This links back to all images uploaded by this user
    images = relationship("Image", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.user_id}, username='{self.username}')>"


# 1. Core Table: Images ---


class Image(Base):
    """
    Core table for image metadata.
    """

    __tablename__ = "images"

    image_id = Column(BigInteger, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, unique=True)
    local_file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer)
    width_px = Column(Integer)
    height_px = Column(Integer)
    mime_type = Column(String(50))
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now())
    capture_date = Column(DateTime)

    ai_caption = Column(String(500))
    # Foreign key link to the User who uploaded the image
    user_id = Column(BigInteger, ForeignKey("users.user_id"))

    # Relationships
    tags = relationship("ImageTagXref", back_populates="image", cascade="all, delete-orphan")
    vector = relationship("FeatureVector", uselist=False, back_populates="image", cascade="all, delete-orphan")
    # Relationship to the User object (the 'owner' of this image)
    user = relationship("User", back_populates="images")

    def __repr__(self):
        return f"<Image(id={self.image_id}, filename='{self.filename}')>"


# 2. Tagging Table: Tags ---
class Tag(Base):
    """
    Table defining descriptive tags/labels.
    """

    __tablename__ = "tags"

    tag_id = Column(BigInteger, primary_key=True, autoincrement=True)
    tag_name = Column(String(100), nullable=False, unique=False)
    tag_category = Column(String(50))

    # Relationships
    images = relationship("ImageTagXref", back_populates="tag")

    def __repr__(self):
        return f"<Tag(id={self.tag_id}, name='{self.tag_name}')>"


# 3. Junction Table: ImageTagXref ---
class ImageTagXref(Base):
    """
    Association table for the Many-to-Many relationship between Images and Tags.
    """

    __tablename__ = "image_tag_xref"

    image_tag_id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Foreign Keys
    image_id = Column(BigInteger, ForeignKey("images.image_id"), nullable=False)
    tag_id = Column(BigInteger, ForeignKey("tags.tag_id"), nullable=False)

    confidence = Column(Float)

    # Relationships
    image = relationship("Image", back_populates="tags")
    tag = relationship("Tag", back_populates="images")

    # Combined constraint to ensure an image is not linked to the same tag twice
    __table_args__ = (UniqueConstraint("image_id", "tag_id", name="_image_tag_uc"),)

    def __repr__(self):
        return f"<ImageTagXref(image_id={self.image_id}, tag_id={self.tag_id}, confidence={self.confidence})>"


# 4. Vector Table: FeatureVectors ---
class FeatureVector(Base):
    """
    Stores the numerical vector embeddings for similarity search using pgvector.

    NOTE: The Vector(3) dimension should be changed to match your model's
    output dimension (e.g., Vector(512) or Vector(1024)).
    """

    __tablename__ = "feature_vectors"

    # image_id is the primary key AND the foreign key (1:1 relationship)
    image_id = Column(BigInteger, ForeignKey("images.image_id"), primary_key=True)

    vector_embedding = Column(Vector(VECTOR_DIMENSION), nullable=False)

    model_version = Column(String(50))

    # Relationship (1:1)
    image = relationship("Image", back_populates="vector")

    def __repr__(self):
        return f"<FeatureVector(image_id={self.image_id}, model='{self.model_version}')>"


async def upload_file(image_path, mime_type: str):
    global CRED_TYPE

    logging.info(f"Calling SDK Upload: {image_path.name} (MIME: {mime_type})")
    if CRED_TYPE == "API":
        try:
            file_resource = await client.aio.files.upload(file=image_path, config={"mime_type": mime_type})
            upload_name = file_resource.name
            logging.info(f"Image Successfully Uploaded: {upload_name}")

            return file_resource
        except Exception as e:
            logging.error(f"SDK Upload Failed for {image_path.name}: {e}")
            return None
    elif CRED_TYPE == "OAUTH":
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            logging.info(f"Successfully read {len(image_bytes)} bytes from {image_path}")
            file_resource = {"bytes": image_bytes, "file_name": image_path.name}
            return file_resource
        except IOError as e:
            logging.critical(f"I/O error({e.errno}): {e.strerror}")
            return None
        except Exception as e:
            logging.critical(f"Unable to read image and turn it into bytes: {e}")
            return None
    else:
        logging.warning(f"Cred Type not found. Exiting... {CRED_TYPE}")
        sys.exit(0)


async def delete_file(file_name):
    global CRED_TYPE

    logging.info(f"Calling SDK Delete: {file_name}")
    if CRED_TYPE == "API":
        try:
            await client.aio.files.delete(name=file_name)
            logging.info(f"Deleted: {file_name}")
        except Exception as e:
            logging.error(f"SDK Upload Failed for {file_name}: {e}")
            pass
    elif CRED_TYPE == "OAUTH":
        logging.info(f"Using {CRED_TYPE}, deletion not necessary.")
    else:
        logging.info(f"CRED_TYPE RETURNS: {CRED_TYPE} \n Which is not OAUTH or API.")
        pass


class ObjectDetection(BaseModel):
    """Defines the structure for a single detected object."""

    box_2d: list[int] = Field(description="Normalized coordinates [y0, x0, y1, x1] (0-1000).")
    label: str = Field(description="A descriptive label for the object.")


class ImageAnalysis(BaseModel):
    """Defines the complete structured output from the Gemini analysis call."""

    caption: str = Field(description="A short, concise description about the image.")
    tags: list[str] = Field(description="A list of 3 keywords that describe the image.")
    objects: list[ObjectDetection] = Field(
        default_factory=list,
        description="A list of up to 50 objects detected in the image.",
        max_length=50,
    )


class PermanentAnalysisError(ValueError):
    """
    Custom exception for errors that should not be retried,
    such as content blocks, empty responses, or parsing failures.
    """

    pass


def sync_convert_gif_to_sprite_sheet(gif_path, temp_path, num_frames=5):
    """
    Extracts frames from a GIF using Pillow, stitches them into a
    horizontal sprite sheet, and saves as a PNG.
    """
    with PILImage.open(gif_path) as img:
        total_frames = img.n_frames

        num_to_sample = min(num_frames, total_frames)
        if num_to_sample <= 0:
            logging.error(f"GIF has no frames: {gif_path.name}")
            return None

        # Pick evenly spaced frame indices
        if num_to_sample == 1:
            indices = [0]
        else:
            indices = [int(i * (total_frames - 1) / (num_to_sample - 1)) for i in range(num_to_sample)]
        indices = sorted(list(set(indices)))  # Remove duplicates
        num_to_sample = len(indices)

        frames = []
        for i in indices:
            img.seek(i)
            # Copy and convert to RGBA for clean pasting
            frames.append(img.copy().convert("RGBA"))

        if not frames:
            logging.error(f"Could not extract frames: {gif_path.name}")
            return None

        # Create the new sprite sheet image
        frame_width = frames[0].width
        frame_height = frames[0].height
        total_width = frame_width * num_to_sample

        sprite_sheet = PILImage.new("RGBA", (total_width, frame_height))

        # Paste each frame side-by-side
        for i, frame in enumerate(frames):
            sprite_sheet.paste(frame, (i * frame_width, 0))

        # Save the final PNG
        sprite_sheet.save(temp_path, format="PNG")
        logging.info(f"Saved {num_to_sample}-frame sprite sheet: {temp_path.name}")
        return temp_path


async def convert_gif_to_sprite_sheet(gif_path, temp_path, num_frames=5):
    """
    Async wrapper to run the sync GIF conversion in a separate thread.
    """
    try:
        # Run the blocking PIL code in a thread to avoid blocking asyncio
        return await asyncio.to_thread(sync_convert_gif_to_sprite_sheet, gif_path, temp_path, num_frames)
    except Exception as e:
        logging.error(f"Failed to convert GIF {gif_path.name}: {e}")
        return None


def _parse_retry_delay(e: genai.errors.ClientError) -> Optional[float]:
    """
    Parses the 'retryDelay' from a Google API ClientError response.
    Returns the delay in seconds (as a float) or None if not found.
    """
    if not (hasattr(e, "response") and e.response and hasattr(e.response, "text") and e.response.text):
        return None

    try:
        error_body = json.loads(e.response.text)
        details = error_body.get("error", {}).get("details", [])

        for detail in details:
            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                delay_str = detail.get("retryDelay")

                # Parse strings like "26s" or "26.500510823s"
                if delay_str and delay_str.endswith("s"):
                    try:
                        return float(delay_str[:-1])
                    except (ValueError, TypeError):
                        logging.warning(f"Could not parse retryDelay value: {delay_str}")
                        return None
    except json.JSONDecodeError:
        logging.warning("Could not parse API error body JSON.")
        return None
    except Exception as parse_err:
        logging.warning(f"Unexpected error parsing retryDelay: {parse_err}")
        return None

    return None


async def get_gemini_analysis_with_retries(
    file_resource: Union[types.File, dict],
    uploaded_mime_type: str,
    is_sprite_sheet: bool = False,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    available_ai_models: list = [],
):
    global AI_MODEL

    # available_ai_models, _ = await get_available_ai_models()

    current_model_index = available_ai_models.index(AI_MODEL)

    file_name_for_logging = file_resource["file_name"] if isinstance(file_resource, dict) else file_resource.name

    while current_model_index < len(available_ai_models):
        AI_MODEL = available_ai_models[current_model_index]
        logging.info(f"Using AI model: {AI_MODEL}")

        attempt = 0
        while attempt < max_retries:
            try:
                # This function will be created later
                return await get_gemini_analysis(file_resource, uploaded_mime_type, is_sprite_sheet)

            except PermanentAnalysisError as e:
                logging.critical(
                    f"Permanent failure for {file_name_for_logging} with model {AI_MODEL}: {e}. No retries will be attempted with this model."
                )
                # Break the inner retry loop and move to the next AI model
                break

            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} for model {AI_MODEL} failed for {file_name_for_logging}.")

                sleep_time = 0.0
                if str(NO_RETRY_DELAY) == "1":
                    logging.warning("No Retry Delay, doing next attempt immediately.")
                else:
                    # Default to exponential backoff
                    sleep_time = backoff_factor**attempt
                    if (
                        isinstance(e, genai.errors.ClientError)
                        and hasattr(e, "response")
                        and e.response
                        and e.response.status_code == 429
                    ):
                        logging.warning("[429 Error]: Resource Exhausted. Trying to parse retryDelay...")
                        parsed_sleep_time = _parse_retry_delay(e)
                        if parsed_sleep_time is not None:
                            sleep_time = parsed_sleep_time + 3
                            logging.warning(f"API requested retry in {sleep_time:.2f}s. Waiting...")
                        else:
                            logging.warning("Could not parse retryDelay. Using exponential backoff.")
                    else:
                        logging.warning(f"Transient error: {e}. Using exponential backoff.")

                attempt += 1
                if attempt < max_retries:
                    if sleep_time > 0:
                        logging.info(f"Sleeping {sleep_time:.2f} seconds before retry...")
                    await asyncio.sleep(sleep_time)

        current_model_index += 1

    raise RuntimeError(f"Failed to get valid Gemini analysis for {file_name_for_logging} after trying all available models.")


async def get_embedding_with_retries(contents, max_retries: int = 3, backoff_factor: float = 2.0, available_embed_models: list = []):
    global EMBEDDING_MODEL

    # _, available_embed_models = await get_available_ai_models()

    current_embed_index = available_embed_models.index(EMBEDDING_MODEL)

    while current_embed_index < len(available_embed_models):
        EMBEDDING_MODEL = available_embed_models[current_embed_index]
        logging.info(f"Using embedding model: {EMBEDDING_MODEL}")

        attempt = 0
        while attempt < max_retries:
            try:
                return await get_embedding(contents)

            except PermanentAnalysisError as e:
                logging.critical(
                    f"Permanent failure for with model {EMBEDDING_MODEL}: {e}. No retries will be attempted with this model."
                )
                # Break the inner retry loop and move to the next embedding model
                break

            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} for model {EMBEDDING_MODEL} failed.")

                sleep_time = 0.0
                if str(NO_RETRY_DELAY) == "1":
                    logging.warning("No Retry Delay, doing next attempt immediately.")
                else:
                    # Default to exponential backoff
                    sleep_time = backoff_factor**attempt
                    if (
                        isinstance(e, genai.errors.ClientError)
                        and hasattr(e, "response")
                        and e.response
                        and e.response.status_code == 429
                    ):
                        logging.warning("[429 Error]: Resource Exhausted. Trying to parse retryDelay...")
                        parsed_sleep_time = _parse_retry_delay(e)
                        if parsed_sleep_time is not None:
                            sleep_time = parsed_sleep_time + 3
                            logging.warning(f"API requested retry in {sleep_time:.2f}s. Waiting...")
                        else:
                            logging.warning("Could not parse retryDelay. Using exponential backoff.")
                    else:
                        logging.warning(f"Transient error: {e}. Using exponential backoff.")

                attempt += 1
                if attempt < max_retries:
                    if sleep_time > 0:
                        logging.info(f"Sleeping {sleep_time:.2f} seconds before retry...")
                    await asyncio.sleep(sleep_time)

        current_embed_index += 1

    raise RuntimeError(f"Failed to get valid embedding after trying all available models.")


async def get_gemini_analysis(file_resource: Union[types.File, dict], uploaded_mime_type: str, is_sprite_sheet: bool = False):
    global CRED_TYPE

    file_name_for_logging = file_resource["file_name"] if isinstance(file_resource, dict) else file_resource.name
    logging.info(f"Calling SDK GenerateContent for file: {file_name_for_logging}")

    caption = None
    tags = []
    response = None

    try:
        if is_sprite_sheet:
            prompt_text = "Analyze this image, which is a sprite sheet of 5 frames from a GIF"
            "laid out horizontally from left to right. Describe the full action or "
            "animation from start to finish, then provide the requested structured data."
            "Do not include the phrase sprite sheet in your description."
        else:
            prompt_text = "Analyze the image and provide the requested structured data."

        if CRED_TYPE == "API":
            contents = [Part.from_text(text=prompt_text), file_resource]
        elif isinstance(file_resource, dict):
            # This is the OAUTH case
            contents = [
                Part.from_text(text=prompt_text),
                Part.from_bytes(data=file_resource["bytes"], mime_type=uploaded_mime_type),
            ]
        else:
            logging.critical(f"Invalid file_resource type: {type(file_resource)}. Exiting...")
            sys.exit(0)

        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=ImageAnalysis.model_json_schema(),
            thinking_config=types.ThinkingConfig(include_thoughts=False),
            should_return_http_response=False,
            safety_settings=safety_settings,
        )

        logging.info("Calling client.aio.models.generate_content...")
        response = await client.aio.models.generate_content(model=AI_MODEL, contents=contents, config=config)
        logging.info("Finished calling client.aio.models.generate_content.")

        if not response or not response.text:
            logging.error("Empty response.text from Gemini")
            logging.error(f"Full response object from Gemini: {response}")
            raise PermanentAnalysisError("Gemini AI model returned empty response.text")

        try:
            resp_json = json.loads(response.text)

            if "prompt_feedback" in resp_json and resp_json["prompt_feedback"]["GenerateContentResponsePromptFeedback"].get(
                "block_reason"
            ):
                block_reason = resp_json["prompt_feedback"]["block_reason"]
                logging.error(f"Model blocked request with reason: {block_reason}")
                raise PermanentAnalysisError(f"Gemini AI model blocked request: {block_reason}")

            if not resp_json.get("caption") or not resp_json.get("tags"):
                logging.error("Response missing required keys 'caption' or 'tags'. Treating as empty.")
                raise PermanentAnalysisError("Gemini AI model returned incomplete response")

            analysis_object = ImageAnalysis.model_validate_json(response.text)
            caption = analysis_object.caption
            tags = analysis_object.tags
            objects = analysis_object.objects

            logging.debug(f"Caption: {caption}")
            logging.debug(f"Tags: {tags}")
            logging.debug(f"Found {len(objects)} objects.")

        except (ValidationError, JSONDecodeError) as e:
            logging.error(f"Failed to parse model's JSON response: {e}")
            if response and hasattr(response, "text"):
                logging.error(f"Raw Model Response: {response.text}")
            raise PermanentAnalysisError(f"Failed to parse model's JSON response: {e}")

        logging.info(f"Analysis Success: {caption[:30].strip()}...")
        return caption, tags, objects

    except genai.errors.ClientError as e:
        logging.error(f"[Client Error]: SDK Combined Analysis FAILED for {file_name_for_logging}")
        if hasattr(e, "response") and e.response and hasattr(e.response, "text") and e.response.text:
            try:
                error_body = json.loads(e.response.text)
                error_code = error_body.get("error", {}).get("code")
                error_status = error_body.get("error", {}).get("status")
                logging.error(f"API Error Body: [{error_code}] Status: {error_status}")
            except json.JSONDecodeError:
                logging.error(f"API Error Body: Could not parse JSON: {e.response.text}")
        raise

    except Exception as e:
        if not isinstance(e, PermanentAnalysisError):
            logging.error(f"SDK Combined Analysis FAILED for {file_name_for_logging}: {e}")
        raise


async def get_embedding(contents):
    try:
        logging.info("Generating Embeddings...")
        vector_response = await client.aio.models.embed_content(
            model=EMBEDDING_MODEL, contents=contents, config=types.EmbedContentConfig(output_dimensionality=768)
        )
        if vector_response.embeddings:
            vector_embedding = vector_response.embeddings[0].values
            logging.info("Embedding Success")

            logging.debug(f"Embedding generated with dimension: {len(vector_embedding)}")

            return vector_embedding
        else:
            raise PermanentAnalysisError("EmbedContentResponse returned no embeddings.")

    except Exception as e:
        if not isinstance(e, PermanentAnalysisError):
            error_body = json.loads(e.response.text)
            error_code = error_body.get("error", {}).get("code")
            error_status = error_body.get("error", {}).get("status")

            logging.error(f"SDK Embedding FAILED: [{error_code}] Status: {error_status}")

        raise


async def get_current_models():
    global AI_MODEL
    global EMBEDDING_MODEL

    available_ai_models, available_embed_models = await get_available_ai_models()

    if not available_embed_models:
        raise RuntimeError("No avaliable Embedding Models")

    if EMBEDDING_MODEL not in available_embed_models:
        logging.warning(f"Default model {EMBEDDING_MODEL} not in available list. Using first available: {available_embed_models[0]}")
        EMBEDDING_MODEL = available_embed_models[0]

    if not available_ai_models:
        raise RuntimeError("No available AI models for content generation.")

    if AI_MODEL not in available_ai_models:
        logging.warning(f"Default model {AI_MODEL} not in available list. Using first available: {available_ai_models[0]}")
        AI_MODEL = available_ai_models[0]

    return available_ai_models, available_embed_models, AI_MODEL, EMBEDDING_MODEL


async def get_gemini_analysis_and_vector(
    file_resource: Union[types.File, dict], uploaded_mime_type: str, is_sprite_sheet: bool = False
):
    global AI_MODEL
    global EMBEDDING_MODEL

    available_ai_models, available_embed_models, AI_MODEL, EMBEDDING_MODEL = await get_current_models()

    caption, tags, objects = await get_gemini_analysis_with_retries(
        file_resource, uploaded_mime_type, is_sprite_sheet, available_ai_models=available_ai_models
    )
    vector_embedding = await get_embedding_with_retries(
        [caption] + tags + [obj.label for obj in objects], available_embed_models=available_embed_models
    )
    return caption, tags, vector_embedding


async def get_gemini_analysis_and_vector_with_retries(
    file_resource: Union[types.File, dict],
    uploaded_mime_type: str,
    is_sprite_sheet: bool = False,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
):
    return await get_gemini_analysis_and_vector(file_resource, uploaded_mime_type, is_sprite_sheet)


async def index_image(session: AsyncSession, image_path, user_id):
    await asyncio.sleep(0.05)
    logging.info(f"[Indexing] Starting image: {image_path.name}")

    # CHECK IF IMAGE ALREADY EXISTS ---
    # existing = await session.execute(select(Image).filter_by(filename=image_path.name))

    file_resource = None  # Can be types.File or dict
    temp_image_path = None
    path_to_upload = image_path
    is_sprite_sheet = False
    original_ext = image_path.suffix.lower()  # <-- Has dot: ".jpg"
    orig_mime = ""  # <-- Store original mime type

    try:
        # 1. NORMALIZE FILE AND MIME TYPE ---
        # *** FIX: Check with dot ***
        if original_ext == ".jpg":
            mime_type = "image/jpeg"
            orig_mime = "image/jpeg"

        elif original_ext == ".gif":
            logging.info(f"Converting GIF: {image_path.name}")
            temp_file_name = image_path.with_suffix(".sprite_sheet.png").name

            temp_image_path = Path(TEMP_DIRECTORY) / temp_file_name
            path_to_upload = await convert_gif_to_sprite_sheet(image_path, temp_image_path, num_frames=5)

            if not path_to_upload:
                logging.error(f"[Indexing] Skipping {image_path.name}, GIF conversion failed.")
                return

            mime_type = "image/png"  # Upload is PNG
            orig_mime = "image/gif"  # Original is GIF
            is_sprite_sheet = True

        elif original_ext in IMAGE_EXTENSIONS:
            # *** FIX: Remove dot for mime type ***
            mime_type = f"image/{original_ext.lstrip('.')}"
            orig_mime = mime_type

        else:
            logging.warning(f"[Indexing] Unknown extension '{original_ext}', skipping file {image_path.name}")
            return

        # file_resource is now the full File object or a dict
        file_resource = await upload_file(path_to_upload, mime_type)
        if not file_resource:
            logging.error(f"[Indexing] Skipping {image_path.name} due to upload failure")
            return

        # *** FIX: Correct logic for handling dict (OAuth) vs File (API) ***
        if isinstance(file_resource, dict):
            # OAUTH case: file_resource is {'bytes': ..., 'file_name': ...}
            caption, ai_tags, vector_embedding = await get_gemini_analysis_and_vector_with_retries(
                file_resource, mime_type, is_sprite_sheet
            )
        else:
            # API case: file_resource is types.File object
            caption, ai_tags, vector_embedding = await get_gemini_analysis_and_vector_with_retries(
                file_resource, mime_type, is_sprite_sheet
            )
        # *** End of fix ***

        width, height = 0, 0
        try:
            with PILImage.open(image_path) as img:
                width, height = img.size
            logging.info(f"[Indexing] Dimensions read: {width}x{height}")
        except Exception as e:
            logging.error(f"[Indexing] Warning: Could not read dimensions for {image_path.name}. Error: {e}")

        new_image = Image(
            filename=image_path.name,
            local_file_path=str(image_path.resolve()),
            file_size_bytes=image_path.stat().st_size,
            width_px=width,
            height_px=height,
            mime_type=orig_mime,  # Use the original file's mime type
            user_id=user_id,
            ai_caption=caption,
        )

        with session.no_autoflush:
            session.add(new_image)
            await session.flush()

        new_vector = FeatureVector(
            image=new_image,
            model_version=EMBEDDING_MODEL,
            vector_embedding=vector_embedding,
        )
        with session.no_autoflush:
            session.add(new_vector)
            await session.flush()

        # Truncate tag name to fit column
        tag_name = caption[:97] + "..." if len(caption) > 100 else caption

        tag_result = await session.execute(select(Tag).filter_by(tag_name=tag_name))
        caption_tag = tag_result.scalar_one_or_none()

        if not caption_tag:
            caption_tag = Tag(tag_name=tag_name, tag_category="AI_Caption")
            with session.no_autoflush:
                session.add(caption_tag)
                await session.flush()

        caption_xref = ImageTagXref(image=new_image, tag=caption_tag, confidence=1.0)

        session.add(caption_xref)

        for tag_name_raw in ai_tags:
            tag_name = tag_name_raw[:99]  # Truncate to 100 char limit

            tag_result = await session.execute(select(Tag).filter_by(tag_name=tag_name))
            keyword_tag = tag_result.scalar_one_or_none()

            if not keyword_tag:
                keyword_tag = Tag(tag_name=tag_name, tag_category="AI_Keyword")
                with session.no_autoflush:
                    session.add(keyword_tag)
                    await session.flush()

            keyword_xref = ImageTagXref(image=new_image, tag=keyword_tag, confidence=0.9)
            session.add(keyword_xref)

        logging.info(f"[Indexing] SUCCESS: {image_path.name} prepped for commit.")

    except PermanentAnalysisError as e:
        # This was already logged, just note that we are skipping this file
        logging.error(f"[Indexing] SKIPPED (Permanent Error): {image_path.name}. Error: {e}")
        # Do not re-raise, allow the process to continue with other images

    except genai.errors.ClientError as e:
        logging.error(f"[Client Error] index_image: {e}")
        if hasattr(e, "response") and e.response and hasattr(e.response, "text") and e.response.text:
            logging.error(f"API Error Body: {e.response.text}")
    except errors.APIError as e:
        logging.error(f"[Indexing Error] {e.code}: {e.message}")

    except Exception as e:
        logging.error(f"[Indexing] An error occurred while indexing {image_path.name}: {e}")
        raise  # Re-raise to trigger rollback in controlled_index_task

    finally:
        # *** FIX: Only delete if file_resource is a types.File (API case) ***
        if file_resource and not isinstance(file_resource, dict):
            await delete_file(file_resource.name)
        elif file_resource and isinstance(file_resource, dict):
            logging.info(f"OAUTH upload, no cloud file to delete for {file_resource.get('file_name')}.")

        if temp_image_path is not None:
            try:
                if isinstance(temp_image_path, Path) and temp_image_path.exists():
                    temp_image_path.unlink()
                    logging.info(f"Cleaned up temp file: {temp_image_path.name}")
            except OSError as e:
                logging.error(f"Failed to clean up temp file: {e}")


async def scan_image_dir_skip(Session: async_sessionmaker):
    gallery_dir = Path(IMAGE_GALLERY)

    image_paths_on_disk = []
    for p in gallery_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths_on_disk.append(p)
        elif p.is_file():
            logging.info(f"[Scanning] SKIPPED: {p.name} (Not an image file)")

    if not image_paths_on_disk:
        logging.warning("[Setup] No images found to index in the gallery directory.")
        return []

    # Get all filenames from disk
    filenames_on_disk = {p.name for p in image_paths_on_disk}

    # Get all existing filenames from DB in one query
    async with Session() as session:
        result = await session.execute(select(Image.filename).where(Image.filename.in_(filenames_on_disk)))
        existing_filenames_in_db = {row[0] for row in result}

    # Find the difference
    new_filenames = filenames_on_disk - existing_filenames_in_db

    # Create a map of filename to path for quick lookup
    path_map = {p.name: p for p in image_paths_on_disk}

    checked_image_paths = [path_map[name] for name in new_filenames]

    num_skipped = len(filenames_on_disk) - len(new_filenames)

    if num_skipped > 0:
        logging.info(f"[Scanning] Skipped {num_skipped} already indexed images.")

    if checked_image_paths:
        logging.info(f"[Scanning] Found {len(checked_image_paths)} new images to index.")

    return checked_image_paths


async def run_indexing_async(Session: async_sessionmaker, user_id, image_paths_to_index):

    # This semaphore belongs to the outer function
    sem = asyncio.Semaphore(int(SEMAPHORE_LIMIT))

    # DEFINE THE TASK FUNCTION *INSIDE* ---
    # This lets it access 'sem' and 'Session'
    async def controlled_index_task(path, user_id, is_last=False):

        # FIX: Define a maximum number of retries
        max_retries = 2  # Try once, then retry one more time

        # FIX: The semaphore must be acquired *outside* the retry loop
        # so that a retrying task still holds its "slot".
        async with sem:

            # FIX: Add a 'for' loop to handle retries
            for attempt in range(max_retries):
                try:
                    # FIX: A new session should be created *inside* the loop
                    # so that each attempt gets a fresh, clean session.
                    async with Session() as session:
                        await index_image(session, path, user_id)
                        await session.commit()

                    # If we get here, it was successful.
                    # We log a success *only if* it wasn't the first attempt.
                    if attempt > 0:
                        logging.info(f"[Indexing] SUCCESS (on retry {attempt}): {path.name}")

                    break  # <-- Success, break the retry loop

                except asyncio.CancelledError:
                    logging.info(f"Task for {path.name} was cancelled.")
                    # No rollback needed, session is closed by 'async with'
                    raise  # Re-raise to stop processing

                # FIX: Specifically catch database errors (like the cache error)
                except sqlalchemy_exc.DBAPIError as e:
                    # We check the error string for our *specific* transient error
                    if "InvalidCachedStatementError" in str(e):
                        logging.warning(
                            f"[Indexing] RETRYING: {path.name} (Attempt {attempt + 1}/{max_retries}). Hit InvalidCachedStatementError."
                        )

                        if attempt + 1 == max_retries:
                            # If this was the last attempt, log final failure
                            logging.error(f"[Indexing] FAILED (Final Attempt): {path.name} after cache error. Error: {e}")
                        else:
                            await asyncio.sleep(1)  # Wait 1s before retrying

                    else:
                        # It was a *different* database error (e.g., "table not found")
                        # This is permanent, so we log and break the retry loop.
                        logging.error(f"[Indexing] FAILED/ROLLED BACK (DB Error): {path.name}. Error: {e}")
                        break  # Stop retrying for this file

                except Exception as e:
                    # This catches non-DB errors (e.g., file read, API failure)
                    # that were not caught by index_image's own logic.
                    logging.error(f"[Indexing] FAILED/ROLLED BACK (General Error): {path.name}. Error: {e}")
                    break  # Stop retrying for this file

            # FIX: This 'finally' logic (the rate-limit delay) is now
            # moved *outside* the retry loop, but *still inside* the 'async with sem' block.
            # This ensures we wait *after* an image is fully processed (or failed).
            if not is_last:
                logging.info(f"[Concurrency] Waiting {INDEX_TASK_DELAY}s to respect rate limit...")
                # Convert to int in case it's read from .env as str
                await asyncio.sleep(int(INDEX_TASK_DELAY))

    # THIS BLOCK RUNS THE TASKS (Unchanged from your original)
    tasks = []
    try:
        logging.info(f"[Concurrency] Limiting concurrent indexing tasks to {SEMAPHORE_LIMIT}.")
        num_tasks = len(image_paths_to_index)
        for i, path in enumerate(image_paths_to_index):
            is_last = i == num_tasks - 1
            tasks.append(controlled_index_task(path, user_id, is_last=is_last))

        return await asyncio.gather(*tasks)
    except Exception as e:
        logging.error(f"[Indexing] FATAL ERROR during concurrent run: {e}")
        return tasks


async def find_and_delete_incomplete_entries(Session: async_sessionmaker):
    """
    Finds and deletes images in the database that are missing an AI caption, AI-generated tags, or a vector embedding.
    Returns a list of file paths for the deleted images to be re-indexed.
    """
    paths_to_rescan = []
    async with Session() as session:
        # Subquery to find image_ids that have at least one AI-generated keyword tag.
        subquery = select(ImageTagXref.image_id).join(Tag).where(Tag.tag_category == "AI_Keyword").distinct()

        # Query for images that are missing a caption, AI tags, or a vector.
        stmt = (
            select(Image)
            .outerjoin(Image.vector)
            .where(
                or_(
                    (Image.ai_caption == None) | (Image.ai_caption == ""),
                    Image.image_id.notin_(subquery),
                    FeatureVector.image_id == None,
                )
            )
        )

        result = await session.execute(stmt)
        incomplete_images = result.scalars().unique().all()

        if not incomplete_images:
            return []

        logging.info(f"[Scanning] Found {len(incomplete_images)} incomplete entries. They will be deleted and re-indexed.")

        for image in incomplete_images:
            image_path = Path(image.local_file_path)
            if image_path.exists():
                paths_to_rescan.append(image_path)
                await session.delete(image)
            else:
                logging.warning(f"[Scanning] Incomplete entry found, but file is missing: {image.local_file_path}. Deleting from DB.")
                await session.delete(image)

        await session.commit()

    return paths_to_rescan


async def embed_text_query(query, task_type: str = "SEMANTIC_SIMILARITY"):
    logging.info(f"[Embed Utility] Generating vector for text query: '{query[:30]}...' (Type: {task_type})")

    try:
        vector_response = await client.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query],
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
        )

        if not vector_response.embeddings:
            logging.error(f"[Embed Utility] ERROR: Could not generate embedding for query: {query}")
            return None
        query_vector = vector_response.embeddings[0].values

        logging.info(f"[Embed Utility] Query vector generated (Dim: {len(query_vector)}).")
        return query_vector

    except genai.errors.ClientError as e:
        logging.error(f"[Client Error] get_gemini_client: {e}")

        if hasattr(e, "response") and e.response and hasattr(e.response, "text") and e.response.text:
            logging.error(f"API Error Body: {e.response.text}")

    except errors.APIError as e:
        logging.error(f"[Embedding Error] {e.code}: {e.message}")

    except Exception as e:
        logging.error(f"[Embed Utility] ERROR: Could not generate embedding for query: {e}")
        return None


async def find_similar_images(Session: async_sessionmaker, query, limit: int = 5):
    query_vector = await embed_text_query(query, task_type="RETRIEVAL_QUERY")
    if not query_vector:
        return []

    results = []
    # Use async session ---
    async with Session() as session:
        try:
            distance = FeatureVector.vector_embedding.l2_distance(query_vector).label("distance")

            # Use SQLAlchemy 2.0 select() style ---
            query_stmt = (
                select(Image, distance)
                .join(FeatureVector, Image.image_id == FeatureVector.image_id)
                # pgvector sorts ASC by default (closest first)
                .order_by(distance)
                .limit(limit)
            )

            logging.info("[Search] Executing vector similarity query...")

            # Execute the query
            query_result = await session.execute(query_stmt)
            raw_results = query_result.all()  # .all() is sync, gets results from cursor

            for image, dist_val in raw_results:
                results.append(
                    {
                        "image_id": image.image_id,
                        "filename": image.filename,
                        "ai_caption": image.ai_caption,
                        "local_file_path": image.local_file_path,
                        "similarity_distance": dist_val,
                    }
                )
        except genai.errors.ClientError as e:
            logging.error(f"[Client Error] get_gemini_client: {e}")
            if hasattr(e, "response") and e.response and hasattr(e.response, "text") and e.response.text:
                logging.error(f"API Error Body: {e.response.text}")
        except errors.APIError as e:
            logging.error(f"[Vector Error] {e.code}: {e.message}")
        except ValueError as e:
            logging.critical(f"[Search] FATAL VALUE ERROR: {e[:20]}")
        except Exception as e:
            logging.critical(f"[Search] FATAL ERROR during vector query: {e}")
            # No rollback needed on SELECT, session closes automatically

    return results


async def initialize_database(db_url):
    if not db_url.startswith("postgresql+asyncpg"):
        logging.warning(f"Database URL does not look like an asyncpg URL: {db_url}")
        if db_url.startswith("sqlite"):
            logging.error("SQLite cannot be used with this async script. Please use PostgreSQL.")
            sys.exit(1)

    logging.info(f"Connecting to database at {db_url.split('@')[-1]}")

    engine = create_async_engine(
        db_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=20,
        max_overflow=0,
        connect_args={"ssl": True},
    )

    try:
        # Use async connection to run setup ---
        async with engine.begin() as conn:
            logging.info("Enabling pgvector extension if not already present...")
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            logging.info("Creating tables if they do not exist...")
            await conn.run_sync(Base.metadata.create_all)

    except Exception as e:
        logging.error(f"FATAL: Could not connect or enable 'vector' extension. Error: {e}")
        sys.exit(1)

    logging.info("Database initialization complete.")

    # Create async sessionmaker ---
    Session = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,  # Important for accessing objects after commit
        class_=AsyncSession,
    )
    return engine, Session


async def temp_dir_cleanup():
    global TEMP_DIRECTORY

    dir_path = Path(TEMP_DIRECTORY)

    if not dir_path.is_dir():
        logging.error(f"Error: Directory not found at {dir_path}")
    else:
        for item in dir_path.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                    logging.debug(f"Deleted file: {item.name}")
                except OSError as e:
                    logging.debug(f"Error deleting {item.name}: {e}")

    logging.info("Finished deleting files.")


async def main():
    """Main asynchronous function to run the script."""
    os.makedirs(Path(TEMP_DIRECTORY), exist_ok=True)

    engine, Session = await initialize_database(DB_URL)

    gallery_dir = Path(IMAGE_GALLERY)
    if not gallery_dir.is_dir():
        logging.error(f"[ERROR] The directory '{gallery_dir}' does not exist.")
        sys.exit(1)

    logging.info(f"[Setup] Scanning image directory for new and incomplete files: {gallery_dir}")

    # Find new images that are on disk but not in the DB
    new_image_paths = await scan_image_dir_skip(Session)

    # Find and delete incomplete entries from the DB, and get their paths for re-indexing
    incomplete_paths_to_rescan = await find_and_delete_incomplete_entries(Session)

    # Combine lists and remove duplicates
    final_paths_to_index = list(set(new_image_paths + incomplete_paths_to_rescan))

    if final_paths_to_index:
        mock_user = None
        # Async setup for mock user ---
        async with Session() as setup_session:
            try:
                user_result = await setup_session.execute(select(User).filter_by(username="image_indexer_bot"))
                mock_user = user_result.scalar_one_or_none()

                if not mock_user:
                    logging.info("Creating mock user 'image_indexer_bot'...")
                    mock_user = User(username="image_indexer_bot", email="indexer@example.com")
                    setup_session.add(mock_user)
                    await setup_session.commit()
                    logging.info(f"Mock user created with ID: {mock_user.user_id}")
                else:
                    logging.info(f"Found existing mock user with ID: {mock_user.user_id}")
            except Exception as e:
                logging.error(f"Failed to create or find mock user: {e}")
                await setup_session.rollback()
                sys.exit(1)
        tasks = None
        try:
            logging.info(f"[Indexing] Starting to index {len(final_paths_to_index)} new and/or incomplete images...")
            tasks = await run_indexing_async(Session, mock_user.user_id, final_paths_to_index)
        except KeyboardInterrupt:
            logging.info("Indexing interrupted by user.")
            if tasks:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await temp_dir_cleanup()
            sys.exit(0)
    else:
        logging.info("[Indexing] No new or incomplete images to index.")

    logging.info("\n" + "=" * 50)
    logging.info("PERFORMING SEMANTIC SEARCH TEST")
    print("=" * 50)

    search_query = "Anything featuring a woman"

    search_results = await find_similar_images(Session, search_query, limit=3)

    if search_results:
        print(f"\n[Search Results for '{search_query}']")
        for result in search_results:
            print(f"  - File: {result['filename']} (ID: {result['image_id']})")
            print(f"    - Caption: {result['ai_caption']}")
            print(f"    - Local Path: {result['local_file_path']}")
            print(f"    - Distance (L2): {result['similarity_distance']:.4f}")
    else:
        logging.info("\nNo search results found.")

    # Test caching
    logging.info("\n" + "=" * 50)
    logging.info("PERFORMING SECOND SEARCH (TESTING CACHE)")
    print("=" * 50)

    search_results_cached = await find_similar_images(Session, search_query, limit=3)
    if search_results_cached:
        logging.info(f"[Cached Search Results for '{search_query[:5].strip()}']")
        logging.debug("(This should have been faster and not logged a 'Fallback' message in embed_text_query)")
    else:
        logging.fino("\nNo search results found.")

    # Clean up engine resources
    await engine.dispose()


if __name__ == "__main__":
    # Run the main async function ---
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.debug("Script terminated by user.")
        # No need to call temp_dir_cleanup() here as it is handled in main()
        sys.exit(0)
