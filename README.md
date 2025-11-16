# rao-search

**rao-search** is a Python-based image gallery application that leverages Google Gemini AI to automatically generate captions, tags, and vector embeddings for images. It stores image metadata and embeddings in a PostgreSQL database with the `pgvector` extension, enabling semantic search capabilities on the image library.

***

## Features

- Automatically indexes images in a specified local directory.
- Automatically switches to fallback models if your default models aren't working.
- Uses Google Gemini AI for image analysis: captions, tags, and object detection.
- Stores image metadata, tags, and vector embeddings in PostgreSQL.
- Supports semantic search of images based on natural language queries.
- Handles GIFs by converting them into sprite sheets for analysis.
- Async and concurrent indexing with retry and exponential backoff.
- OAuth2 and API key authentication for Gemini API.
- Colorized logging to console and file.
- Exception handling for robust error management.

***

## Requirements

- Python 3.10+
- PostgreSQL with `pgvector` extension installed
- Google Cloud credentials for Gemini API (OAuth client secrets or API key)
- Packages as specified in code imports (e.g., `colorlog`, `pydantic`, `sqlalchemy[asyncio]`, `google-auth`, `google-genai`, `pgvector`, `pillow`, `python-dotenv`)

***

## Installation

1. Clone the repository.
2. Create a Python virtual environment and activate it. Example: On Linux/MacOS you can type `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install required packages, for example:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file to configure the environment variables or modify the default values in the `Settings` class.
5. Place your Google OAuth client secret JSON in `.secrets/client_secret.json` or set the API key in `.env`. You can change your secret and temp directories with cli arguments or your `.env`.
6. Ensure PostgreSQL is running and accessible with `pgvector` extension installed.

***

## Configuration

The application is configured via environment variables or CLI, with defaults defined in the `Settings` class:

- `PROJECT_ID`: Google Cloud Project ID. (Mandatory if using OAUTH)
- `LOCATION`: Google Cloud location (default `us-central1`). (Mandatory if using OAUTH)
- `CLIENT_SECRETS_FILE`: Path to OAuth client secrets JSON. (Mandatory if using OAUTH). You can find instructions on how to make your client file [here in the Rclone documentation](https://rclone.org/drive/#making-your-own-client-id).
- `TOKEN_DIR`: Directory for OAuth token storage.
- `GOOGLE_API_KEY`: API key for Gemini API (if using API auth).
- `CRED_TYPE`: Authentication type, `"API"` or `"OAUTH"`.
- `DB_URL`: Async PostgreSQL URL with `asyncpg` driver.
- `AI_MODEL`: Gemini AI model for image analysis.
- `EMBEDDING_MODEL`: Model for text embedding.
- `VECTOR_DIMENSION`: Dimensionality of embeddings (default 768).
- `IMAGE_GALLERY`: Directory containing images to index.
- `TEMP_DIRECTORY`: Temp directory for GIF conversions.
- `LOG_DIR`: Directory to save logs.
- `SEMAPHORE_LIMIT`: Number of concurrent indexing tasks.
- `INDEX_TASK_DELAY`: Delay in seconds between indexing tasks.
- `MAX_RETRIES`: Number of retries for Gemini API calls.
- `IMAGE_EXTENSIONS`: Allowed image file extensions. The default is what file types Gemini currently allows.
- `NO_RETRY_DELAY`: Set to True if you don't want tasks to wait before starting.

### Tip
See the `.env.sample` for inspiration in your own config. The `Settings class` of the `app.py` will always have the most up to date selection of env and cli parementers. 

***

## Usage

Run the main application:

```bash
python rao_search.py
```

The application will:

1. Initialize logging and database.
2. Authenticate with Google Gemini using OAuth or API key. API is default.
3. Scan the image gallery directory for new or incomplete images.
4. Index the images asynchronously, performing:
    - Upload to Gemini API.
    - AI analysis for captions, tags, objects.
    - Generate embedding vectors.
    - Store results in PostgreSQL.
5. Perform a sample semantic search.
6. Clean up temporary files.

***

## Code Structure Overview

### 1. Configuration

- `Settings` class manages env vars and defaults with Pydantic.
- Supports CLI args and `.env` files.


### 2. Logging

- Uses `colorlog` for color console logs.
- Writes detailed debug logs to files with timestamped names.


### 3. Exceptions

- Custom exceptions for error handling:
    - `RaoSearchError`, `PermanentAnalysisError`, `DatabaseError`, `ConfigurationError`, `FileProcessingError`.


### 4. Database Models

- SQLAlchemy declarative models:
    - `User`, `Image`, `Tag`, `ImageTagXref`, `FeatureVector`.
- Includes relations and constraints for robust data integrity.


### 5. Pydantic Models

- `ObjectDetection` and `ImageAnalysis` model Gemini AI response structure for validation.


### 6. Async Retry Decorator

- Retries async functions with exponential backoff on transient errors.


### 7. Gemini Service

- Handles authentication (OAuth or API key).
- Wraps Gemini AI API calls for models, file upload, analysis, embedding, and file deletion.
- Error handling and model caching.


### 8. Database Service

- Async PostgreSQL engine and session management.
- Methods for initialization, user creation, image fetching, vector search.


### 9. Image Utilities

- Convert GIFs to sprite sheets for animation analysis.
- Identify MIME types.
- Cleanup temp directory.


### 10. Main Logic

- Process images: upload, analyze, embed, and store in DB.
- Concurrent task runner with semaphore and delay.
- Performs semantic search test and logs results.
- Graceful shutdown with error handling.

***

## Notes

- The application assumes that the `pgvector` extension is enabled in PostgreSQL. Initialization script enables it if missing.
- Handles OAuth authentication headlessly with `NO_BROWSER` flag by prompting for a code.
- Limits tags and objects per image for structured AI output.
- Designed for extensibility with retries and structured logging.
- Supports semantic similarity search using vector L2 distance.
- Free Tier, and therefore `OAUTH` will likely reach its rate limit and quota frequently. Consider a paid tier if needed. See [Gemini documentation](https://ai.google.dev/gemini-api/docs/rate-limits) for latest rate limit information.

***

## Example

After indexing, a search for `"Give me images that include cats."` will display the closest matching images with captions and similarity scores.

```bash

(.venv) $ python app.py -q "Give me images that include cats."
2025-11-16 20:27:19 - INFO - --- Rao Search Application Starting ---
2025-11-16 20:27:19 - INFO - Connecting to database...
2025-11-16 20:27:20 - INFO - Database initialization complete.
2025-11-16 20:27:20 - INFO - Initializing Gemini client with API Key...
2025-11-16 20:27:20 - INFO - Temporary directory cleaned up.
2025-11-16 20:27:20 - INFO - Skipped 10 already indexed images.
2025-11-16 20:27:20 - INFO - Image gallery is fully indexed and up to date.
2025-11-16 20:27:20 - INFO - 
==================================================
PERFORMING SEMANTIC SEARCH TEST
==================================================
2025-11-16 20:27:20 - INFO - Search Query: 'Give me images that include cats.'
2025-11-16 20:27:20 - INFO - Fetching available models from the API...
2025-11-16 20:27:20 - DEBUG - Found 40 AI models and 5 embedding models.
2025-11-16 20:27:20 - INFO - Generating embedding for 1 strings with model 'text-embedding-004'...
  - File: pinterest_836051118369623033.jpg (ID: 3)
    Caption: A white image displays five separate humorous short stories or jokes typed in black text.
    Distance (L2): 0.9853

  - File: pinterest_836051118369657507.jpg (ID: 1)
    Caption: A classical painting of a woman sleeping in a chair, overlaid with a humorous text about naps and waking up twice in one day.
    Distance (L2): 1.0093

  - File: pinterest_836051118369605133.jpg (ID: 8)
    Caption: A text-based meme expressing a feeling of being largely feral and unable to integrate back into society.
    Distance (L2): 1.0106

2025-11-16 20:27:21 - INFO - --- Application Finished ---

```


***
## TODOS
- Investigate if I can somehow use OAUTH to allow the Gemini SDK to read directly from Google Drive/Photos
- Possibly adapt this script to allow generation from other models (maybe using OpenAI adapters)
- Maybe turn this into a proper cli package and then fork this to create a web frontend.
- Investigate making the cli prettier with Rich.
- Investigate if I need to normalize the Distance measurements. :sweat: