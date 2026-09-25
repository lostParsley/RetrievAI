# Minimal End-to-End RAG App

This project is a very small Retrieval-Augmented Generation (RAG) application built for learning. It shows the full flow clearly:

1. Load local documents from a folder.
2. Split them into chunks.
3. Create embeddings for those chunks.
4. Create a Qdrant collection and store vectors plus metadata.
5. Embed a user question.
6. Retrieve the most similar chunks.
7. Generate a grounded answer using only the retrieved context.

The code is intentionally small. It is not a production platform.

## Short implementation plan

1. Put a few documents into `sample_docs/`.
2. Start Qdrant with Docker Compose.
3. Run the FastAPI app.
4. Call `POST /index` to build the vector database.
5. Call `POST /ask` to retrieve chunks and generate a grounded answer.

## Overall architecture in plain English

The app has one API server and one vector database.

- The API server is a FastAPI app.
- The vector database is Qdrant running locally in Docker.
- When you call `/index`, the app reads files from a local folder, breaks them into chunks, turns each chunk into an embedding, creates a Qdrant collection, and stores each chunk as a vector with metadata.
- When you call `/ask`, the app embeds the question, searches Qdrant for the most similar chunks, builds a prompt from only those chunks, and uses a small local text-generation model to produce the final answer.

## Simple RAG diagram

```text
Local files in sample_docs/
        |
        v
LangChain document loaders
        |
        v
Text splitter creates chunks
        |
        v
sentence-transformers creates embeddings
        |
        v
Qdrant stores:
- vector
- chunk text
- metadata (source, chunk_index, page)

User question
        |
        v
sentence-transformers embeds the question
        |
        v
Qdrant similarity search finds top-k chunks
        |
        v
App builds a prompt from retrieved chunks only
        |
        v
Local generation model writes the grounded answer
```

In short, the RAG exists to let the app answer questions from your local documents by first retrieving the most relevant text, then generating an answer from that retrieved context.

## Where each technology is used in this project

- FastAPI handles the `/index`, `/ask`, and `/health` routes.
- LangChain handles document loaders, text splitting, and retrieval wiring.
- sentence-transformers creates embeddings for both document chunks and the user query.
- Qdrant stores vectors plus metadata and performs similarity search.
- Docker Compose starts the Qdrant container locally.

One extra library is also used:

- `transformers` runs a small open-source text-generation model for the final answer step after retrieval.

## Project structure

- `app/main.py`: FastAPI application and the three routes.
- `app/config.py`: configuration loaded from environment variables.
- `app/rag_pipeline.py`: the full RAG lifecycle in one place.
- `app/schemas.py`: request and response models.
- `templates/index.html`: single-page teaching UI.
- `static/styles.css`: minimal styling for the UI.
- `static/app.js`: browser logic that calls the existing API routes.
- `sample_docs/`: sample files you can index immediately.
- `docker-compose.yml`: starts Qdrant locally.
- `requirements.txt`: Python dependencies.
- `.env.example`: example environment variables.
- `README.md`: how the system works and how to run it.
- `AGENTS.md`: repo guidance for future edits.

## Indexing flow

`POST /index` runs these steps:

1. Load documents from the local folder.
   - `.txt` files use `TextLoader`
   - `.md` files use `TextLoader`
   - `.pdf` files use `PyPDFLoader`
2. Split the loaded text into chunks with `RecursiveCharacterTextSplitter`.
3. Create embeddings for each chunk with `sentence-transformers`.
4. Create a Qdrant collection.
5. Upsert vectors and metadata into Qdrant.

Stored metadata for each chunk:

- `source`: file name
- `chunk_index`: chunk number across the indexed set
- `page`: page number for PDFs when available
- `text`: the chunk text itself

In Qdrant, this project stores:

- the embedding vector
- `text` as the chunk body
- `metadata.source`
- `metadata.chunk_index`
- `metadata.page`

## Query flow

`POST /ask` runs these steps:

1. Embed the question with `sentence-transformers`.
2. Retrieve the top-k most similar chunks from Qdrant.
   - LangChain's Qdrant vector store wrapper is used here to keep retrieval code small.
3. Build a prompt using only the retrieved chunks.
4. Generate the answer from that prompt.
5. Return both:
   - the final answer
   - the retrieved chunks and metadata

If the retrieved chunks are too weak, the app says it cannot answer confidently from the context.

## Line-by-line walkthrough of `app/rag_pipeline.py`

The main teaching file is [`app/rag_pipeline.py`](/Users/carlosdiaz/CodexProjects/RAG_Project/app/rag_pipeline.py).

### `SentenceTransformerEmbeddings`

This small adapter exists only because LangChain expects an embeddings object with:

- `embed_documents(...)`
- `embed_query(...)`

The real embedding work is still done by `sentence-transformers`. The adapter just lets LangChain call that model during retrieval.

### `RAGPipeline.__init__`

This sets up the three core runtime components:

- `QdrantClient(...)`
  - connects to the Qdrant server running in Docker
- `SentenceTransformer(...)`
  - loads the embedding model
- `pipeline("text2text-generation", ...)`
  - loads the small local answer-generation model

### `index_documents(...)`

This is the full indexing path:

1. Resolve the document folder path
2. Load files from disk
3. Split them into chunks
4. Turn chunks into Qdrant points
5. Recreate the Qdrant collection
6. Upsert the points into Qdrant
7. Return a summary response

### `_load_documents(...)`

This walks through the local folder and loads only supported file types:

- `.txt`
- `.md`
- `.pdf`

Each loaded file becomes one or more LangChain `Document` objects.

### `_split_documents(...)`

This uses `RecursiveCharacterTextSplitter` to break large documents into smaller chunks. It also normalizes the metadata we want to preserve:

- `source`
- `chunk_index`
- `page`

### `_build_points(...)`

This is where Qdrant storage becomes concrete.

For each chunk, the app creates a `PointStruct` with:

- `id`: a unique ID
- `vector`: the numeric embedding
- `payload.text`: the chunk text
- `payload.metadata`: the source metadata

### `_recreate_collection(...)`

This creates the Qdrant collection with:

- a collection name from `QDRANT_COLLECTION_NAME`
- a vector size equal to the embedding model output dimension
- cosine distance for similarity search

In this teaching project, the collection is deleted and recreated on every `/index` call to keep behavior easy to understand.

### `ask(...)`

This is the query path:

1. Confirm the collection exists
2. Build a LangChain `QdrantVectorStore`
3. Run similarity search with score
4. Convert results into a response-friendly format
5. Build the final prompt
6. Generate the answer

### `_build_prompt(...)`

This creates a plain prompt that includes:

- the question
- the retrieved chunks
- source metadata for each chunk
- an instruction to answer only from the provided context

### `_generate_answer(...)`

This is intentionally conservative:

- if nothing is retrieved, it says so
- if similarity is too low, it says confidence is low
- otherwise it sends the prompt to the small local generation model

## Concrete example with one sample document

Take [`sample_docs/company_handbook.md`](/Users/carlosdiaz/CodexProjects/RAG_Project/sample_docs/company_handbook.md).

It contains this fact:

- employees may work remotely three days per week

During indexing, the app does this:

1. Loads the file from disk
2. Splits it into chunks
   - in your current sample data it stayed as one chunk because it is short
3. Embeds that chunk into a vector
4. Stores it in Qdrant with payload like this:

```json
{
  "id": "some-uuid",
  "vector": [0.12, -0.44, "..."],
  "payload": {
    "text": "# Company Handbook\n\nAcme Analytics allows employees to work remotely three days per week...",
    "metadata": {
      "source": "company_handbook.md",
      "chunk_index": 0,
      "page": null
    }
  }
}
```

Then if you ask:

```json
{"question":"How many remote days per week are allowed?"}
```

the app does this:

1. Embeds the question into a vector
2. Searches Qdrant for nearby chunk vectors
3. Finds the chunk from `company_handbook.md`
4. Builds a prompt containing that chunk
5. Generates the answer from that context

That is the core idea of RAG in this app:

- documents are stored outside the model
- retrieval happens first
- answer generation happens second

## What Qdrant is and how it compares to MySQL or PostgreSQL

Qdrant is a database, but it is not a traditional relational database like MySQL or PostgreSQL.

Think of the comparison like this:

- MySQL/PostgreSQL are optimized for rows, columns, joins, filters, and exact queries
- Qdrant is optimized for vector similarity search

In a regular SQL database, you might ask:

- give me all orders from customer 42
- find all rows where status is `pending`

In Qdrant, you ask:

- find the stored chunks whose vectors are most similar to this question vector

So Qdrant is still a real database, but it is designed around nearest-neighbor search over embeddings rather than relational joins.

Qdrant can also store metadata payloads, which makes it useful for hybrid behavior:

- vector similarity search
- filtering by metadata
- returning stored payload fields with results

For this teaching app, we use the simplest subset:

- one collection
- one vector per chunk
- metadata payload
- cosine similarity search

## How Qdrant is configured in this project

Qdrant is configured in two places:

### 1. Docker Compose

[`docker-compose.yml`](/Users/carlosdiaz/CodexProjects/RAG_Project/docker-compose.yml) starts the Qdrant server locally and maps:

- port `6333` for HTTP API access
- port `6334` for gRPC

It also mounts a Docker volume:

- `qdrant_data:/qdrant/storage`

That volume is where Qdrant persists its collection data on disk.

### 2. Application code

[`app/config.py`](/Users/carlosdiaz/CodexProjects/RAG_Project/app/config.py) provides:

- `QDRANT_URL`
- `QDRANT_COLLECTION_NAME`

[`app/rag_pipeline.py`](/Users/carlosdiaz/CodexProjects/RAG_Project/app/rag_pipeline.py) creates the collection with:

- vector size = embedding model dimension
- distance metric = cosine

## How persistence works in this project

Persistence happens at two levels:

### Qdrant persistence

Qdrant persists data in the Docker volume named `qdrant_data`. That means if you stop and restart the container, the stored collections can still be there.

### App behavior

This app intentionally recreates the collection every time you call `/index`. So even though Qdrant can persist data across restarts, this specific app chooses to replace the collection on each indexing run to keep the learning story simple.

So the rule is:

- stop/start container: data can remain
- call `/index`: collection is rebuilt from current local documents

## How LangChain is being used here

LangChain is deliberately used in a narrow way.

It is used for:

- document loading
- text splitting
- retrieval wiring

Concretely:

- `TextLoader` and `PyPDFLoader` load files from disk
- `RecursiveCharacterTextSplitter` creates chunks
- `QdrantVectorStore` performs similarity retrieval using the existing Qdrant collection

LangChain is not being used here for:

- serving the API
- storing data
- generating embeddings
- running the vector search engine

Those responsibilities belong to:

- FastAPI
- Qdrant
- sentence-transformers

## Ways to interact with Qdrant locally

Yes, you can interact with Qdrant directly while it is running in Docker.

### Option 1: open the local Qdrant web UI

Qdrant exposes a dashboard in the browser when the container is running. Open:

- [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

There you can inspect collections and points.

### Option 2: call the Qdrant HTTP API directly

List collections:

```bash
curl http://localhost:6333/collections
```

Get details for this project's collection:

```bash
curl http://localhost:6333/collections/sample_docs
```

### Option 3: exec into the Docker container

Open a shell inside the running container:

```bash
docker compose exec qdrant sh
```

Inside the container, Qdrant stores its local data under:

```bash
/qdrant/storage
```

That can be useful for basic inspection, but in practice the dashboard and HTTP API are the more useful ways to explore it.

## What Qdrant is capable of beyond this demo

This project uses only the basics, but Qdrant can do more than this demo shows.

Examples:

- store many collections
- filter search results by metadata
- update points incrementally
- delete points selectively
- support larger datasets and approximate nearest-neighbor search
- expose both HTTP and gRPC interfaces
- support hybrid search patterns in broader architectures

## How the vector database is created

The collection is created in `app/rag_pipeline.py`.

- Collection name: from `QDRANT_COLLECTION_NAME`
- Vector size: taken from the embedding model dimension
- Distance metric: cosine similarity
- Payload metadata fields:
  - `text`
  - `metadata.source`
  - `metadata.chunk_index`
  - `metadata.page`

To keep indexing behavior easy to understand, `/index` recreates the collection each time before inserting the current document set.

## Why each file exists

- `app/main.py` keeps the API surface tiny and readable.
- `app/config.py` centralizes the configurable values.
- `app/rag_pipeline.py` shows the full RAG lifecycle without scattering it across many modules.
- `app/schemas.py` makes request and response shapes explicit.
- `sample_docs/` gives you a ready-to-run document set.

## Run locally

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Create a virtual environment and install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Copy the environment file

```bash
cp .env.example .env
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000).
The teaching UI will be available at [http://localhost:8000](http://localhost:8000).

## Example curl commands

Index the sample documents:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{}'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How many remote days per week are allowed?"}'
```

Optional: index a different local folder:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"docs_path":"./sample_docs"}'
```

## Example response shape

`POST /ask` returns:

```json
{
  "answer": "Employees are allowed to work remotely three days per week.",
  "prompt_used": "Answer the question using only the context below...",
  "retrieved_chunks": [
    {
      "score": 0.83,
      "text": "Acme Analytics allows employees to work remotely three days per week.",
      "metadata": {
        "source": "company_handbook.md",
        "chunk_index": 0,
        "page": null
      }
    }
  ]
}
```

## Environment variables

- `QDRANT_URL`: local Qdrant URL
- `QDRANT_COLLECTION_NAME`: collection name
- `DOCS_PATH`: default document folder
- `EMBEDDING_MODEL`: sentence-transformers embedding model
- `ANSWER_MODEL`: small text-generation model used for the final answer
- `CHUNK_SIZE`: chunk size
- `CHUNK_OVERLAP`: chunk overlap
- `TOP_K`: default number of retrieved chunks
- `SIMILARITY_THRESHOLD`: minimum score before answering confidently

## Limitations

- This project recreates the collection on each `/index` call instead of doing incremental updates.
- It is designed for local learning, not for large-scale data or high traffic.
- The answer step uses a very small local model, so quality is limited.
- There is no UI, auth, background job system, or document upload flow.

## Simple next improvements

- Add better prompt formatting with source citations in the answer.
- Cache models at startup more explicitly and add startup logging.
- Add tests for indexing and retrieval behavior.
- Support incremental indexing instead of recreating the collection.
- Add a tiny frontend that calls the same two API routes.
