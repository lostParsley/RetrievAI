const healthStatus = document.getElementById("health-status");
const indexButton = document.getElementById("index-button");
const askButton = document.getElementById("ask-button");
const docsPathInput = document.getElementById("docs-path");
const questionInput = document.getElementById("question");
const topKInput = document.getElementById("top-k");

const indexMessage = document.getElementById("index-message");
const askMessage = document.getElementById("ask-message");
const indexSummary = document.getElementById("index-summary");
const indexedChunks = document.getElementById("indexed-chunks");
const answerBox = document.getElementById("answer-box");
const retrievedChunks = document.getElementById("retrieved-chunks");
const promptUsed = document.getElementById("prompt-used");

function setMessage(element, text, kind = "") {
  element.textContent = text;
  element.className = `message ${kind}`.trim();
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("Health check failed");
    }

    const data = await response.json();
    healthStatus.textContent = data.status;
  } catch (error) {
    healthStatus.textContent = "unavailable";
  }
}

function renderIndexSummary(data) {
  indexSummary.classList.remove("empty");
  indexSummary.innerHTML = `
    <div class="summary-grid">
      <div class="summary-card">
        <span>Collection</span>
        <strong>${escapeHtml(data.collection_name)}</strong>
      </div>
      <div class="summary-card">
        <span>Documents loaded</span>
        <strong>${data.documents_loaded}</strong>
      </div>
      <div class="summary-card">
        <span>Chunks indexed</span>
        <strong>${data.chunks_indexed}</strong>
      </div>
      <div class="summary-card">
        <span>Embedding model</span>
        <strong>${escapeHtml(data.embedding_model)}</strong>
      </div>
      <div class="summary-card">
        <span>Chunk size</span>
        <strong>${data.chunk_size}</strong>
      </div>
      <div class="summary-card">
        <span>Chunk overlap</span>
        <strong>${data.chunk_overlap}</strong>
      </div>
    </div>
  `;
}

function renderIndexedChunks(chunks) {
  if (!chunks.length) {
    indexedChunks.className = "list empty";
    indexedChunks.innerHTML = "<p>No chunks were indexed.</p>";
    return;
  }

  indexedChunks.className = "list";
  indexedChunks.innerHTML = chunks
    .map(
      (chunk) => `
        <article class="chunk-card">
          <div class="meta-row">
            <span><strong>Source:</strong> ${escapeHtml(chunk.source)}</span>
            <span><strong>Chunk:</strong> ${chunk.chunk_index}</span>
            <span><strong>Page:</strong> ${chunk.page ?? "n/a"}</span>
          </div>
          <p>${escapeHtml(chunk.text_preview)}</p>
        </article>
      `
    )
    .join("");
}

function renderAnswer(data) {
  answerBox.classList.remove("empty");
  answerBox.innerHTML = `<p>${escapeHtml(data.answer)}</p>`;
  promptUsed.textContent = data.prompt_used;
}

function renderRetrievedChunks(chunks) {
  if (!chunks.length) {
    retrievedChunks.className = "list empty";
    retrievedChunks.innerHTML = "<p>No chunks were retrieved.</p>";
    return;
  }

  retrievedChunks.className = "list";
  retrievedChunks.innerHTML = chunks
    .map(
      (chunk) => `
        <article class="retrieved-card">
          <div class="meta-row">
            <span class="retrieved-score">score: ${chunk.score.toFixed(4)}</span>
            <span><strong>Source:</strong> ${escapeHtml(chunk.metadata.source)}</span>
            <span><strong>Chunk:</strong> ${chunk.metadata.chunk_index}</span>
            <span><strong>Page:</strong> ${chunk.metadata.page ?? "n/a"}</span>
          </div>
          <p>${escapeHtml(chunk.text)}</p>
        </article>
      `
    )
    .join("");
}

async function indexDocuments() {
  indexButton.disabled = true;
  setMessage(indexMessage, "Indexing documents into Qdrant...", "");

  try {
    const docsPath = docsPathInput.value.trim();
    const response = await fetch("/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ docs_path: docsPath || null }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Indexing failed.");
    }

    renderIndexSummary(data);
    renderIndexedChunks(data.indexed_chunks);
    setMessage(
      indexMessage,
      `Indexed ${data.documents_loaded} documents into collection ${data.collection_name}.`,
      "success"
    );
  } catch (error) {
    setMessage(indexMessage, error.message, "error");
  } finally {
    indexButton.disabled = false;
  }
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    setMessage(askMessage, "Please enter a question first.", "error");
    return;
  }

  askButton.disabled = true;
  setMessage(askMessage, "Retrieving chunks and generating an answer...", "");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Number(topKInput.value) || 3,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Question failed.");
    }

    renderAnswer(data);
    renderRetrievedChunks(data.retrieved_chunks);
    setMessage(
      askMessage,
      `Retrieved ${data.retrieved_chunks.length} chunks and produced a grounded answer.`,
      "success"
    );
  } catch (error) {
    setMessage(askMessage, error.message, "error");
  } finally {
    askButton.disabled = false;
  }
}

indexButton.addEventListener("click", indexDocuments);
askButton.addEventListener("click", askQuestion);
checkHealth();
