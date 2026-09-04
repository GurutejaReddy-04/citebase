/**
 * CiteBase Frontend Application Logic.
 * Communicates with FastAPI backend at API_BASE.
 * Fully compatible with Track 2 features:
 * - X-API-Key cryptographic authentication & tenant isolation
 * - Asynchronous background ingestion with polling via GET /tasks/{task_id}
 * - Multi-collection querying with Cross-Encoder & Web Search toggles
 * - Rich citation rendering: retrieval mode, cache pills, rerank confidence, channels
 */
const API_BASE = (window.location.origin && window.location.origin !== 'null' && !window.location.origin.startsWith('file'))
  ? window.location.origin
  : 'http://localhost:8000';

/* --- DOM References --- */
const apiKeyInput           = document.getElementById('apiKeyInput');
const toggleKeyVisibilityBtn= document.getElementById('toggleKeyVisibilityBtn');
const connectBtn            = document.getElementById('connectBtn');
const tenantBadge           = document.getElementById('tenantBadge');

const fileInput             = document.getElementById('fileInput');
const collNameInput         = document.getElementById('collNameInput');
const docTitleInput         = document.getElementById('docTitleInput');
const categoryInput         = document.getElementById('categoryInput');
const forceCheckbox         = document.getElementById('forceCheckbox');
const uploadBtn             = document.getElementById('uploadBtn');
const uploadFeedback        = document.getElementById('uploadFeedback');

const collectionSelect      = document.getElementById('collectionSelect');
const deleteBtn             = document.getElementById('deleteBtn');
const resetBtn              = document.getElementById('resetBtn');
const resetConfirm          = document.getElementById('resetConfirm');
const confirmResetBtn       = document.getElementById('confirmResetBtn');
const cancelResetBtn        = document.getElementById('cancelResetBtn');
const manageFeedback        = document.getElementById('manageFeedback');

const queryCollectionSelect = document.getElementById('queryCollectionSelect');
const queryCategoryFilter   = document.getElementById('queryCategoryFilter');
const rerankToggle          = document.getElementById('rerankToggle');
const webFallbackToggle     = document.getElementById('webFallbackToggle');
const questionInput         = document.getElementById('questionInput');
const askBtn                = document.getElementById('askBtn');
const queryFeedback         = document.getElementById('queryFeedback');

const resultsContainer      = document.getElementById('resultsContainer');
const retrievalModeBadge    = document.getElementById('retrievalModeBadge');
const cachePill             = document.getElementById('cachePill');
const answerDisplay         = document.getElementById('answerDisplay');
const sourcesDisplay        = document.getElementById('sourcesDisplay');

/* --- Authentication & Storage --- */
const DEFAULT_DEV_KEY = 'sk_live_dev_test_key_master_12345';

function getApiKey() {
  return (apiKeyInput ? apiKeyInput.value.trim() : '') || localStorage.getItem('citebase_api_key') || DEFAULT_DEV_KEY;
}

function saveApiKey(key) {
  if (apiKeyInput) apiKeyInput.value = key;
  localStorage.setItem('citebase_api_key', key);
}

// Initialize key from localStorage or dev default
if (apiKeyInput) {
  apiKeyInput.value = localStorage.getItem('citebase_api_key') || DEFAULT_DEV_KEY;
}

if (toggleKeyVisibilityBtn) {
  toggleKeyVisibilityBtn.addEventListener('click', () => {
    if (apiKeyInput.type === 'password') {
      apiKeyInput.type = 'text';
      toggleKeyVisibilityBtn.textContent = '🔒';
    } else {
      apiKeyInput.type = 'password';
      toggleKeyVisibilityBtn.textContent = '👁️';
    }
  });
}

if (connectBtn) {
  connectBtn.addEventListener('click', () => {
    const key = apiKeyInput.value.trim();
    if (!key) {
      showFeedback(manageFeedback, 'Please enter a valid API key.', 'warning');
      return;
    }
    saveApiKey(key);
    loadCollections();
  });
}

/* --- Utility: Show Feedback Message --- */
function showFeedback(element, message, type = '') {
  if (!element) return;
  element.innerHTML = message ? `<div class="${type}">${message}</div>` : '';
}

/* --- Utility: Escape HTML for safe rendering --- */
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/* --- Fetch and Populate Collections --- */
async function loadCollections() {
  const apiKey = getApiKey();
  if (!apiKey) {
    if (tenantBadge) {
      tenantBadge.textContent = 'Key Required';
      tenantBadge.className = 'tenant-pill disconnected';
    }
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/collections`, {
      headers: {
        'X-API-Key': apiKey,
      },
    });

    if (response.status === 401) {
      if (tenantBadge) {
        tenantBadge.textContent = 'Invalid Key (401)';
        tenantBadge.className = 'tenant-pill disconnected';
      }
      showFeedback(manageFeedback, 'Authentication failed: Invalid or missing API key.', 'error');
      return;
    }

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    const names = data.collections || [];

    // Update tenant indicator
    if (tenantBadge) {
      tenantBadge.textContent = `Tenant: ${data.tenant || 'default'}`;
      tenantBadge.className = 'tenant-pill';
    }

    // Populate dropdowns
    collectionSelect.innerHTML = names.length > 0
      ? '<option value="">-- select collection --</option>'
      : '<option value="">-- no collections found --</option>';

    queryCollectionSelect.innerHTML = names.length > 0
      ? ''
      : '<option value="">-- no collections found --</option>';

    names.forEach(name => {
      const opt1 = document.createElement('option');
      opt1.value = name;
      opt1.textContent = name;
      collectionSelect.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = name;
      opt2.textContent = name;
      queryCollectionSelect.appendChild(opt2);
    });

    // Auto-select first item if available
    if (names.length > 0 && queryCollectionSelect.options.length > 0) {
      queryCollectionSelect.options[0].selected = true;
    }

    showFeedback(manageFeedback, '', '');
  } catch (error) {
    if (tenantBadge) {
      tenantBadge.textContent = 'Offline / Error';
      tenantBadge.className = 'tenant-pill disconnected';
    }
    showFeedback(manageFeedback, `Failed to load collections: ${error.message}`, 'error');
  }
}

/* --- Asynchronous Document Ingestion Flow --- */
uploadBtn.addEventListener('click', async () => {
  const apiKey = getApiKey();
  const file = fileInput.files[0];
  const collection = collNameInput.value.trim();
  const docTitle = docTitleInput.value.trim();
  const category = categoryInput.value.trim();

  if (!apiKey) return showFeedback(uploadFeedback, 'API key required. Check top header.', 'warning');
  if (!file) return showFeedback(uploadFeedback, 'Choose a PDF file.', 'warning');
  if (!collection) return showFeedback(uploadFeedback, 'Enter a collection name.', 'warning');

  const form = new FormData();
  form.append('file', file);
  form.append('collection_name', collection);
  form.append('force', forceCheckbox.checked);
  if (docTitle) form.append('doc_title', docTitle);
  if (category) form.append('category', category);

  uploadBtn.disabled = true;
  showFeedback(uploadFeedback, '⏳ Submitting document for ingestion...', '');

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      headers: {
        'X-API-Key': apiKey,
      },
      body: form,
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);

    // If backend returns HTTP 202 Accepted (asynchronous processing)
    if (res.status === 202 && data.task_id) {
      const taskId = data.task_id;
      showFeedback(
        uploadFeedback,
        `⏳ Upload accepted. Background ingestion started <span class="task-progress">Task: ${escapeHtml(taskId)}</span>...`,
        ''
      );
      pollIngestionTask(taskId, collection);
    } else {
      // Synchronous dev bypass fallback
      showFeedback(uploadFeedback, `✓ Ingestion complete! Chunks: ${data.total_chunks || 'N/A'}.`, 'success');
      resetUploadForm();
      loadCollections();
      uploadBtn.disabled = false;
    }
  } catch (err) {
    showFeedback(uploadFeedback, `Upload failed: ${err.message}`, 'error');
    uploadBtn.disabled = false;
  }
});

/* Poll async task status until completion or failure */
async function pollIngestionTask(taskId, collectionName) {
  const apiKey = getApiKey();
  let attempts = 0;
  const maxAttempts = 120; // 120 * 1.5s = 180 seconds ceiling
  const startTime = Date.now();

  const pollInterval = setInterval(async () => {
    attempts++;
    const elapsedSeconds = Math.round((Date.now() - startTime) / 1000);

    try {
      const res = await fetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}`, {
        headers: { 'X-API-Key': apiKey },
      });

      if (!res.ok) {
        clearInterval(pollInterval);
        uploadBtn.disabled = false;
        showFeedback(uploadFeedback, `Task status query failed (HTTP ${res.status}).`, 'error');
        return;
      }

      const task = await res.json();

      if (task.status === 'completed') {
        clearInterval(pollInterval);
        uploadBtn.disabled = false;
        showFeedback(
          uploadFeedback,
          `✓ Ingestion completed in ${elapsedSeconds}s! Created <strong>${task.total_chunks}</strong> chunks in collection <code>${escapeHtml(task.collection || collectionName)}</code>.`,
          'success'
        );
        resetUploadForm();
        loadCollections();
      } else if (task.status === 'failed') {
        clearInterval(pollInterval);
        uploadBtn.disabled = false;
        showFeedback(uploadFeedback, `❌ Ingestion failed: ${escapeHtml(task.error_message || 'Unknown processing failure')}`, 'error');
      } else if (task.status === 'stale') {
        clearInterval(pollInterval);
        uploadBtn.disabled = false;
        showFeedback(uploadFeedback, `⚠️ Ingestion task timed out (>5m) or worker restarted. Please retry with force enabled.`, 'warning');
      } else {
        // Still pending or processing
        showFeedback(
          uploadFeedback,
          `⚙️ Extracting, chunking & indexing vectors <span class="task-progress">[${elapsedSeconds}s elapsed]</span>...`,
          ''
        );
      }
    } catch (err) {
      clearInterval(pollInterval);
      uploadBtn.disabled = false;
      showFeedback(uploadFeedback, `Polling error: ${err.message}`, 'error');
    }

    if (attempts >= maxAttempts) {
      clearInterval(pollInterval);
      uploadBtn.disabled = false;
      showFeedback(uploadFeedback, '⚠️ Polling timed out waiting for ingestion task.', 'warning');
    }
  }, 1500);
}

function resetUploadForm() {
  fileInput.value = '';
  collNameInput.value = '';
  docTitleInput.value = '';
  categoryInput.value = '';
  forceCheckbox.checked = false;
}

/* --- Delete Single Collection --- */
deleteBtn.addEventListener('click', async () => {
  const apiKey = getApiKey();
  const name = collectionSelect.value;
  if (!apiKey) return showFeedback(manageFeedback, 'API key required.', 'warning');
  if (!name) return showFeedback(manageFeedback, 'Select a collection to delete.', 'warning');

  try {
    const res = await fetch(`${API_BASE}/collections/${encodeURIComponent(name)}`, {
      method: 'DELETE',
      headers: { 'X-API-Key': apiKey },
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    showFeedback(manageFeedback, data.message || `Collection '${name}' deleted.`, 'success');
    loadCollections();
  } catch (err) {
    showFeedback(manageFeedback, `Delete failed: ${err.message}`, 'error');
  }
});

/* --- Reset All Collections --- */
resetBtn.addEventListener('click', () => { resetConfirm.style.display = 'block'; });
cancelResetBtn.addEventListener('click', () => { resetConfirm.style.display = 'none'; });

confirmResetBtn.addEventListener('click', async () => {
  const apiKey = getApiKey();
  if (!apiKey) return showFeedback(manageFeedback, 'API key required.', 'warning');

  try {
    const res = await fetch(`${API_BASE}/reset?wipe_uploads=false`, {
      method: 'POST',
      headers: { 'X-API-Key': apiKey },
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    showFeedback(manageFeedback, data.message || 'All collections reset.', 'success');
    resetConfirm.style.display = 'none';
    loadCollections();
  } catch (err) {
    showFeedback(manageFeedback, `Reset failed: ${err.message}`, 'error');
  }
});

/* --- Multi-Collection Target Resolution --- */
function getSelectedCollections() {
  const selected = [];
  if (queryCollectionSelect) {
    for (let i = 0; i < queryCollectionSelect.options.length; i++) {
      if (queryCollectionSelect.options[i].selected && queryCollectionSelect.options[i].value) {
        selected.push(queryCollectionSelect.options[i].value);
      }
    }
  }
  return selected;
}

/* --- Query Handler --- */
askBtn.addEventListener('click', async () => {
  const apiKey = getApiKey();
  let question = questionInput.value.trim();
  if ((question.startsWith('"') && question.endsWith('"')) || (question.startsWith("'") && question.endsWith("'"))) {
    question = question.slice(1, -1).trim();
  }
  const selectedCollections = getSelectedCollections();
  const categoryFilter = queryCategoryFilter.value.trim();
  const enableRerank = rerankToggle ? rerankToggle.checked : true;
  const enableWebFallback = webFallbackToggle ? webFallbackToggle.checked : true;

  if (!apiKey) return showFeedback(queryFeedback, 'API key required. Check top header.', 'warning');
  if (!question) return showFeedback(queryFeedback, 'Please enter a question.', 'warning');
  if (selectedCollections.length === 0) {
    return showFeedback(queryFeedback, 'Select or type at least one target collection.', 'warning');
  }

  const payload = {
    question,
    collection_names: selectedCollections,
    enable_rerank: enableRerank,
    enable_web_fallback: enableWebFallback,
  };
  if (categoryFilter) {
    payload.filters = { category: categoryFilter.toLowerCase() };
  }

  askBtn.disabled = true;
  showFeedback(queryFeedback, '🔍 Retrieving context passages & generating grounded answer...', '');

  try {
    const t0 = performance.now();
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,
      },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    const latencyMs = Math.round(performance.now() - t0);
    if (!res.ok) throw new Error(data.detail || res.statusText);

    showFeedback(queryFeedback, '', '');
    resultsContainer.style.display = 'block';

    // 1. Render Retrieval Mode Badge
    const mode = data.retrieval_mode || 'local_document';
    retrievalModeBadge.className = `mode-badge ${mode}`;
    if (mode === 'cached') {
      retrievalModeBadge.innerHTML = '⚡ Cached Response';
    } else if (mode === 'web_fallback') {
      retrievalModeBadge.innerHTML = '🌐 Web Search Fallback';
    } else if (mode === 'blended') {
      retrievalModeBadge.innerHTML = '🔀 Hybrid Blended (Docs + Web)';
    } else {
      retrievalModeBadge.innerHTML = '📄 Local Document';
    }

    // 2. Render Cache Pill
    if (data.cached) {
      cachePill.style.display = 'inline-block';
      cachePill.textContent = `⚡ Cache Hit (${latencyMs}ms)`;
      cachePill.className = 'cache-pill hit';
    } else {
      cachePill.style.display = 'inline-block';
      cachePill.textContent = `⏱️ Live (${latencyMs}ms)`;
      cachePill.className = 'cache-pill live';
    }

    // 3. Render Answer with optional web-fallback banner
    let answerHtml = '';
    if (data.fallback_triggered && mode !== 'cached') {
      answerHtml += `
        <div class="fallback-banner">
          🌐 Web Search Fallback Triggered — Local document confidence was below threshold, so live web results were retrieved.
        </div>`;
    }
    answerHtml += `<div class="answer-text">${escapeHtml(data.answer)}</div>`;
    answerDisplay.innerHTML = answerHtml;

    // 4. Render Sources with rich badges
    const sources = data.sources || [];
    if (sources.length === 0) {
      sourcesDisplay.innerHTML = '<div class="no-sources">No source passages returned.</div>';
    } else {
      sourcesDisplay.innerHTML = sources.map(s => {
        const isWeb = s.source_type === 'web';

        // Score display
        let scoreLabel = '';
        if (s.rerank_score !== undefined && s.rerank_score !== null) {
          scoreLabel = `${(s.rerank_score * 100).toFixed(1)}% rerank score`;
        } else if (s.score !== undefined && s.score !== null) {
          scoreLabel = isWeb ? '100% web match' : `${(Math.max(0, 1 - s.score / 2) * 100).toFixed(1)}% match`;
        }

        // Title and Link
        const titleText = escapeHtml(s.doc_title || s.source || (isWeb ? 'Live Web Result' : 'Document'));
        const titleHtml = isWeb && s.url
          ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" class="source-link">${titleText}</a>`
          : titleText;

        // Badges
        const typeBadge = isWeb
          ? `<span class="badge badge-web">🌐 Web (${escapeHtml(s.provider || 'DDG')})</span>`
          : `<span class="badge badge-collection">📂 ${escapeHtml(s.collection_name || 'Document')}</span>`;

        const pageBadge = (!isWeb && s.page)
          ? `<span class="source-page">Page ${s.page}</span>`
          : '';

        const categoryBadge = (s.category && s.category !== 'uncategorized')
          ? `<span class="badge category-badge">${escapeHtml(s.category)}</span>`
          : '';

        const channelBadge = (s.retrieval_channels && s.retrieval_channels.length > 0)
          ? `<span class="badge badge-channel">${s.retrieval_channels.map(escapeHtml).join(' + ')}</span>`
          : '';

        const breadcrumbBadge = (!isWeb && s.breadcrumb && s.breadcrumb !== 'General')
          ? `<div class="source-breadcrumb">📍 ${escapeHtml(s.breadcrumb)}</div>`
          : '';

        const urlSnippet = (isWeb && s.url)
          ? `<div class="url-snippet">🔗 ${escapeHtml(s.url)}</div>`
          : '';

        return `
          <div class="source-card ${isWeb ? 'web' : 'document'}">
            <div class="source-header">
              <div>
                <span class="source-name">${titleHtml}</span>
                ${pageBadge}
                ${typeBadge}
                ${categoryBadge}
                ${channelBadge}
              </div>
              <span class="source-score">${scoreLabel}</span>
            </div>
            ${urlSnippet}
            ${breadcrumbBadge}
          </div>`;
      }).join('');
    }
  } catch (err) {
    showFeedback(queryFeedback, `Query failed: ${err.message}`, 'error');
  } finally {
    askBtn.disabled = false;
  }
});

/* --- Initial Load on Page Startup --- */
document.addEventListener('DOMContentLoaded', () => {
  loadCollections();
});