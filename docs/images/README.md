# CiteBase Documentation Assets

This directory contains static visual assets and screenshots embedded across the project documentation.

## Asset Directory

| File | Resolution | Description | Embedded Location |
|---|---|---|---|
| `swagger-ui.png` | 1280x960 | Live interactive OpenAPI / Swagger UI endpoints | `README.md` (API Reference) |
| `frontend-ui.png` | 1280x850 | Live Document Intelligence web portal UI | `README.md` (Quick Start) |
| `terminal.png` | 960x520 | Automated test suite execution (44/44 passed) & static analysis | `README.md` (Running Tests) |
| `query-response.png` | 960x530 | Sample JSON response payload with bracketed citations | `README.md` (API Reference) |

---

## How to Update Screenshots

### 1. Live Swagger UI
1. Launch the API service:
   ```bash
   docker compose up -d
   # or locally:
   cd backend && uvicorn main:app --port 8000
   ```
2. Capture via headless browser CLI or browser window:
   ```bash
   msedge --headless --screenshot=docs/images/swagger-ui.png --window-size=1280,960 http://localhost:8000/docs
   ```

### 2. Live Frontend Portal UI
1. Open [http://localhost:8000](http://localhost:8000) or open `frontend/index.html` in your browser.
2. Capture window using Snipping Tool (`Win + Shift + S`) or headless Edge:
   ```bash
   msedge --headless --screenshot=docs/images/frontend-ui.png --window-size=1280,850 http://localhost:8000/frontend/index.html
   ```

### 3. Terminal Test Suite
Run the test suite and capture the terminal output:
```bash
pytest tests/ -v --tb=short
```
