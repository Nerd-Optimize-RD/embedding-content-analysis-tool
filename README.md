# Embedding Content Analysis Tool

A lightweight SEO content analysis tool for comparing your content with top competitors using embeddings and AI insights.

## Key Features

- Single article embedding analysis
- Competitor comparison (Auto SERP API or manual input)
- Content gap detection from embedding dimensions
- AI-generated recommendations (DeepSeek)
- Visual outputs (PCA scatter, similarity, radar, and heatmap)

## Tech Stack

- Backend: Flask (Python)
- Embeddings: Google Gemini API (`gemini-embedding-exp-03-07`)
- Competitor search: SerpAPI
- AI analysis: DeepSeek API
- Content extraction: Trafilatura
- Visualization: Matplotlib, Seaborn, scikit-learn, Chart.js

## Quick Start (Docker)

### 1) Configure environment variables

Copy `.env.example` to `.env` and fill your keys:

```bash
cp .env.example .env
```

Required keys:

- `GOOGLE_API_KEY` (Gemini embeddings)
- `SERPAPI_API_KEY` (competitor search)
- `DEEPSEEK_API_KEY` (AI analysis)

### 2) Build and run

```bash
docker compose up -d --build
```

### 3) Open the app

With current Docker config (`APPLICATION_ROOT=/embedding-content-analysis-tool`), use:

`http://localhost:5001/embedding-content-analysis-tool/`

## Local Run (without Docker)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python embedding.py
```

Default URL (without reverse proxy path): `http://localhost:5001`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | - | Gemini API for embeddings |
| `SERPAPI_API_KEY` | Yes | - | SerpAPI for competitor search |
| `DEEPSEEK_API_KEY` | Yes | - | DeepSeek API for AI recommendations |
| `PORT` | No | `5001` | App port |
| `HOST` | No | `0.0.0.0` | App host |
| `FLASK_DEBUG` | No | `false` | Debug mode |
| `APPLICATION_ROOT` | No | - | Optional app subpath (e.g. `/embedding-content-analysis-tool`) |

## Notes for Open Source Distribution

- Do not commit real API keys in `.env`.
- Keep `.env` local and share only `.env.example`.
- Keep NerdOptimize attribution and license files intact.

## License & Policy

This tool is by **NerdOptimize**.  
See [LICENSE](LICENSE) and [POLICY.md](POLICY.md).

Usage is permitted, but commercial resale is strictly prohibited and NerdOptimize attribution must not be removed.
