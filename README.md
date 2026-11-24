# LCA Knowledge Assistant

**A small research prototype for document-grounded LCA knowledge exploration, candidate ontology population, and explainable resource matching.**

This repository was created as a learning/MVP project inspired by the research themes in the LS2N DUKe 2027 Master's internship description. It is an independent prototype: it is **not** an implementation of the I-DAVE WP3 metamodel, is not affiliated with Nantes Université, and does not reproduce the laboratory's existing work.

## What it does

1. **Document ingestion:** reads PDF, Markdown, and TXT files, splits them into traceable chunks, and keeps source/page metadata.
2. **Retrieval-grounded Q&A:** retrieves relevant chunks using TF-IDF by default. An optional SentenceTransformers backend provides dense semantic retrieval. If an LLM endpoint is enabled, answers are generated from retrieved passages with source references; otherwise the app displays the retrieved evidence as an extractive fallback.
3. **Candidate knowledge extraction:** optionally asks an LLM to extract typed entities and relations into a validated JSON schema. Without an LLM, a small vocabulary-based extractor identifies selected demo terms.
4. **RDF ontology population:** represents extracted candidates and the resource catalogue in RDF/Turtle using RDFLib. The schema is intentionally small and illustrative.
5. **Explainable resource matching:** ranks catalogue entries using text similarity and user-selected criteria. Explicit region, study-phase, and impact-category constraints are enforced as hard filters.
6. **Smoke evaluation:** runs a few hand-labelled retrieval checks and reports Recall@3. This is only a software sanity check, not a research-grade evaluation.

## Important limitations

- The bundled documents and resource catalogue are **synthetic demonstration data**. The catalogue is not a real-world evaluation of LCA software, databases, or impact assessment methods.
- The vocabulary-based extraction fallback is intentionally limited and should not be treated as a general ontology-learning system.
- LLM-generated entities and relations are candidates, not ground truth. Human validation and provenance checks are necessary.
- TF-IDF is the default retrieval/ranking baseline. Dense embeddings are optional.
- A semantic similarity score is not proof of technical suitability. Real recommendations require verified metadata, licences, versions, geographical/temporal coverage, methodological constraints, and domain-expert review.
- The project uses a small example ontology, not the official I-DAVE metamodel or ACV ontology.

## Quick start

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/<your-github-username>/lca-knowledge-assistant.git
cd lca-knowledge-assistant
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e .
streamlit run app.py
```

The application starts with a tiny synthetic corpus and catalogue. Open the **Documents & index** tab and select **Build / rebuild retrieval index**.

## Optional: dense sentence embeddings

```bash
pip install -e '.[embeddings]'
```

Then choose `sentence-transformers` in the sidebar and rebuild the index. The first model use may download the configured model. If it cannot load, the code falls back to TF-IDF.

## Optional: local or hosted LLM

The application accepts an OpenAI-compatible `/chat/completions` endpoint. For a local endpoint, configure the URL and model in the sidebar. For environment-based configuration, copy `.env.example` to `.env` and export the values in your shell (the app does not automatically load `.env`). Example shell configuration:

```bash
export LLM_ENABLED=true
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=<your-local-chat-model>
export LLM_API_KEY=
```

Then enable **LLM endpoint** in the sidebar. The endpoint must expose an OpenAI-compatible chat completions route. Do not commit API keys or private documents.

If the endpoint is disabled or unavailable, the app remains usable with retrieval-only answers and vocabulary-based extraction.

## Add your own documents

Upload PDF, Markdown, or TXT files in the **Documents & index** tab. Keep source files local unless you have permission to publish them. For a public GitHub demo, use openly licensed documents and preserve their original citations/licence notices.

## Run the CLI demonstration

```bash
python scripts/run_demo.py
```

## Run tests

```bash
pip install -e '.[dev]'
pytest -q
```

## Project structure

```text
.
├── app.py
├── data/
│   ├── documents/             # synthetic demo notes
│   └── resources.json         # synthetic resource catalogue
├── ontology/
│   └── lca_schema.ttl         # small illustrative RDF schema
├── scripts/
│   └── run_demo.py
├── src/lca_assistant/
│   ├── catalog.py
│   ├── extraction.py
│   ├── ingest.py
│   ├── llm.py
│   ├── models.py
│   ├── ontology.py
│   ├── pipeline.py
│   ├── recommend.py
│   └── retrieval.py
└── tests/
```

## Possible next research steps

- Replace the synthetic catalogue with verified, source-attributed ACV resource metadata.
- Align the domain schema with a published or supervisor-provided ACV metamodel.
- Compare TF-IDF and multiple embedding models on an independently labelled retrieval set.
- Evaluate LLM extraction with manually annotated entities/relations, precision, recall, and F1.
- Add ontology consistency checks, SHACL constraints, and more robust entity resolution.
- Compare ranking strategies and report constraint satisfaction, ranking quality, sensitivity to weights, and explanations.

## Author

Mohamed Nadjib Taleb · [GitHub](https://github.com/N4dj1b)

## Licence

MIT. The source code is provided for educational and research prototyping. Check the licences of any external documents, models, and data you add.
