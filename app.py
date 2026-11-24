from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from rdflib.namespace import RDFS

from lca_assistant.catalog import load_resources
from lca_assistant.ingest import load_directory, load_uploaded_files
from lca_assistant.llm import LLMClient
from lca_assistant.models import RecommendationRequest
from lca_assistant.ontology import LCA, add_extraction, add_resources, new_ontology, ontology_summary
from lca_assistant.pipeline import answer_question, extract_knowledge
from lca_assistant.recommend import recommend
from lca_assistant.retrieval import Retriever

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "data" / "documents"
CATALOG_PATH = ROOT / "data" / "resources.json"

st.set_page_config(page_title="LCA Knowledge Assistant", page_icon="🌱", layout="wide")
st.title("🌱 LCA Knowledge Assistant")
st.caption("A research prototype combining document retrieval, LLM-assisted extraction, RDF knowledge representation, and explainable resource matching.")

with st.sidebar:
    st.header("Configuration")
    embedding_backend = st.selectbox("Retrieval / matching backend", ["tfidf", "sentence-transformers"], index=0)
    st.caption("SentenceTransformers is optional. Install with: pip install -e '.[embeddings]'.")
    st.divider()
    use_llm = st.checkbox("Enable LLM endpoint", value=False, help="Uses an OpenAI-compatible local or hosted chat-completions endpoint.")
    llm_base_url = st.text_input("LLM base URL", value=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_model = st.text_input("LLM model", value=os.getenv("LLM_MODEL", "llama3.2"))
    llm_api_key = st.text_input("API key (optional for local endpoints)", value=os.getenv("LLM_API_KEY", ""), type="password")
    st.caption("When disabled or unavailable, the app uses an extractive / vocabulary-based fallback.")

llm = LLMClient(base_url=llm_base_url, model=llm_model, api_key=llm_api_key, enabled=use_llm)

if "chunks" not in st.session_state:
    st.session_state.chunks = load_directory(DOCS_DIR)
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "extraction" not in st.session_state:
    st.session_state.extraction = None
if "ontology_ttl" not in st.session_state:
    st.session_state.ontology_ttl = None

resources = load_resources(CATALOG_PATH)

tab_docs, tab_rag, tab_ontology, tab_recommend, tab_eval = st.tabs([
    "1 · Documents & index",
    "2 · Ask the corpus (RAG)",
    "3 · Knowledge extraction & ontology",
    "4 · Resource recommendation",
    "5 · Quick evaluation",
])

with tab_docs:
    st.subheader("Document corpus")
    st.write("Upload LCA-related PDF, Markdown, or text files. The included demo notes are synthetic examples for exercising the prototype.")
    uploaded = st.file_uploader("Add documents", type=["pdf", "md", "txt"], accept_multiple_files=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Add uploaded documents", disabled=not uploaded):
            new_chunks = load_uploaded_files(uploaded)
            st.session_state.chunks.extend(new_chunks)
            st.session_state.retriever = None
            st.success(f"Added {len(new_chunks)} text chunks.")
    with col_b:
        if st.button("Reset to bundled demo corpus"):
            st.session_state.chunks = load_directory(DOCS_DIR)
            st.session_state.retriever = None
            st.session_state.extraction = None
            st.session_state.ontology_ttl = None
            st.success("Restored the bundled synthetic demo corpus.")

    st.metric("Text chunks available", len(st.session_state.chunks))
    if st.session_state.chunks:
        docs_df = pd.DataFrame([
            {"Source": chunk.source, "Page": chunk.page or "—", "Chunk ID": chunk.chunk_id, "Characters": len(chunk.text)}
            for chunk in st.session_state.chunks
        ])
        st.dataframe(docs_df, use_container_width=True, hide_index=True)
        with st.expander("Preview first chunk"):
            st.write(st.session_state.chunks[0].text)
    if st.button("Build / rebuild retrieval index", type="primary", disabled=not st.session_state.chunks):
        retriever = Retriever(backend=embedding_backend)
        with st.spinner("Building retrieval index..."):
            retriever.fit(st.session_state.chunks)
        st.session_state.retriever = retriever
        st.success(f"Index ready using: {retriever.backend_used}")
    if st.session_state.retriever is not None:
        st.info(f"Current index: {len(st.session_state.retriever.chunks)} chunks · {st.session_state.retriever.backend_used}")

with tab_rag:
    st.subheader("Ask questions grounded in retrieved passages")
    if st.session_state.retriever is None:
        st.warning("Build the retrieval index in the Documents tab first.")
    else:
        question = st.text_area("Question", value="What should be considered when selecting resources for a cradle-to-gate study?", height=90)
        top_k = st.slider("Retrieved passages", min_value=1, max_value=8, value=4)
        if st.button("Retrieve and answer", type="primary"):
            with st.spinner("Retrieving evidence and preparing the answer..."):
                answer, metadata = answer_question(st.session_state.retriever, question, llm, top_k=top_k)
            st.markdown(answer)
            if metadata:
                mode, _ = metadata[-1]
                st.caption(f"Answer mode: {mode}")
                st.markdown("**Retrieved evidence**")
                for chunk, score in metadata[:-1]:
                    with st.expander(f"{chunk.source} · page {chunk.page or 'N/A'} · score {score:.3f}"):
                        st.write(chunk.text)

with tab_ontology:
    st.subheader("Extract candidate concepts and populate an RDF graph")
    st.write("The LLM path asks for structured JSON; when disabled, a small vocabulary-based extractor identifies a limited set of known demo terms. Validate extracted concepts before using them as domain knowledge.")
    if not st.session_state.chunks:
        st.warning("Add documents first.")
    else:
        if st.button("Extract knowledge and build ontology", type="primary"):
            with st.spinner("Extracting candidate knowledge..."):
                extraction, mode = extract_knowledge(st.session_state.chunks, llm)
            graph = new_ontology()
            graph = add_extraction(graph, extraction)
            graph = add_resources(graph, resources)
            st.session_state.extraction = extraction
            st.session_state.ontology_ttl = graph.serialize(format="turtle")
            st.session_state.extraction_mode = mode
        if st.session_state.extraction is not None:
            extraction = st.session_state.extraction
            st.caption(f"Extraction mode: {st.session_state.get('extraction_mode', 'unknown')}")
            col1, col2 = st.columns(2)
            col1.metric("Candidate entities", len(extraction.entities))
            col2.metric("Candidate relations", len(extraction.relations))
            if extraction.entities:
                st.dataframe(pd.DataFrame([entity.model_dump() for entity in extraction.entities]), use_container_width=True, hide_index=True)
            if extraction.relations:
                st.dataframe(pd.DataFrame([relation.model_dump() for relation in extraction.relations]), use_container_width=True, hide_index=True)
            else:
                st.info("No explicit relations were produced. The current fallback extracts vocabulary terms only.")
            if st.session_state.ontology_ttl:
                st.download_button("Download populated ontology (.ttl)", data=st.session_state.ontology_ttl, file_name="lca_knowledge_graph.ttl", mime="text/turtle")
                with st.expander("Preview Turtle serialization"):
                    st.code(st.session_state.ontology_ttl[:12000], language="turtle")

with tab_recommend:
    st.subheader("Hybrid, constraint-aware resource matching")
    st.write("This is a transparent ranking prototype over a synthetic catalogue. Selected region, study-phase, and impact-category filters are treated as hard constraints.")
    requirements = st.text_area("Describe the study requirements", value="European food product assessment with a cradle-to-gate boundary; prioritize inventory data and climate change impacts.", height=90)
    resource_type = st.selectbox("Resource type", ["Any", "Software", "Database", "ImpactAssessmentMethod"])
    col1, col2, col3 = st.columns(3)
    with col1:
        region = st.selectbox("Required region (optional)", ["", "Europe", "North America", "Asia", "Global"])
    with col2:
        phase = st.selectbox("Required study phase (optional)", ["", "cradle-to-gate", "cradle-to-grave"])
    with col3:
        category = st.selectbox("Required impact category (optional)", ["", "climate change", "water use"])
    st.caption("Weights apply to semantic relevance and to each criterion that you specify.")
    semantic_weight = st.slider("Semantic similarity weight", 0.0, 1.0, 0.55, 0.05)
    phase_weight = st.slider("Study-phase weight", 0.0, 1.0, 0.15, 0.05)
    region_weight = st.slider("Region weight", 0.0, 1.0, 0.15, 0.05)
    category_weight = st.slider("Impact-category weight", 0.0, 1.0, 0.15, 0.05)
    if st.button("Rank resources", type="primary"):
        try:
            request = RecommendationRequest(
                requirements=requirements,
                resource_type=resource_type,
                region=region,
                study_phase=phase,
                impact_category=category,
                weights={"semantic": semantic_weight, "phase": phase_weight, "region": region_weight, "category": category_weight},
            )
            ranked, backend_used = recommend(request, resources, embedding_backend=embedding_backend)
            st.caption(f"Matching backend: {backend_used}")
            if not ranked:
                st.warning("No resources satisfy all selected hard constraints in the demo catalogue.")
            for index, item in enumerate(ranked, start=1):
                with st.expander(f"#{index} · {item.resource.name} · score {item.total_score:.3f}", expanded=index == 1):
                    st.write(item.resource.description)
                    st.write(f"**Type:** {item.resource.resource_type} · **Regions:** {', '.join(item.resource.regions)}")
                    st.write(f"**Phases:** {', '.join(item.resource.study_phases)} · **Impact categories:** {', '.join(item.resource.impact_categories)}")
                    st.markdown("**Why it appears here**")
                    for explanation in item.explanations:
                        st.write(f"- {explanation}")
                    st.caption(item.resource.source_note)
        except Exception as exc:
            st.error(f"Could not calculate recommendations: {exc}")

with tab_eval:
    st.subheader("Small reproducible smoke evaluation")
    st.write("This checks whether the bundled demo corpus retrieves the expected source for a few hand-labelled queries. It is a smoke test, not a statistically representative evaluation.")
    if st.session_state.retriever is None:
        st.warning("Build the retrieval index first.")
    else:
        evaluation_cases = [
            ("What is a cradle-to-gate study boundary?", "demo_lca_overview.md"),
            ("What should be checked before choosing an LCA resource?", "demo_resources_note.md"),
        ]
        rows = []
        for query, expected_source in evaluation_cases:
            hits = st.session_state.retriever.search(query, top_k=3)
            sources = [hit.chunk.source for hit in hits]
            rows.append({
                "Query": query,
                "Expected source": expected_source,
                "Top-3 sources": ", ".join(sources),
                "Expected source in top-3": expected_source in sources,
            })
        eval_df = pd.DataFrame(rows)
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
        recall_at_3 = eval_df["Expected source in top-3"].mean() if not eval_df.empty else 0.0
        st.metric("Smoke-test Recall@3", f"{recall_at_3:.0%}")
        st.caption("The cases are authored for this tiny synthetic corpus. Add independently labelled queries and real documents before reporting research-grade metrics.")

st.divider()
st.caption("Prototype only · Demo resource records and bundled documents are synthetic · Verify real-world LCA data, versions, licences, boundaries, and methods against authoritative sources.")
