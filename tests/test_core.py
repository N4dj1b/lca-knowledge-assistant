from pathlib import Path

from lca_assistant.catalog import load_resources
from lca_assistant.extraction import extract_from_chunks
from lca_assistant.ingest import load_directory
from lca_assistant.models import DocumentChunk, RecommendationRequest
from lca_assistant.ontology import LCA, add_extraction, add_resources, new_ontology
from lca_assistant.recommend import recommend
from lca_assistant.retrieval import Retriever
from rdflib import RDF, RDFS

ROOT = Path(__file__).resolve().parents[1]


def test_demo_documents_load_and_retrieve():
    chunks = load_directory(ROOT / "data" / "documents")
    assert len(chunks) >= 2
    retriever = Retriever()
    retriever.fit(chunks)
    hits = retriever.search("cradle-to-gate study boundary", top_k=3)
    assert hits
    assert any("demo_lca_overview.md" == hit.chunk.source for hit in hits)


def test_vocabulary_extraction_and_rdf_population():
    chunk = DocumentChunk(
        text="The cradle-to-gate boundary may consider climate change and water use categories.",
        source="sample.md",
        chunk_id="abc123",
    )
    extraction = extract_from_chunks([chunk])
    labels = {entity.name for entity in extraction.entities}
    assert "cradle-to-gate" in labels
    assert "climate change" in labels
    graph = add_extraction(new_ontology(), extraction)
    assert (None, RDF.type, LCA.StudyPhase) in graph
    assert (None, RDFS.label, None) in graph


def test_recommendation_respects_hard_constraints():
    resources = load_resources(ROOT / "data" / "resources.json")
    request = RecommendationRequest(
        requirements="food inventory data for a cradle-to-gate product assessment",
        resource_type="Database",
        region="Europe",
        study_phase="cradle-to-gate",
        weights={"semantic": 0.55, "phase": 0.15, "region": 0.15, "category": 0.15},
    )
    ranked, backend = recommend(request, resources)
    assert backend == "tfidf"
    assert ranked
    assert ranked[0].resource.id == "DEMO-DB-01"
    assert all(item.resource.study_phases for item in ranked)


def test_resources_are_added_to_graph():
    resources = load_resources(ROOT / "data" / "resources.json")
    graph = add_resources(new_ontology(), resources)
    assert len(list(graph.subjects(RDF.type, LCA.Resource))) == len(resources)
