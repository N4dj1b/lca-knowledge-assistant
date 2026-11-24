"""Run a small command-line demonstration without starting Streamlit."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lca_assistant.catalog import load_resources
from lca_assistant.ingest import load_directory
from lca_assistant.models import RecommendationRequest
from lca_assistant.ontology import add_extraction, add_resources, new_ontology, ontology_summary
from lca_assistant.extraction import extract_from_chunks
from lca_assistant.recommend import recommend
from lca_assistant.retrieval import Retriever


def main() -> None:
    chunks = load_directory(ROOT / "data" / "documents")
    retriever = Retriever()
    retriever.fit(chunks)
    print(f"Indexed {len(chunks)} chunks using {retriever.backend_used}.")
    for hit in retriever.search("What does cradle-to-gate mean?", top_k=2):
        print(f"\n[{hit.chunk.source} | score={hit.score:.3f}]\n{hit.chunk.text}")

    extraction = extract_from_chunks(chunks)
    resources = load_resources(ROOT / "data" / "resources.json")
    graph = add_resources(add_extraction(new_ontology(), extraction), resources)
    print("\nOntology summary:", ontology_summary(graph))

    request = RecommendationRequest(
        requirements="European food inventory database for a cradle-to-gate assessment",
        resource_type="Database",
        region="Europe",
        study_phase="cradle-to-gate",
        weights={"semantic": 0.55, "phase": 0.15, "region": 0.15, "category": 0.15},
    )
    ranked, backend = recommend(request, resources)
    print(f"\nRecommendations ({backend}):")
    for item in ranked:
        print(f"- {item.resource.name}: {item.total_score:.3f}")
        for explanation in item.explanations:
            print(f"    • {explanation}")


if __name__ == "__main__":
    main()
