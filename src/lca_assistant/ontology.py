"""A small RDF ontology and provenance-aware population utilities."""

from __future__ import annotations

import re
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import DCTERMS, XSD

from .models import KnowledgeExtraction, Resource

LCA = Namespace("https://example.org/lca-knowledge#")

CLASS_NAMES = [
    "Study",
    "StudyRequirement",
    "Resource",
    "Software",
    "Database",
    "ImpactAssessmentMethod",
    "ImpactCategory",
    "StudyPhase",
    "Organization",
    "Other",
]

PREDICATE_MAP = {
    "mentions": LCA.mentions,
    "usesDatabase": LCA.usesDatabase,
    "usesMethod": LCA.usesMethod,
    "coversCategory": LCA.coversCategory,
    "hasStudyPhase": LCA.hasStudyPhase,
    "describedIn": DCTERMS.source,
    "relatedTo": LCA.relatedTo,
}


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value or "unnamed"


def new_ontology() -> Graph:
    graph = Graph()
    graph.bind("lca", LCA)
    graph.bind("rdfs", RDFS)
    graph.bind("dcterms", DCTERMS)
    graph.bind("xsd", XSD)
    for class_name in CLASS_NAMES:
        graph.add((LCA[class_name], RDF.type, RDFS.Class))
        graph.add((LCA[class_name], RDFS.label, Literal(class_name)))
    for child_class in ["Software", "Database", "ImpactAssessmentMethod"]:
        graph.add((LCA[child_class], RDFS.subClassOf, LCA.Resource))
    for predicate, label in [
        (LCA.usesDatabase, "uses database"),
        (LCA.usesMethod, "uses impact assessment method"),
        (LCA.coversCategory, "covers impact category"),
        (LCA.hasStudyPhase, "has study phase"),
        (LCA.mentions, "mentions"),
        (LCA.relatedTo, "related to"),
    ]:
        graph.add((predicate, RDF.type, RDF.Property))
        graph.add((predicate, RDFS.label, Literal(label)))
    return graph


def add_extraction(graph: Graph, extraction: KnowledgeExtraction) -> Graph:
    entity_uris = {
        entity.name.casefold(): LCA[f"{slug(entity.entity_type)}/{slug(entity.name)}"]
        for entity in extraction.entities
    }
    for entity in extraction.entities:
        entity_uri = LCA[f"{slug(entity.entity_type)}/{slug(entity.name)}"]
        graph.add((entity_uri, RDF.type, LCA[entity.entity_type]))
        graph.add((entity_uri, RDFS.label, Literal(entity.name)))
        if entity.description:
            graph.add((entity_uri, RDFS.comment, Literal(entity.description)))
        if entity.source:
            graph.add((entity_uri, DCTERMS.source, Literal(entity.source)))
        graph.add((entity_uri, LCA.confidence, Literal(entity.confidence, datatype=XSD.float)))

    for relation in extraction.relations:
        subject = entity_uris.get(relation.subject.casefold(), LCA[f"entity/{slug(relation.subject)}"])
        obj = entity_uris.get(relation.object.casefold(), LCA[f"entity/{slug(relation.object)}"])
        graph.add((subject, RDFS.label, Literal(relation.subject)))
        graph.add((obj, RDFS.label, Literal(relation.object)))
        graph.add((subject, PREDICATE_MAP[relation.predicate], obj))
        if relation.source:
            graph.add((subject, DCTERMS.source, Literal(relation.source)))
    return graph


def add_resources(graph: Graph, resources: list[Resource]) -> Graph:
    for resource in resources:
        uri = LCA[f"resource/{slug(resource.id)}"]
        graph.add((uri, RDF.type, LCA.Resource))
        graph.add((uri, RDF.type, LCA[resource.resource_type]))
        graph.add((uri, RDFS.label, Literal(resource.name)))
        graph.add((uri, RDFS.comment, Literal(resource.description)))
        graph.add((uri, LCA.resourceId, Literal(resource.id)))
        graph.add((uri, LCA.sourceNote, Literal(resource.source_note)))
        for region in resource.regions:
            graph.add((uri, LCA.availableInRegion, Literal(region)))
        for phase in resource.study_phases:
            graph.add((uri, LCA.supportsStudyPhase, Literal(phase)))
        for category in resource.impact_categories:
            graph.add((uri, LCA.coversImpactCategory, Literal(category)))
        for tag in resource.tags:
            graph.add((uri, LCA.hasTag, Literal(tag)))
    return graph


def ontology_summary(graph: Graph) -> dict[str, int]:
    return {
        "triples": len(graph),
        "resources": len(list(graph.subjects(RDF.type, LCA.Resource))),
        "entities": len(set(graph.subjects(RDF.type, None))),
    }
