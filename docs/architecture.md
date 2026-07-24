# Repository Structure

This is the **Knowledge Graph Generator** — a toolkit for building structured knowledge graphs from raw text, designed for LLM training and Graph RAG workflows. Built for the Vietnam AI Challenge.

---

## Directory Map

```
hackathon/
│
├── pyproject.toml                     # Project config, dependencies, CLI entry point
├── README.md                          # Overview & quick start
├── .gitignore                         # Files excluded from version control
│
├── src/kg_generator/                  # ← All source code lives here
│   ├── __init__.py                    # Package metadata
│   ├── cli.py                         # Command-line interface (kg-gen)
│   ├── config.py                      # Configuration, ontology, language backends
│   ├── pipeline.py                    # Orchestrator — ties all stages together
│   │
│   ├── ingest/                        # Stage 1: Data loading & cleaning
│   │   ├── loader.py                  #   Loads TXT, JSON, CSV, JSONL from files/dirs
│   │   └── cleaner.py                 #   Text normalization (whitespace, encoding, quotes)
│   │
│   ├── dedup/                         # Deduplication & quality filtering
│   │   ├── near_dedup.py              #   MinHash, SimHash, n-gram duplicate detection
│   │   └── quality.py                 #   Heuristic quality scoring & low-quality filtering
│   │
│   ├── extract/                       # Stage 2: Entity & relation extraction
│   │   ├── entities.py                #   NER (spaCy for EN, underthesea stub for VI)
│   │   └── relations.py              #   Rule-based + LLM-powered relation extraction
│   │
│   ├── resolve/                       # Stage 3: Entity resolution
│   │   └── resolver.py               #   Embedding-based or string-similarity dedup
│   │
│   ├── graph/                         # Stage 4: Graph construction
│   │   ├── builder.py                 #   Builds NetworkX DiGraph from triples + ontology validation
│   │   └── enrich.py                  #   Stub for Wikidata/DBpedia linking
│   │
│   ├── evaluate/                      # Stage 5: Quality evaluation
│   │   └── metrics.py                 #   7 metrics: completeness, consistency, duplication,
│   │                                  #     missing info, format errors, labeling quality, reusability
│   │
│   └── export/                        # Export to multiple formats
│       └── exporter.py                #   JSON, GraphML, Neo4j CSV, RDF/Turtle, Cytoscape.js
│
├── configs/                           # YAML configuration files
│   ├── default_ontology.yaml          #   Entity types, relationship types, attributes
│   ├── *_manifest.yaml                #   Source manifests for curation
│   ├── pipelines/                     #   Pipeline presets
│   │   ├── debug.yaml                 #     Fast debug run on small sample
│   │   ├── default.yaml               #     English Wikipedia preset
│   │   ├── vietnamese.yaml            #     Vietnamese best quality
│   │   ├── vietnamese_fast.yaml       #     Vietnamese dependency-light
│   │   └── vietnamese_global_quality.yaml  # Vietnamese Neo4j global graph
│   └── evaluation/                    #   Evaluation configs
│       ├── overnight_vi.yaml          #     Fairness ablation preset
│       └── vietnamese_global_eval.yaml #    GraphGen QA eval preset
│
├── tests/                             # Pytest test suite
│   ├── test_ingest.py                 #   Loader & cleaner tests
│   ├── test_extract.py                #   Entity extraction tests
│   └── test_graph.py                  #   Graph building, dedup, and quality filter tests
│
├── docs/                              # User-facing documentation
│   ├── usage.md                       #   CLI commands & pipeline details
│   └── UV_SETUP.md                    #   Beginner-friendly uv setup guide
│
├── data/sample/                       # Sample input data for testing
│   ├── marie_curie.txt                #   Biography → entities: people, places, events
│   └── alan_turing.txt               #   Biography → entities: people, places, events
│
├── help_files/                        # Reference materials from the hackathon brief
│   ├── Problem_description.md         #   Original problem statement
│   ├── how_to_create_a_knowledge_graph.md  #   KG construction methodology
│   ├── data_management_survey_summary.md   #   LLM data management best practices
│   └── REPO_STRUCTURE.md              #   ← This file
│
└── output/                            # Generated output (gitignored)
    ├── knowledge_graph.json           #   Full KG: graph, entities, triples
    ├── knowledge_graph.graphml        #   GraphML for Gephi/Cytoscape
    └── metrics.json                   #   Quality evaluation scores
```

---

## Pipeline Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Ingest  │───▶│  Dedup   │───▶│ Extract  │───▶│ Resolve  │───▶│  Graph   │
│ + Clean  │    │+ Quality │    │Entities+ │    │ Entities │    │  Build   │
│          │    │  Filter  │    │Relations │    │          │    │+Enrich   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                                     │
                                                              ┌──────▼──────┐
                                                              │  Evaluate   │
                                                              │  + Export   │
                                                              └─────────────┘
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **Language abstraction** | English now (`spaCy`), Vietnamese later (`underthesea`) — swap a backend, not the whole pipeline |
| **MinHash + fallback chain** | MinHash for scale, SimHash/n-gram as lighter alternatives, exact hash as last resort |
| **NetworkX for prototyping** | Zero-config, no server needed. Neo4j available via `GraphBackend.NEO4J` for production |
| **Core vs optional deps** | `sentence-transformers`, `underthesea`, `neo4j`, and `openai` are all optional extras |
| **Export flexibility** | JSON (debugging), GraphML (Gephi), Neo4j CSV (production), RDF (semantic web), Cytoscape.js (visualization) |

---

## Entry Points

| Command | What it does |
|---|---|
| `kg-gen quick -i <path>` | Full pipeline with sensible defaults |
| `kg-gen run -c config.yaml` | Full pipeline with a config file |
| `kg-gen evaluate -i output/kg.json` | Quality evaluation only |
| `python -m kg_generator.cli` | Same as `kg-gen` (if CLI not installed) |
| `python -m pytest tests/ -v` | Run test suite |

---

## Adding New Features

1. **New file format support** → add loader in `ingest/loader.py`
2. **New language** → add backend class in `extract/entities.py` and `ingest/cleaner.py`, then wire it into `config.py`
3. **New export format** → add method in `export/exporter.py`, add format string to `config.py`
4. **New quality metric** → add method in `evaluate/metrics.py`, include it in `evaluate_graph()`
5. **External KB linking** → implement `graph/enrich.py` stubs
