# Maximizing VAIC 2026 Hackathon Score — Recommendations

**TL;DR**: Your KG Generator toolkit is strong on technical depth and AI-native design but has gaps in UX, safety/grounding, business viability, and demo readiness. Below is a prioritized list of enhancements mapped to each scoring pillar, ordered by impact-to-effort ratio.

---

## What You Have (Strengths)

- ✅ Fully modular 5-stage KG pipeline (ingest → dedup → extract → resolve → build → evaluate)
- ✅ Dataset curation with immutable versioning, MinHash/semantic dedup, audit trail
- ✅ 7 quality metrics (completeness, consistency, duplication, missing info, format errors, labeling, reusability)
- ✅ Structural audit (orphan detection, density, schema compliance, entity dup, multi-hop connectivity)
- ✅ SFT data generation from KG + quality evaluation (deepeval + heuristic fallback)
- ✅ GraphGen-style subgraph organization + multi-hop QA generation
- ✅ Fact coverage evaluation against source documents
- ✅ Auto-generated evaluation plots
- ✅ Vietnamese language support (underthesea)
- ✅ Multiple export formats (JSON, GraphML, Neo4j CSV, RDF, Cytoscape.js)
- ✅ Wikipedia dataset downloader
- ✅ Good documentation (usage, curation, dedup experiments, UV setup)

### What's Missing or Could Be Improved

---

## Scoring Pillar 1: Technical Implementation & Engineering Depth (20 pts)

**Current Estimate**: ~14-15/20 — Strong foundation, but missing deployment/polish

### High-Impact Actions

1. **Add a Web UI / Interactive Demo (CRITICAL)**
   - Build a Streamlit or Gradio app that lets judges upload a file → run pipeline → visualize KG
   - Includes: drag-and-drop upload, progress bar, interactive graph visualization, metric dashboard
   - This is the #1 thing for the "Live Administrator URL" deliverable
   - *Effort: Medium | Impact: Very High*

2. **Docker Containerization**
   - Create `Dockerfile` + `docker-compose.yml`
   - One-command setup: `docker compose up`
   - Include optional Neo4j service in compose
   - *Effort: Low | Impact: Medium*

3. **CI/CD Pipeline (GitHub Actions)**
   - Run pytest suite on push/PR
   - Lint check (ruff/mypy)
   - Build and push Docker image
   - *Effort: Low | Impact: Medium*

4. **Cache Layer for Repeated Operations**
   - Cache spaCy NER results, embeddings, MinHash signatures
   - Use disk-based cache (diskcache or joblib)
   - Shows "scalability thinking"
   - *Effort: Medium | Impact: Medium*

5. **Add a Real API (FastAPI)**
   - REST endpoints: `POST /pipeline/run`, `GET /pipeline/status/{id}`, `GET /results/{id}`
   - Async background processing with Celery or ARQ
   - OpenAPI/Swagger docs auto-generated
   - *Effort: Medium | Impact: High*

6. **Performance Benchmarks**
   - Add `--benchmark` flag to CLI
   - Report: docs/sec processing, memory usage, bottleneck stage
   - Compare spaCy vs LLM extraction speed/quality
   - *Effort: Low | Impact: Medium*

---

## Scoring Pillar 2: AI-Native Architecture & Innovation (20 pts)

**Current Estimate**: ~15-16/20 — Good LLM integration, but could demonstrate more advanced AI patterns

### High-Impact Actions

7. **Graph RAG Demo Integration**
   - Build a minimal Graph RAG demo: query → retrieve from KG → LLM answers with citations
   - Show KG-augmented retrieval vs naive retrieval side-by-side
   - Use LangChain/LlamaIndex + your KG as the knowledge base
   - This is a *structural differentiator* — proves KG value for LLM training
   - *Effort: Medium | Impact: Very High*

8. **Agentic Workflow for Dataset Curation**
   - Auto-detect data quality issues → suggest fixes → apply with user approval
   - Example: "Detected 15% duplicate rate in vi_wikipedia. Apply dedup? [y/N]"
   - Show "AI-Native Oath" transparency: log every AI decision
   - *Effort: Medium | Impact: High*

9. **Vietnamese-Specific Enhancements**
   - Add Vietnamese entity types relevant to Vietnam (government bodies, provinces, etc.)
   - Vietnamese relation types (works_at Vietnamese organizations, etc.)
   - Demo with Vietnamese Wikipedia data
   - *Effort: Low-Medium | Impact: High (directly addresses problem statement)*

10. **Iterative Self-Improvement / Active Learning Loop**
    - Track which extracted triples the user corrects → feed back to improve extraction
    - Simple version: flag low-confidence extractions for human review
    - *Effort: Medium | Impact: Medium*

11. **Multi-Modal Data Support (Stretch)**
    - Extract text from PDFs, images (OCR via Tesseract)
    - Process tabular data → entity extraction
    - *Effort: High | Impact: Medium*

---

## Scoring Pillar 3: Business Viability & Pilot Pathway (20 pts)

**Current Estimate**: ~8-10/20 — Weakest area; needs significant attention

### High-Impact Actions

12. **Document a Specific Vietnamese Pilot Use Case**
    - Pick ONE concrete partner scenario: e.g., "Vietnam National University — building a KG from 10,000 Vietnamese academic papers"
    - Write a 1-page pilot proposal: timeline, cost, success metrics, team needed
    - Show projected impact (e.g., "50% reduction in dataset preparation time")
    - *Effort: Low | Impact: Very High*

13. **Cost Estimation & Pricing Model**
    - Document compute costs for different scales (100 docs, 1K docs, 100K docs, 1M docs)
    - Compare: spaCy-only ($0) vs LLM extraction ($X per 1K docs via DeepSeek API)
    - Show TCO savings vs manual curation
    - *Effort: Low | Impact: High*

14. **Create a "Path to Production" Roadmap Slide**
    - Phase 1: Open-source toolkit (current)
    - Phase 2: Managed cloud service (SaaS)
    - Phase 3: Enterprise on-prem deployment
    - Include revenue model, target customers, go-to-market strategy
    - *Effort: Low | Impact: Medium*

15. **Market Analysis Addendum**
    - Map competitors: Diffbot, Neo4j ETL tools, Unstructured.io, LangChain/LlamaIndex
    - Your unfair advantage: Vietnamese-first, KG-native for LLM training, open source
    - *Effort: Low | Impact: Medium*

16. **Add a `kg-gen serve` Command for One-Click Demo**
    - `kg-gen serve` starts a local web server with the Streamlit UI
    - Pre-loaded with Vietnamese sample data
    - This is the "Live Administrator URL" deliverable
    - *Effort: Medium | Impact: High*

---

## Scoring Pillar 4: AI-Native UX & Design Thinking (15 pts)

**Current Estimate**: ~6-8/15 — CLI-only; needs UI and explainability

### High-Impact Actions

17. **Interactive KG Explorer (Web UI)**
    - Streamlit app with: file upload → config selection → pipeline run → interactive graph viz → metric dashboard
    - Use pyvis or vis-network for interactive graph (zoom, pan, click nodes for details)
    - Show entity descriptions, triples, provenance on click
    - *Effort: Medium | Impact: Very High*

18. **Explainability Layer**
    - For every extracted triple: show source text snippet, confidence, extraction method
    - For entity resolution: show why two entities were merged (similarity score + evidence)
    - For dedup: show side-by-side comparison of near-duplicate documents
    - *Effort: Medium | Impact: High*

19. **Decision Support Dashboard**
    - Quality report with red/yellow/green indicators
    - Actionable recommendations: "5 entities lack types — click to annotate"
    - Export filtered views for different stakeholders
    - *Effort: Medium | Impact: Medium*

20. **Progressive Onboarding**
    - `kg-gen quick` already exists — add interactive tutorial mode
    - `kg-gen tutorial` — walks user through each stage with explanations
    - *Effort: Low | Impact: Medium*

---

## Scoring Pillar 5: AI Safety, Grounding & Trust (15 pts)

**Current Estimate**: ~7-9/15 — Has audit trail, but safety is largely unaddressed

### High-Impact Actions

21. **Triple-Level Provenance Citations (CRITICAL)**
    - Every extracted triple should link back to source `(document_id, chunk_id, character_span)`
    - Export provenance trail in output JSON
    - This is critical for "verifiability" — judges will look for this
    - *Effort: Medium | Impact: Very High*

22. **Confidence Scoring for Extracted Triples**
    - spaCy NER: use entity label confidence
    - LLM extraction: ask LLM to self-rate confidence (1-5)
    - Flag low-confidence triples in output
    - *Effort: Medium | Impact: High*

23. **PII Detection & Data Privacy Scanner**
    - Add optional PII scan before processing (email, phone, ID numbers, names)
    - Flag documents with potential PII → user decides
    - Critical for "Data Security & Privacy" score
    - *Effort: Medium | Impact: High*

24. **Bias Detection in Extracted Entities**
    - Check gender/ethnicity distribution in PERSON entities
    - Flag underrepresented groups
    - Simple statistical check, not complex
    - *Effort: Low | Impact: Medium*

25. **Hallucination Detection for LLM Extraction**
    - Cross-check LLM-extracted entities against spaCy NER
    - Flag entities found by LLM but not by spaCy (or vice versa)
    - Report hallucination rate per chunk
    - *Effort: Medium | Impact: High*

26. **Model Card / Datasheet Generation**
    - Auto-generate a model card for each curated dataset
    - Include: intended use, limitations, data sources, preprocessing steps, bias considerations
    - Follow HuggingFace model card conventions
    - *Effort: Low | Impact: Medium*

27. **Transparency of Limitations Documentation**
    - Document what the system CANNOT do (PII redaction, toxicity filtering, mixed-language detection — already noted)
    - Add a "Known Limitations" section prominently in docs and UI
    - *Effort: Low | Impact: Medium*

---

## Scoring Pillar 6: Presentation, Demo & Defensibility (10 pts)

**Current Estimate**: ~5-6/10 — Good docs, but no slide deck or demo video yet

### High-Impact Actions

28. **Create a Polished 5-Minute Demo Video**
    - Script: Problem → Architecture → Live Demo (upload Vietnamese data → KG → metrics → Graph RAG query)
    - Show before/after: raw text → clean KG → LLM answers improved
    - Keep it under 5 minutes as required
    - *Effort: Medium | Impact: Very High*

29. **Prepare Slide Deck (~10-12 slides)**
    - Slide 1: Problem statement (Vietnam's LLM dataset gap)
    - Slide 2: Our solution overview
    - Slide 3: Architecture diagram (Mermaid or diagram)
    - Slide 4-5: Technical depth (pipeline stages, AI-native design)
    - Slide 6: Demo screenshot walkthrough
    - Slide 7: Evaluation results (metrics, ablation study)
    - Slide 8: Business model & pilot pathway
    - Slide 9: AI safety & trust features
    - Slide 10: Competitive advantage
    - Slide 11: Team & roadmap
    - Slide 12: Thank you / Q&A
    - *Effort: Medium | Impact: Very High*

30. **AI Collaboration Log Template**
    - Document all AI tool usage (which model, what prompt, what output)
    - Categorize: code generation, debugging, documentation, evaluation, test generation
    - This is REQUIRED by the rules — missing = disqualification risk
    - *Effort: Low | Impact: Critical*

31. **Prepare Anticipated Q&A**
    - "Why KG instead of just fine-tuning on raw text?" → Show ablation results
    - "How does this handle Vietnamese specifically?" → underthesea + vi demo
    - "What about data privacy?" → PII scanner, local-only mode, audit trail
    - "How scalable is this?" → Benchmarks, chunking, caching
    - "What makes this AI-native vs traditional ETL?" → LLM extraction, agentic curation, Graph RAG
    - *Effort: Low | Impact: Medium*

32. **Competitive Comparison Table**
    - Compare with: manual curation, Diffbot, Unstructured.io, LangChain document loaders
    - Your advantages: Vietnamese support, KG-native, evaluation built-in, open source, auditable
    - *Effort: Low | Impact: Medium*

---

## Problem Description Deliverables Checklist

| # | Deliverable | Status | Action Needed |
|---|-------------|--------|---------------|
| 1 | Working prototype | ✅ Done | Polish edge cases |
| 2 | Technical solution description | ⚠️ Partial | Write formal architecture document (use LLM to generate from codebase) |
| 3 | Demo video or live demo | ❌ Missing | Create video + Streamlit app |
| 4 | Basic usage documentation | ✅ Good | Add Vietnamese-specific guide |
| 5 | Extension notes | ❌ Missing | Write roadmap for scaling (more languages, more formats, cloud deployment) |

---

## Problem Description 5 Components Checklist

| # | Component | Status | Action Needed |
|---|-----------|--------|---------------|
| 1 | Data Collection & Standardization | ✅ Good | Add PDF/HTML support |
| 2 | Deduplication & Cleaning | ✅ Good | Evaluate on "bad data" samples |
| 3 | Labeling & Metadata Management | ⚠️ Basic | Add versioning UI, label editor, metadata schema editor |
| 4 | Quality Evaluation | ✅ Strong | Add PII/bias/hallucination detection |
| 5 | SOPs, Guides, Templates | ⚠️ Partial | Generate formal SOP document, create template files |

---

## Prioritized Implementation Order

### Phase 1: Demo-Ready (Highest Priority — Do These First)

These directly enable deliverables #3 and #4 (demo video + live URL):

1. **Streamlit Web UI** (#17) — interactive KG explorer + pipeline runner
2. **Triple-Level Provenance Citations** (#21) — every triple links to source
3. **AI Collaboration Log** (#30) — REQUIRED, cannot skip
4. **Demo Video Script & Recording** (#28)
5. **Slide Deck** (#29)
6. **`kg-gen serve` Command** (#16)

### Phase 2: Safety & Trust (High Priority)

These address the weakest scoring pillar:

7. **PII Detection Scanner** (#23)
8. **Confidence Scoring for Triples** (#22)
9. **Hallucination Detection** (#25)
10. **Model Card Generation** (#26)
11. **Transparency Docs Update** (#27)

### Phase 3: Business & Polish (Medium Priority)

These round out the presentation:

12. **Vietnamese Pilot Use Case** (#12)
13. **Cost Estimation** (#13)
14. **Graph RAG Demo** (#7)
15. **Competitive Comparison** (#32)
16. **Docker + CI/CD** (#2, #3)

### Phase 4: Stretch Goals (If Time Permits)

17. **Agentic Curation Workflow** (#8)
18. **Performance Benchmarks** (#6)
19. **FastAPI Backend** (#5)
20. **Multi-Modal Support** (#11)

---

## Relevant Files to Modify/Create

### New Files to Create

- `src/kg_generator/ui/` — Streamlit app (new directory)
- `Dockerfile` + `docker-compose.yml`
- `.github/workflows/ci.yml`
- `docs/architecture.md` — Formal technical architecture document
- `docs/pilot_proposal.md` — Vietnamese pilot use case
- `docs/ai_collaboration_log.md` — AI usage log template
- `presentation/slides.md` — Slide deck outline
- `presentation/demo_script.md` — Demo video script

### Files to Modify

- `src/kg_generator/evaluate/metrics.py` — Add confidence scoring, hallucination detection
- `src/kg_generator/extract/entities.py` — Add confidence scores to Entity dataclass
- `src/kg_generator/extract/relations.py` — Add provenance tracking to triples
- `src/kg_generator/extract/graphgen.py` — Add confidence self-rating to LLM prompt
- `src/kg_generator/export/exporter.py` — Include provenance in all export formats
- `src/kg_generator/cli.py` — Add `serve`, `tutorial`, `benchmark` commands
- `src/kg_generator/dedup/quality.py` — Add PII detection, bias stats
- `configs/default_ontology.yaml` — Add Vietnamese-specific entity/relation types
- `README.md` — Add architecture diagram, competitive comparison, known limitations

### Reference Files (Don't Modify, Use as Templates)

- `evaluation/data_eval/structural_audit.py` — Pattern for safety audit reports
- `evaluation/data_eval/coverage.py` — Pattern for fact extraction/verification
- `src/kg_generator/curate/` — Pattern for auditable workflows
- `help_files/hackathon_scoring.md` — Scoring criteria reference
- `help_files/Problem_description.md` — Requirements reference

---

## Verification Checklist

After implementation, verify:

1. `docker compose up` works end-to-end on a fresh machine
2. Streamlit UI accepts file upload → runs pipeline → shows KG visualization → exports
3. Every triple in `knowledge_graph.json` has `source_document_id`, `source_chunk_id`, `source_span`
4. `kg-gen serve` starts the web UI
5. Demo video < 5 min, covers problem → solution → live demo
6. Slide deck has all 12 slides
7. AI Collaboration Log has entries for all phases
8. PII scanner flags test document with fake email/phone
9. Confidence scores appear in extraction output
10. Hallucination report generated for LLM extraction mode
11. Vietnamese sample data runs through pipeline without errors
12. All pytest tests pass
13. Coverage report shows >80% fact coverage for sample data
14. Known Limitations section visible in README and UI
