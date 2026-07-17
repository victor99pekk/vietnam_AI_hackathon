
# Toolkit for standardizing & building datasets for Vietnamese LLM development


## The Problem
Vietnam is pushing to develop large language models at a national scale. Building high-quality AI datasets is an urgent need to support this effort. However, Vietnam currently lacks standard tools and processes for building, managing, and evaluating dataset quality for language model development. This gap makes it difficult to ensure consistency, reusability, and scalability of datasets.

## What We Need to Build
Develop a toolkit and a clear process to standardize, build, and preliminarily evaluate dataset quality for Vietnamese language model development. The solution should include five components. First, tools to collect, pre-process, and standardize data from multiple sources. Second, tools to detect duplicate and erroneous data, and to normalize and clean formats so the data is ready for model training. Third, tools to support labeling, manage metadata, and handle dataset versioning. Fourth, tools to perform preliminary quality evaluation against metrics such as completeness, consistency, duplication level, missing information, format errors, labeling quality, and reusability. Fifth, a set of standard operating procedures, guides, and templates that teams can follow when building AI datasets for training and evaluating Vietnamese language models.

## What We Need to Deliver
A working prototype toolkit that performs at least one or more functions such as standardization, duplicate detection, labeling support, metadata management, or preliminary quality evaluation. A technical solution description that covers overall architecture, the processing pipeline, and technologies used. A demo video or live demonstration. And basic usage documentation along with notes on how the toolkit can be extended over time. The data can come from any combination of legal sources. This includes government open data, public-domain data or data released under appropriate licenses such as Creative Commons or Open License, open-source AI research datasets, self-built or self-compiled data with clear ownership rights, and open-source models, libraries, and tools.

## 5 Components to Build

### 1. Data Collection & Standardization
Tools to collect, pre-process, and standardize data from multiple sources.
- How to collect/scrape? **ASK**
- Support multiple data formats

### 2. Deduplication & Cleaning
Detect duplicate/erroneous data, normalize and clean formats for model training readiness.
- ✅ Done — needs evaluation on bad data samples to verify effectiveness

### 3. Labeling & Metadata Management
Support labeling, manage metadata, and handle dataset versioning.
- 🔍 Look into — relates to documents saved in the KG
- Versioning: replace old versions (same source) with new?

### 4. Quality Evaluation
Preliminary quality metrics: completeness, consistency, duplication level, missing info, format errors, labeling quality, reusability.
- 🔍 Look into missing info detection (hallucinated facts, etc.)

### 5. Standard Operating Procedures
Guides and templates for teams building AI datasets for Vietnamese LLMs.
- Ask LLM to generate guidelines
- Fix pipeline
- Define input data specification

---

## 5 Deliverables

| # | Deliverable | Effort |
|---|-------------|--------|
| 1 | **Working prototype** — performs at least one function (standardization, dedup, labeling, metadata, or quality eval) | 🟢 Easy — ask LLM |
| 2 | **Technical solution description** — architecture, processing pipeline, technologies used | 🟢 Easy — ask LLM |
| 3 | **Demo video or live demonstration** | 🟡 Medium |
| 4 | **Basic usage documentation** | 🟢 Easy — ask LLM |
| 5 | **Extension notes** — how the toolkit can be scaled over time | 🟢 Easy — ask LLM