(() => {
  const $ = (selector) => document.querySelector(selector);
  const state = { global: null, lab: null, cyGlobal: null, cyLab: null, started: 0, options: null };
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const getJSON = async (url) => { const response = await fetch(url); if (!response.ok) throw new Error(`${response.status} ${response.statusText}`); return response.json(); };

  function normalize(raw) {
    const graph = raw?.graph || raw || {};
    let nodes = graph.nodes || graph.elements?.nodes || [];
    let edges = graph.edges || graph.links || graph.elements?.edges || [];
    if (!Array.isArray(nodes) && typeof nodes === 'object') nodes = Object.entries(nodes).map(([id, value]) => ({ id, ...value }));
    if (!Array.isArray(edges)) edges = [];
    let chunkCount = 0;
    const cleanNodes = nodes.map((node, index) => {
      const data = node.data || node;
      const type = String(data.type || data.entity_type || 'CONCEPT').toUpperCase();
      const id = String(data.id ?? data.node_id ?? `node-${index}`);
      if (type === 'CHUNK') chunkCount += 1;
      const chunkNumber = data.chunk_index ?? data.index ?? chunkCount;
      const label = type === 'CHUNK'
        ? (data.chunk_label || `Chunk ${String(chunkNumber).padStart(2, '0')}`)
        : (data.label || data.name || data.title || `Node ${index + 1}`);
      return { ...data, id, label, type, evidence: data.evidence || data.sentence || data.context || '', contentPreview: type === 'CHUNK' ? String(data.text || data.content || data.name || '').slice(0, 240) : '' };
    });
    const cleanEdges = [];
    edges.forEach((edge, index) => {
      const data = edge.data || edge;
      const source = data.source ?? data.from ?? data.subject;
      const target = data.target ?? data.to ?? data.object;
      const predicates = Array.isArray(data.predicates) ? data.predicates : (Array.isArray(data.predicate) ? data.predicate : [data.label || data.relation || data.predicate || 'related_to']);
      if (source && target) predicates.forEach((predicate, predicateIndex) => {
        const relation = Array.isArray(data.relations) ? (data.relations[predicateIndex] || data.relations[0] || {}) : {};
        cleanEdges.push({ ...data, ...relation, id: String(data.id ?? `edge-${index}-${predicateIndex}`), source: String(source), target: String(target), label: String(predicate), evidence: relation.evidence_sentence || data.evidence_sentence || data.evidence || data.description || '', sourceChunkId: relation.source_chunk_id || data.source_chunk_id || '' });
      });
    });
    const stats = raw?.stats || {};
    return { ...raw, graph: { nodes: cleanNodes, edges: cleanEdges }, stats: { ...stats, entities: stats.entities ?? stats.num_nodes ?? cleanNodes.length, relations: stats.relations ?? stats.num_edges ?? cleanEdges.length, triples: stats.triples ?? stats.num_triples ?? raw?.triples?.length ?? cleanEdges.length }, metrics: raw?.metrics || {} };
  }

  function graphStyle() { return [
    { selector: 'node', style: { 'background-color': (element) => ({ PERSON: '#ff8c70', PLACE: '#55d4c4', GPE: '#55d4c4', ORGANISATION: '#9f8cff', ORG: '#9f8cff', CONCEPT: '#f2cc68', CHUNK: '#4b8fe2', DOCUMENT: '#b8c7ce' }[element.data('type')] || '#55d4c4'), label: 'data(label)', color: '#eaf2f3', 'font-size': 10, 'text-valign': 'bottom', 'text-margin-y': 8, 'text-max-width': 100, 'text-wrap': 'ellipsis', 'border-color': '#0b1928', 'border-width': 2, width: 17, height: 17 } },
    { selector: 'node[type = "CHUNK"]', style: { shape: 'round-rectangle', width: 24, height: 12, 'font-size': 8, 'border-color': '#83b7ff', 'border-width': 1 } },
    { selector: 'node[type = "DOCUMENT"]', style: { shape: 'diamond', width: 23, height: 23, color: '#b8c7ce' } },
    { selector: 'edge', style: { width: 1.8, 'line-color': '#40606a', 'target-arrow-color': '#40606a', 'target-arrow-shape': 'triangle', label: 'data(label)', color: '#7d999f', 'font-size': 8, 'text-background-color': '#0b1928', 'text-background-opacity': .72, 'text-background-padding': 2, 'curve-style': 'bezier', 'overlay-opacity': 0, 'overlay-padding': 10 } },
    { selector: 'node:selected', style: { 'border-color': '#fff', 'border-width': 3 } },
    { selector: 'edge:selected', style: { width: 3, 'line-color': '#ff8c70', 'target-arrow-color': '#ff8c70', color: '#f7c4b7', 'z-index': 999, 'overlay-opacity': .08, 'overlay-color': '#ff8c70', 'overlay-padding': 7 } },
  ]; }

  function mount(which, result) {
    const container = $(`#${which}Graph`); container.innerHTML = '';
    if (!result.graph.nodes.length || typeof cytoscape === 'undefined') { container.innerHTML = '<div class="empty"><b>No entities found</b><small>Try a longer document.</small></div>'; return; }
    const key = which === 'global' ? 'cyGlobal' : 'cyLab'; state[key]?.destroy();
    const nodeCount = result.graph.nodes.length;
    const largeGraph = nodeCount > 500;
    state[key] = cytoscape({ container, elements: { nodes: result.graph.nodes.map((node) => ({ data: node })), edges: result.graph.edges.map((edge) => ({ data: edge })) },
      layout: largeGraph
        ? { name: 'cose', animate: false, padding: 30, nodeRepulsion: () => 8000, idealEdgeLength: () => 60, numIter: 800, coolingFactor: 0.92 }
        : { name: 'cose', animate: false, padding: 30 },
      style: graphStyle() });
    state[key].on('tap', 'node', (event) => showDetail(which, event.target.data()));
    state[key].on('tap', 'edge', (event) => showRelationshipDetail(which, event.target.data()));
  }

  function showDetail(which, node) {
    const result = state[which]; if (!result) return;
    const links = result.graph.edges.filter((edge) => edge.source === node.id || edge.target === node.id);
    const description = node.type === 'CHUNK' ? (node.contentPreview || 'Chunk text is hidden from the graph label to keep the visualization readable.') : (node.description || `${links.length} linked fact${links.length === 1 ? '' : 's'} in this graph.`);
    const descriptionHtml = which === 'global' ? '' : `<p>${esc(description)}</p>`;
    $(`#${which}Detail`).innerHTML = `<div class="detail-content"><div><span class="detail-type">${esc(node.type)}</span><h3>${esc(node.label)}</h3>${descriptionHtml}</div><div class="evidence"><span class="evidence-label">${node.type === 'CHUNK' ? 'CHUNK PREVIEW' : 'EVIDENCE / SOURCE SENTENCE'}</span><blockquote>“${esc(node.evidence || node.contentPreview || 'Evidence sentence not provided by the extractor.')}”</blockquote></div></div>`;
  }

  function showRelationshipDetail(which, edge) {
    const result = state[which]; if (!result) return;
    const source = result.graph.nodes.find((node) => node.id === edge.source);
    const target = result.graph.nodes.find((node) => node.id === edge.target);
    const sourceLabel = source?.label || edge.source;
    const targetLabel = target?.label || edge.target;
    $(`#${which}Detail`).innerHTML = `<div class="detail-content relationship-detail"><div><span class="detail-type">RELATIONSHIP</span><h3>${esc(edge.label)}</h3><p><strong>${esc(sourceLabel)}</strong><span class="relation-arrow">→</span><strong>${esc(targetLabel)}</strong></p></div><div class="evidence"><span class="evidence-label">RELATIONSHIP EVIDENCE</span><blockquote>“${esc(edge.evidence || 'Evidence sentence not provided for this relationship.') }”</blockquote>${edge.sourceChunkId ? `<small>Source: ${esc(edge.sourceChunkId)}</small>` : ''}</div></div>`;
  }

  function search(which, value) { const cy = which === 'global' ? state.cyGlobal : state.cyLab; if (!cy) return; const query = value.toLowerCase(); cy.nodes().forEach((node) => node.style('opacity', !query || String(node.data('label')).toLowerCase().includes(query) ? 1 : 0.14)); }

  async function loadGlobal(limit = 300) {
    try {
      const result = normalize(await getJSON(`/api/graphs/global?limit=${limit}`)); state.global = result;
      $('#globalBadge').textContent = result.metadata?.source === 'sample' ? 'DEMO DATA' : 'LIVE NEO4J';
      $('#gNodes').textContent = result.stats.entities; $('#gEdges').textContent = result.stats.relations; $('#gTriples').textContent = result.stats.triples;
      $('#globalFit').textContent = limit >= 5000 ? '⛶ Fit' : '⛶ Fit';
      mount('global', result);
    } catch (error) { $('#globalBadge').textContent = 'UNAVAILABLE'; $('#globalGraph').innerHTML = '<div class="empty"><b>Global graph unavailable</b><small>Check the API connection.</small></div>'; }
  }

  function applyAvailability(options) {
    const availability = options?.availability || {};
    const method = $('#method'); const graphgen = method.querySelector('option[value="graphgen"]'); if (graphgen) { graphgen.disabled = availability.graphgen === false; graphgen.textContent = availability.graphgen === false ? 'LLM-guided — Recommended · API key needed' : 'LLM-guided — Recommended · best relationships'; if (graphgen.disabled && method.value === 'graphgen') method.value = 'offline'; }
    ['chunkMethod', 'documentDedupMethod', 'dedupMethod', 'resolveMethod'].forEach((id) => { const select = $(`#${id}`); if (!select) return; const advanced = [...select.options].filter((option) => ['semantic', 'layered', 'embedding'].includes(option.value)); advanced.forEach((option) => { option.disabled = availability.embeddings === false; if (availability.embeddings === false && option.value === select.value) select.value = select.options[0].value; }); });
    updateGuidance();
  }

  function updateGuidance() {
    const guidance = {
      extraction: {
        offline: 'Experimental beta. This rules-based baseline is fast, but its relation logic can connect too many node pairs. It is included for comparison, not as our recommended pipeline.',
        graphgen: 'Our recommended extraction strategy. DeepSeek Flash jointly identifies meaningful entities and grounded relationships instead of linking nodes with broad rules.',
      },
      chunk: {
        sentence: 'Recommended default: 450 target tokens with 60-token overlap. Preserves sentence boundaries at good demo speed.',
        fixed: 'Fast and predictable, but can split sentences. Recommended starting values: 500 characters with 100 overlap.',
        none: 'Fastest for short input. Avoid it for long documents because the extractor receives one large block.',
        semantic: 'Best contextual boundaries. Uses embeddings and is the slowest chunking choice.',
      },
      quality: {
        heuristic: 'Recommended. Applies lightweight quality checks without an external model call.',
        none: 'Skips quality filtering for maximum speed; noisy input can produce a noisier graph.',
      },
      dedup: {
        minhash: 'Recommended at 0.85: fast, robust near-duplicate detection for normal documents.',
        none: 'Fastest, but repeated text can create repeated entities and facts.',
        exact: 'Removes identical text only. Very fast, but misses small wording changes.',
        simhash: 'Fast near-match detection; useful for lightly edited copies.',
        ngram: 'Balances lexical precision and speed for similar phrasing.',
        semantic: 'Finds paraphrases by meaning. Better recall, with extra embedding time.',
        layered: 'Most thorough option: combines multiple checks, with the highest runtime cost.',
      },
      resolution: {
        string: 'Recommended for the demo. Fast and deterministic; merges names with strong textual similarity.',
        embedding: 'Best for aliases and semantic variants, but slower because it computes embeddings.',
      },
    };
    $('#extractionHelp').textContent = guidance.extraction[$('#method').value];
    $('#chunkHelp').textContent = guidance.chunk[$('#chunkMethod').value];
    $('#qualityHelp').textContent = guidance.quality[$('#qualityMethod').value];
    $('#documentDedupHelp').textContent = guidance.dedup[$('#documentDedupMethod').value];
    $('#chunkDedupHelp').textContent = guidance.dedup[$('#dedupMethod').value];
    $('#resolutionHelp').textContent = guidance.resolution[$('#resolveMethod').value];
  }

  async function loadOptions() { try { state.options = await getJSON('/api/options'); applyAvailability(state.options); } catch (error) { /* defaults remain usable */ } }

  function setStage(index) { document.querySelectorAll('[data-stage]').forEach((element, position) => { element.classList.toggle('active', position === index); element.classList.toggle('done', position < index); }); }

  const clampScore = (value) => Math.max(0, Math.min(100, Number(value) || 0));
  const scoreTone = (score) => score >= 80 ? 'var(--teal)' : (score >= 60 ? '#f2cc68' : 'var(--coral)');
  const scoreFlag = (score) => score >= 80 ? 'green' : (score >= 60 ? 'yellow' : 'red');
  const percentScore = (value) => clampScore(Number(value) <= 1 ? Number(value) * 100 : value);

  function resetAudit() {
    $('#metricsPanel').innerHTML = '<div class="metrics-head"><div><small class="eyebrow">METHOD 1 / STRUCTURAL AUDIT</small><h2>LLM training readiness</h2></div><span class="audit-badge">Audit running</span></div><div class="metrics-body empty-metrics"><div class="score-ring" style="--score:0"><strong>—</strong><small>overall health</small></div><p>Checking graph structure for orphan nodes, density, schema compliance, duplicate entities, and multi-hop reasoning.</p></div>';
  }

  function renderAudit(metrics) {
    const audit = metrics.structural_audit || {};
    const fallbackOverall = percentScore(metrics.overall_score ?? metrics.quality_score ?? metrics.quality ?? 0);
    const overall = clampScore(audit.overall_health_score ?? fallbackOverall);
    const orphan = audit.orphan_analysis || {};
    const density = audit.density_analysis || {};
    const schema = audit.schema_compliance || {};
    const duplication = audit.entity_duplication || {};
    const multiHop = audit.multi_hop_connectivity || {};
    const cards = [
      ['Orphan control', orphan.health_score ?? 100, `${Math.round(Number(orphan.orphan_count || 0))} disconnected nodes`],
      ['Density balance', density.health_score ?? percentScore(metrics.consistency ?? 0), `density ${Number(density.density || 0).toFixed(3)}`],
      ['Schema compliance', schema.health_score ?? percentScore(metrics.consistency ?? 0), `${Math.round(Number(schema.violation_count || 0))} violations`],
      ['Entity uniqueness', duplication.health_score ?? percentScore(1 - Number(metrics.duplication_level || 0)), `${Math.round(Number(duplication.duplicate_pair_count || 0))} duplicate pairs`],
      ['Multi-hop reach', multiHop.health_score ?? percentScore(metrics.reusability ?? 0), `${Math.round(Number(multiHop.reachable_3hop_pct || 0) * 100)}% within 3 hops`],
    ];
    const recommendations = [orphan, density, schema, duplication, multiHop].map((item) => item.recommendation).filter(Boolean);
    const verdict = String(audit.verdict || (overall >= 80 ? 'Structurally ready for training-data generation.' : 'Review graph quality before using it for training data.')).replace(/^[^A-Za-zÀ-ỹ]+/, '');
    $('#metricsPanel').innerHTML = `<div class="metrics-head"><div><small class="eyebrow">METHOD 1 / STRUCTURAL AUDIT</small><h2>LLM training readiness</h2></div><span class="audit-badge ${scoreFlag(overall)}">${overall >= 80 ? 'Training ready' : (overall >= 60 ? 'Review advised' : 'Needs work')}</span></div><div class="metrics-body"><div class="score-summary"><div class="score-ring" style="--score:${overall};--tone:${scoreTone(overall)}"><strong>${Math.round(overall)}</strong><small>overall health</small></div><p>${esc(verdict)}</p></div><div class="audit-grid">${cards.map(([label, rawScore, detail]) => { const score = clampScore(rawScore); return `<div class="audit-card"><div class="audit-card-top"><small>${esc(label)}</small><b>${Math.round(score)}</b></div><small>${esc(detail)}</small><div class="audit-bar" style="--value:${score};--tone:${scoreTone(score)}"><i></i></div></div>`; }).join('')}<div class="metric-notes"><strong>${recommendations.length ? 'REVIEW' : 'AUDIT NOTE'}</strong><span>${esc(recommendations.join(' ') || 'No structural warnings detected. These metrics assess graph readiness; they do not measure the final fine-tuned model.')}</span></div></div></div>`;
    return overall;
  }

  function selectedConfig() {
    return { language: $('#language').value, extraction: $('#method').value, llm_model: 'deepseek-v4-flash', chunk_method: $('#chunkMethod').value, chunk_size: Number($('#chunkSize').value || 0), chunk_overlap: Number($('#chunkOverlap').value || 0), chunk_target_tokens: Number($('#chunkTargetTokens').value || 450), chunk_overlap_tokens: Number($('#chunkOverlapTokens').value || 60), semantic_chunk_threshold: Number($('#semanticChunkThreshold').value || 0.55), quality_method: $('#qualityMethod').value, document_dedup_method: $('#documentDedupMethod').value, dedup_method: $('#dedupMethod').value, document_dedup_threshold: Number($('#dedupThreshold').value || 0.85), dedup_threshold: Number($('#dedupThreshold').value || 0.85), semantic_dedup_threshold: Number($('#dedupThreshold').value || 0.92), resolve_method: $('#resolveMethod').value, resolve_threshold: 0.85, graphgen_max_gleanings: 3 };
  }

  function renderLab(raw) {
    const result = normalize(raw); state.lab = result; const readiness = renderAudit(result.metrics);
    const extraction = result.metrics?.extraction?.method === 'graphgen' || $('#method').value === 'graphgen' ? 'LLM-guided extraction' : 'Rules-based experiment';
    $('#chips').innerHTML = `<span>Entities ${result.stats.entities}</span><span>Relations ${result.stats.relations}</span><span>Triples ${result.stats.triples}</span><span>Training readiness ${Math.round(readiness)}%</span><span>${esc(extraction)}</span><span>${result.persistence?.status === 'persisted' ? 'Interactive graph replaced' : 'Preview generated'}</span>`;
    $('#elapsed').textContent = `${Number(result.metadata?.elapsed_seconds ?? ((Date.now() - state.started) / 1000)).toFixed(1)}s`; setStage(4); mount('lab', result);
    if (result.persistence?.status === 'failed') { $('#error').textContent = `Graph generated, but the interactive Neo4j graph was not replaced. ${result.persistence.reason || 'The previous graph was preserved.'}`; $('#error').classList.remove('hidden'); }
    if (result.persistence?.status === 'skipped') { $('#error').textContent = 'Preview generated. Interactive Neo4j is not configured, so the stored graph was not replaced.'; $('#error').classList.remove('hidden'); }
  }

  async function run() {
    const text = $('#source').value.trim(); if (!text) { $('#error').textContent = 'Add text or load the sample first.'; $('#error').classList.remove('hidden'); return; }
    $('#error').classList.add('hidden'); $('#run').disabled = true; state.started = Date.now(); setStage(0); resetAudit(); $('#labGraph').innerHTML = '<div class="empty"><b>Running pipeline…</b><small>Ingesting, extracting, resolving, and auditing the graph.</small></div>';
    try {
      const response = await fetch('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source: { text, title: $('#filename').textContent === 'No file selected' ? 'Interactive source' : $('#filename').textContent, license: 'Demo' }, options: selectedConfig() }) });
      if (response.status === 409) throw new Error('Another visitor is running the pipeline. Please wait and retry.');
      const body = await response.json(); if (!response.ok) throw new Error(body.detail || `Pipeline returned ${response.status}.`); renderLab(body);
    } catch (error) { $('#error').textContent = error.message; $('#error').classList.remove('hidden'); setStage(0); } finally { $('#run').disabled = false; }
  }

  document.querySelectorAll('.view').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('.view').forEach((item) => item.classList.toggle('active', item === button)); $('#globalView').classList.toggle('hidden', button.dataset.view !== 'global'); $('#labView').classList.toggle('hidden', button.dataset.view !== 'lab'); }));
  document.querySelectorAll('.tab').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('.tab').forEach((item) => item.classList.toggle('active', item === button)); ['sample', 'paste', 'upload'].forEach((name) => $(`#${name}Pane`).classList.toggle('hidden', button.dataset.tab !== name)); }));
  $('#method').addEventListener('change', () => { applyAvailability(state.options); });
  $('#chunkMethod').addEventListener('change', () => { const method = $('#chunkMethod').value; const fixed = method === 'fixed'; const advanced = method === 'sentence' || method === 'semantic'; $('#chunkSizeField').classList.toggle('hidden', !fixed); $('#chunkOverlapField').classList.toggle('hidden', !fixed); $('#chunkTargetField').classList.toggle('hidden', !advanced); $('#chunkTokenOverlapField').classList.toggle('hidden', !advanced); $('#semanticThresholdField').classList.toggle('hidden', method !== 'semantic'); updateGuidance(); });
  ['qualityMethod', 'documentDedupMethod', 'dedupMethod', 'resolveMethod'].forEach((id) => $(`#${id}`).addEventListener('change', updateGuidance));
  $('#loadSample').addEventListener('click', async () => { try { const result = await getJSON('/api/demo/sample'); $('#source').value = result.text || ''; $('#count').textContent = $('#source').value.length; document.querySelector('[data-tab="paste"]').click(); } catch (error) { $('#error').textContent = 'Sample unavailable; paste your own text.'; $('#error').classList.remove('hidden'); } });
  $('#source').addEventListener('input', () => { $('#count').textContent = $('#source').value.length; });
  $('#file').addEventListener('change', (event) => { const file = event.target.files[0]; if (!file) return; $('#filename').textContent = file.name; const reader = new FileReader(); reader.onload = () => { $('#source').value = String(reader.result).slice(0, 20000); $('#count').textContent = $('#source').value.length; document.querySelector('[data-tab="paste"]').click(); }; reader.readAsText(file); });
  $('#run').addEventListener('click', run); $('#globalSearch').addEventListener('input', (event) => search('global', event.target.value)); $('#labSearch').addEventListener('input', (event) => search('lab', event.target.value)); $('#globalFit').addEventListener('click', () => state.cyGlobal?.fit(undefined, 35)); $('#labFit').addEventListener('click', () => state.cyLab?.fit(undefined, 35)); $('#globalReset').addEventListener('click', () => { state.cyGlobal?.elements().removeStyle('opacity'); state.cyGlobal?.fit(undefined, 35); });
  $('#download').addEventListener('click', () => { if (!state.lab) return; const url = URL.createObjectURL(new Blob([JSON.stringify(state.lab, null, 2)], { type: 'application/json' })); const link = document.createElement('a'); link.href = url; link.download = 'vietgraph-result.json'; link.click(); URL.revokeObjectURL(url); });
  $('#globalLoadAll').addEventListener('click', () => { $('#globalLoadAll').disabled = true; $('#globalLoadAll').textContent = 'Loading…'; loadGlobal(5000).finally(() => { $('#globalLoadAll').textContent = 'All nodes loaded'; }); });
  $('#chunkMethod').dispatchEvent(new Event('change')); updateGuidance(); loadOptions(); loadGlobal();
})();
