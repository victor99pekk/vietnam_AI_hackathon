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
    const cleanNodes = nodes.map((node, index) => { const data = node.data || node; return { ...data, id: String(data.id ?? data.node_id ?? `node-${index}`), label: data.label || data.name || data.text || `Node ${index + 1}`, type: String(data.type || data.entity_type || 'CONCEPT').toUpperCase(), evidence: data.evidence || data.sentence || data.context || '' }; });
    const cleanEdges = [];
    edges.forEach((edge, index) => {
      const data = edge.data || edge;
      const source = data.source ?? data.from ?? data.subject;
      const target = data.target ?? data.to ?? data.object;
      const predicates = Array.isArray(data.predicates) ? data.predicates : (Array.isArray(data.predicate) ? data.predicate : [data.label || data.relation || data.predicate || 'related_to']);
      if (source && target) predicates.forEach((predicate, predicateIndex) => cleanEdges.push({ ...data, id: String(data.id ?? `edge-${index}-${predicateIndex}`), source: String(source), target: String(target), label: String(predicate) }));
    });
    const stats = raw?.stats || {};
    return { ...raw, graph: { nodes: cleanNodes, edges: cleanEdges }, stats: { ...stats, entities: stats.entities ?? stats.num_nodes ?? cleanNodes.length, relations: stats.relations ?? stats.num_edges ?? cleanEdges.length, triples: stats.triples ?? stats.num_triples ?? raw?.triples?.length ?? cleanEdges.length }, metrics: raw?.metrics || {} };
  }

  function graphStyle() { return [{ selector: 'node', style: { 'background-color': (element) => ({ PERSON: '#ff8c70', PLACE: '#55d4c4', GPE: '#55d4c4', ORGANISATION: '#9f8cff', ORG: '#9f8cff', CONCEPT: '#f2cc68' }[element.data('type')] || '#55d4c4'), label: 'data(label)', color: '#eaf2f3', 'font-size': 10, 'text-valign': 'bottom', 'text-margin-y': 8, 'border-color': '#0b1928', 'border-width': 2, width: 17, height: 17 } }, { selector: 'edge', style: { width: 1, 'line-color': '#40606a', 'target-arrow-color': '#40606a', 'target-arrow-shape': 'triangle', label: 'data(label)', color: '#7d999f', 'font-size': 8, 'curve-style': 'bezier' } }, { selector: ':selected', style: { 'border-color': '#fff', 'border-width': 3 } }]; }

  function mount(which, result) {
    const container = $(`#${which}Graph`); container.innerHTML = '';
    if (!result.graph.nodes.length || typeof cytoscape === 'undefined') { container.innerHTML = '<div class="empty"><b>No entities found</b><small>Try a longer document.</small></div>'; return; }
    const key = which === 'global' ? 'cyGlobal' : 'cyLab'; state[key]?.destroy();
    state[key] = cytoscape({ container, elements: { nodes: result.graph.nodes.map((node) => ({ data: node })), edges: result.graph.edges.map((edge) => ({ data: edge })) }, layout: { name: 'cose', animate: false, padding: 30 }, style: graphStyle() });
    state[key].on('tap', 'node', (event) => showDetail(which, event.target.data()));
  }

  function showDetail(which, node) {
    const result = state[which]; if (!result) return;
    const links = result.graph.edges.filter((edge) => edge.source === node.id || edge.target === node.id);
    $(`#${which}Detail`).innerHTML = `<div class="detail-content"><div><span class="detail-type">${esc(node.type)}</span><h3>${esc(node.label)}</h3><p>${esc(node.description || `${links.length} linked fact${links.length === 1 ? '' : 's'} in this graph.`)}</p></div><div class="evidence"><span class="evidence-label">EVIDENCE / SOURCE SENTENCE</span><blockquote>“${esc(node.evidence || 'Evidence sentence not provided by the extractor.')}”</blockquote></div></div>`;
  }

  function search(which, value) { const cy = which === 'global' ? state.cyGlobal : state.cyLab; if (!cy) return; const query = value.toLowerCase(); cy.nodes().forEach((node) => node.style('opacity', !query || String(node.data('label')).toLowerCase().includes(query) ? 1 : 0.14)); }

  async function loadGlobal() {
    try {
      const result = normalize(await getJSON('/api/graphs/global?limit=150')); state.global = result;
      $('#globalBadge').textContent = result.metadata?.source === 'sample' ? 'DEMO DATA' : 'LIVE NEO4J';
      $('#gNodes').textContent = result.stats.entities; $('#gEdges').textContent = result.stats.relations; $('#gTriples').textContent = result.stats.triples; mount('global', result);
    } catch (error) { $('#globalBadge').textContent = 'UNAVAILABLE'; $('#globalGraph').innerHTML = '<div class="empty"><b>Global graph unavailable</b><small>Check the API connection.</small></div>'; }
  }

  function applyAvailability(options) {
    const availability = options?.availability || {};
    const method = $('#method'); const graphgen = method.querySelector('option[value="graphgen"]'); if (graphgen) { graphgen.disabled = availability.graphgen === false; graphgen.textContent = availability.graphgen === false ? 'GraphGen (API key needed)' : 'GraphGen'; }
    ['chunkMethod', 'documentDedupMethod', 'dedupMethod', 'resolveMethod'].forEach((id) => { const select = $(`#${id}`); if (!select) return; const advanced = [...select.options].filter((option) => ['semantic', 'layered', 'embedding'].includes(option.value)); advanced.forEach((option) => { option.disabled = availability.embeddings === false; if (availability.embeddings === false && option.value === select.value) select.value = select.options[0].value; }); });
    $('#modelField').classList.toggle('hidden', method.value !== 'graphgen');
  }

  async function loadOptions() { try { state.options = await getJSON('/api/options'); applyAvailability(state.options); } catch (error) { /* defaults remain usable */ } }

  function setStage(index) { document.querySelectorAll('[data-stage]').forEach((element, position) => { element.classList.toggle('active', position === index); element.classList.toggle('done', position < index); }); }
  function selectedConfig() {
    return { language: $('#language').value, extraction: $('#method').value, llm_model: $('#llmModel').value, chunk_method: $('#chunkMethod').value, chunk_size: Number($('#chunkSize').value || 0), chunk_overlap: Number($('#chunkOverlap').value || 0), chunk_target_tokens: Number($('#chunkTargetTokens').value || 450), chunk_overlap_tokens: Number($('#chunkOverlapTokens').value || 60), semantic_chunk_threshold: Number($('#semanticChunkThreshold').value || 0.55), quality_method: $('#qualityMethod').value, document_dedup_method: $('#documentDedupMethod').value, dedup_method: $('#dedupMethod').value, document_dedup_threshold: Number($('#dedupThreshold').value || 0.85), dedup_threshold: Number($('#dedupThreshold').value || 0.85), semantic_dedup_threshold: Number($('#dedupThreshold').value || 0.92), resolve_method: $('#resolveMethod').value, resolve_threshold: 0.85, graphgen_max_gleanings: 3 };
  }

  function renderLab(raw) {
    const result = normalize(raw); state.lab = result; const score = result.metrics.overall_score ?? result.metrics.quality_score ?? result.metrics.quality;
    $('#chips').innerHTML = `<span>Entities ${result.stats.entities}</span><span>Relations ${result.stats.relations}</span><span>Triples ${result.stats.triples}</span><span>Quality ${score == null ? '—' : `${Math.round(Number(score) * (Number(score) <= 1 ? 100 : 1))}%`}</span><span>${result.persistence?.status === 'persisted' ? 'Interactive graph replaced' : 'Preview generated'}</span>`;
    $('#elapsed').textContent = `${Number(result.metadata?.elapsed_seconds ?? ((Date.now() - state.started) / 1000)).toFixed(1)}s`; setStage(4); mount('lab', result);
    if (result.persistence?.status === 'failed') { $('#error').textContent = 'Graph generated, but the interactive Neo4j graph was not replaced.'; $('#error').classList.remove('hidden'); }
    if (result.persistence?.status === 'skipped') { $('#error').textContent = 'Preview generated. Interactive Neo4j is not configured, so the stored graph was not replaced.'; $('#error').classList.remove('hidden'); }
  }

  async function run() {
    const text = $('#source').value.trim(); if (!text) { $('#error').textContent = 'Add text or load the sample first.'; $('#error').classList.remove('hidden'); return; }
    $('#error').classList.add('hidden'); $('#run').disabled = true; state.started = Date.now(); setStage(0); $('#labGraph').innerHTML = '<div class="empty"><b>Running pipeline…</b><small>Ingesting, extracting, resolving and scoring.</small></div>';
    const ticker = setInterval(() => { const active = [...document.querySelectorAll('#labView [data-stage]')].findIndex((element) => element.classList.contains('active')); if (active < 3) setStage(active + 1); }, 700);
    try {
      const response = await fetch('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source: { text, title: $('#filename').textContent === 'No file selected' ? 'Interactive source' : $('#filename').textContent, license: 'Demo' }, options: selectedConfig() }) });
      if (response.status === 409) throw new Error('Another visitor is running the pipeline. Please wait and retry.');
      const body = await response.json(); if (!response.ok) throw new Error(body.detail || `Pipeline returned ${response.status}.`); renderLab(body);
    } catch (error) { $('#error').textContent = error.message; $('#error').classList.remove('hidden'); setStage(0); } finally { clearInterval(ticker); $('#run').disabled = false; }
  }

  document.querySelectorAll('.view').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('.view').forEach((item) => item.classList.toggle('active', item === button)); $('#globalView').classList.toggle('hidden', button.dataset.view !== 'global'); $('#labView').classList.toggle('hidden', button.dataset.view !== 'lab'); }));
  document.querySelectorAll('.tab').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('.tab').forEach((item) => item.classList.toggle('active', item === button)); ['sample', 'paste', 'upload'].forEach((name) => $(`#${name}Pane`).classList.toggle('hidden', button.dataset.tab !== name)); }));
  $('#method').addEventListener('change', () => { $('#modelField').classList.toggle('hidden', $('#method').value !== 'graphgen'); applyAvailability(state.options); });
  $('#chunkMethod').addEventListener('change', () => { const method = $('#chunkMethod').value; const fixed = method === 'fixed'; const advanced = method === 'sentence' || method === 'semantic'; $('#chunkSizeField').classList.toggle('hidden', !fixed); $('#chunkOverlapField').classList.toggle('hidden', !fixed); $('#chunkTargetField').classList.toggle('hidden', !advanced); $('#chunkTokenOverlapField').classList.toggle('hidden', !advanced); $('#semanticThresholdField').classList.toggle('hidden', method !== 'semantic'); });
  $('#loadSample').addEventListener('click', async () => { try { const result = await getJSON('/api/demo/sample'); $('#source').value = result.text || ''; $('#count').textContent = $('#source').value.length; document.querySelector('[data-tab="paste"]').click(); } catch (error) { $('#error').textContent = 'Sample unavailable; paste your own text.'; $('#error').classList.remove('hidden'); } });
  $('#source').addEventListener('input', () => { $('#count').textContent = $('#source').value.length; });
  $('#file').addEventListener('change', (event) => { const file = event.target.files[0]; if (!file) return; $('#filename').textContent = file.name; const reader = new FileReader(); reader.onload = () => { $('#source').value = String(reader.result).slice(0, 20000); $('#count').textContent = $('#source').value.length; document.querySelector('[data-tab="paste"]').click(); }; reader.readAsText(file); });
  $('#run').addEventListener('click', run); $('#globalSearch').addEventListener('input', (event) => search('global', event.target.value)); $('#labSearch').addEventListener('input', (event) => search('lab', event.target.value)); $('#globalFit').addEventListener('click', () => state.cyGlobal?.fit(undefined, 35)); $('#labFit').addEventListener('click', () => state.cyLab?.fit(undefined, 35)); $('#globalReset').addEventListener('click', () => { state.cyGlobal?.elements().removeStyle('opacity'); state.cyGlobal?.fit(undefined, 35); });
  $('#download').addEventListener('click', () => { if (!state.lab) return; const url = URL.createObjectURL(new Blob([JSON.stringify(state.lab, null, 2)], { type: 'application/json' })); const link = document.createElement('a'); link.href = url; link.download = 'vietgraph-result.json'; link.click(); URL.revokeObjectURL(url); });
  $('#modelField').classList.add('hidden'); $('#chunkMethod').dispatchEvent(new Event('change')); loadOptions(); loadGlobal();
})();
