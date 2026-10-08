// Judgment in the one report layout (ARCHITECTURE §7, owner decisions 2026-09-28).
// Draws claim underlines from #judgment-data, starts or resumes the run, polls
// results and fills each citation window: the label's judgment part and state,
// the note above the collapsed evidence, and any sentence the judge relied on
// that the evidence list does not already hold. A citation with several
// propositions has one window per proposition (Citation N - A, B, ...); the
// window shows only that proposition's results.
(() => {
  const dataEl = document.getElementById('judgment-data');
  const layout = window.sourceFidelityLayout;
  if (!dataEl || !layout) return;
  const data = JSON.parse(dataEl.textContent);
  const base = `/report/${encodeURIComponent(data.report_id)}/judgment`;
  const panel = document.getElementById('evidence-panel');
  const live = document.getElementById('judgment-live');
  const SVG = 'http://www.w3.org/2000/svg';
  // An undecided judge ranks below any decision (owner decision 2026-09-29).
  const SEVERITY = {insufficient: 4, contradicts: 3, qualified: 2, supported: 1, undecided: 0.5};
  // Owner-approved labels (2026-09-28).
  const NAMES = {supported: 'Supports', qualified: 'Qualified or Mixed', contradicts: 'Contradicts',
                 insufficient: 'Not Supported', undecided: 'LLM Undecided', not_judged: 'Not Judged', pending: ''};
  const byKey = new Map(data.marks.map(mark => [mark.key, mark]));
  const groups = new Map();
  for (const mark of data.marks) {
    if (!groups.has(mark.group)) groups.set(mark.group, {key: mark.group, marks: [], els: [], citation: mark.citation, order: mark.order});
    groups.get(mark.group).marks.push(mark);
  }
  const ordered = [...groups.values()].sort((a, b) => (a.order - b.order) || ((a.citation || 0) - (b.citation || 0)));
  // One underline per word (owner review 2026-10-08): a proposition takes a
  // second, lower line only where it would overlap another one of its
  // citation on the page, and never more than one: a third line ran into the
  // line of text below.
  const overlaps = (a, b) => a.page_index === b.page_index && a.x0 < b.x1 - 1 && b.x0 < a.x1 - 1
    && a.y0 < b.y1 - 1 && b.y0 < a.y1 - 1;
  const firstRects = group => (group.marks.find(m => m.placement === 'exact') || group.marks.find(m => m.rects.length) || {rects: []}).rects;
  const laid = new Map();
  for (const group of ordered) {
    const placed = laid.get(group.citation) || [];
    const mine = firstRects(group);
    group.lane = placed.some(other => other.lane === 0 && firstRects(other).some(r => mine.some(q => overlaps(r, q)))) ? 1 : 0;
    placed.push(group); laid.set(group.citation, placed);
  }
  const results = new Map();          // verification report id -> Map(candidate -> result item)
  const JUDGED = new Set(['supported', 'qualified', 'contradicts', 'insufficient']);
  const ANSWERED = new Set([...JUDGED, 'undecided']);
  const isProposition = group => !/:(whole|static)$/.test(group.key);
  const propositionsOf = number => ordered.filter(g => g.citation === number && isProposition(g));
  let polling = null, after = 0, delay = 1000, finished = false, filling = false;

  const say = text => { if (live) live.textContent = text; };
  const worst = states => {
    const judged = states.filter(s => s in SEVERITY);
    if (judged.length) return judged.reduce((a, b) => SEVERITY[a] >= SEVERITY[b] ? a : b);
    return states.some(s => s === 'pending') ? 'pending' : 'not_judged';
  };
  const groupState = group => worst(group.marks.map(m => m.state));
  const rectsOf = group => (group.marks.find(m => m.placement === 'exact') || group.marks.find(m => m.rects.length) || {rects: []}).rects;

  function shape(g, rect, state, lane) {
    const y = rect.y1 + 0.7 + lane * 2.6;
    const line = yy => { const l = document.createElementNS(SVG, 'line');
      Object.entries({x1: rect.x0, y1: yy, x2: rect.x1, y2: yy}).forEach(([k, v]) => l.setAttribute(k, v));
      l.setAttribute('class', `judgment-line state-${state}`); g.append(l); };
    line(y);
    if (state === 'insufficient' || state === 'undecided') line(y + 2.4);
  }
  function paint(group) {
    const state = groupState(group), rects = rectsOf(group);
    group.els.forEach((g, i) => {
      g.replaceChildren();
      const r = rects[i], hit = document.createElementNS(SVG, 'rect');
      Object.entries({x: r.x0, y: r.y0, width: r.x1 - r.x0, height: r.y1 - r.y0 + 3}).forEach(([k, v]) => hit.setAttribute(k, v));
      hit.setAttribute('class', 'judgment-hit'); g.append(hit);
      shape(g, r, state, group.lane);
      g.setAttribute('data-state', state);
      g.setAttribute('aria-label', `Citation ${group.citation ?? ''}${NAMES[state] ? ': ' + NAMES[state] : ''}`);
    });
  }
  // Larger claims are drawn first, so a claim contained in another sits on top
  // and stays clickable (owner review 2026-09-29).
  const extent = group => rectsOf(group).reduce((sum, r) => sum + (r.x1 - r.x0) * (r.y1 - r.y0), 0);
  function draw() {
    for (const group of [...ordered].sort((a, b) => extent(b) - extent(a))) {
      rectsOf(group).forEach((rect, i) => {
        const svg = document.querySelector(`svg.page-surface[data-page-index="${rect.page_index}"]`);
        if (!svg) return;
        const g = document.createElementNS(SVG, 'g');
        g.setAttribute('class', 'judgment-overlay'); g.setAttribute('data-judgment-group', group.key);
        g.setAttribute('role', 'button'); g.setAttribute('tabindex', i === 0 ? '0' : '-1');
        svg.append(g); group.els.push(g);
      });
      paint(group);
    }
    // A patchwriting highlight stays on top, so a click on it opens the window that shows it.
    document.querySelectorAll('svg.page-surface .patchwriting-overlay').forEach(el => el.parentNode.append(el));
  }

  // One source part of the open citation window, for its proposition when it has several.
  function fillRecord(record) {
    const proposition = panel.dataset.proposition || '';
    const inGroup = item => {
      if (!proposition) return true;
      const mark = byKey.get(`${item.citation_number}:${record}:${item.candidate_id}`);
      return !!mark && mark.group === proposition;
    };
    const items = [...(results.get(record)?.values() || [])].filter(inGroup).sort((a, b) => a.seq - b.seq);
    const waiting = !finished && data.marks.some(m => m.record === record && m.state === 'pending'
                                                  && (!proposition || m.group === proposition));
    for (const label of panel.querySelectorAll(`[data-judgment-label="${CSS.escape(record)}"]`)) {
      const state = items.length ? worst(items.map(i => i.display_state)) : (waiting ? 'pending' : 'not_judged');
      if (state === 'pending') continue;
      label.className = `judgment-label jk state-${state}`;
      const part = label.querySelector('.judgment-part'), sep = label.querySelector('.judgment-sep');
      if (part) { part.textContent = NAMES[state]; part.hidden = false; }
      if (sep) sep.hidden = false;
    }
    for (const slot of panel.querySelectorAll(`[data-judgment-record="${CSS.escape(record)}"]`)) {
      slot.innerHTML = items.map(i => i.window_html).join('');
      const member = slot.closest('.member');
      // The judge found evidence: the selector's "no clearly relevant passage"
      // line no longer applies (owner decision 2026-10-08).
      const judgeFound = items.some(i => (i.evidence || []).length || ['supported', 'qualified', 'contradicts'].includes(i.display_state));
      member?.querySelectorAll('[data-no-passage-note]').forEach(note => { note.hidden = judgeFound; });
      let list = member?.querySelector('[data-evidence-list]');
      if (!list && items.some(i => (i.evidence || []).length)) {
        // No stored selection (a report checked before 2026-09-28, or the gate
        // fallback): the sentences the judge relied on form the evidence list.
        const details = document.createElement('details'); details.className = 'evidence-disclosure';
        const summary = document.createElement('summary'); summary.textContent = 'Evidence';
        list = document.createElement('ul'); list.className = 'evidence-sentences'; list.dataset.evidenceList = '';
        details.append(summary, list); slot.after(details);
      }
      if (!list) continue;
      // The window is filled again on each visit: restore the sentence the
      // last pass lifted out as the key sentence before choosing again.
      for (const li of list.querySelectorAll('[data-key-lifted]')) { li.hidden = false; delete li.dataset.keyLifted; }
      // One sentence split two ways by overlapping passages ("p:12090:12246"
      // and "p:12090:12375") is listed once, with the longer text (2026-10-02).
      const span = key => { const m = /^(.*):(\d+):(\d+)$/.exec(key || ''); return m ? [m[1], +m[2], +m[3]] : null; };
      const overlapping = key => {
        const a = span(key); if (!a) return null;
        for (const li of list.querySelectorAll('[data-evidence-key]')) {
          const b = span(li.dataset.evidenceKey);
          if (b && a[0] === b[0] && a[1] < b[2] && b[1] < a[2]) return li;
        }
        return null;
      };
      for (const item of items) {
        for (const sentence of item.evidence || []) {
          if (list.querySelector(`[data-evidence-key="${CSS.escape(sentence.key)}"]`)) continue;
          const same = overlapping(sentence.key);
          if (same) {
            const q = same.querySelector('q');
            if (q && (sentence.text || '').length > q.textContent.length) q.textContent = sentence.text;
            continue;
          }
          const li = document.createElement('li'); li.dataset.evidenceKey = sentence.key;
          if (sentence.page) { const page = document.createElement('span'); page.className = 'ev-page'; page.textContent = `p. ${sentence.page}`; li.append(page, ' '); }
          const q = document.createElement('q'); q.textContent = sentence.text; li.append(q); list.append(li);
        }
      }
      // A Supports result shows its key sentence outside the list (owner decision 2026-09-30).
      // A Supports result leads with the judge-cited sentence the evidence
      // selector marked as bearing on the statement, when there is one
      // (owner decision 2026-10-02); otherwise the most-cited sentence.
      for (const shown of slot.querySelectorAll('[data-key-evidence]')) {
        let alternatives = [];
        try { alternatives = JSON.parse(shown.dataset.keyAlternatives || '[]'); } catch (e) { alternatives = []; }
        const bears = alternatives.find(a => {
          const li = list.querySelector(`[data-evidence-key="${CSS.escape(a.key)}"]`) || overlapping(a.key);
          return li && li.dataset.reason === 'bears_on_statement';
        });
        if (bears && bears.key !== shown.dataset.keyEvidence) {
          shown.dataset.keyEvidence = bears.key;
          shown.textContent = '';
          if (bears.page) { const page = document.createElement('span'); page.className = 'ev-page'; page.textContent = `p. ${bears.page}`; shown.append(page, ' '); }
          const q = document.createElement('q'); q.textContent = bears.text; shown.append(q);
        }
      }
      for (const shown of slot.querySelectorAll('[data-key-evidence]')) {
        const lifted = list.querySelector(`[data-evidence-key="${CSS.escape(shown.dataset.keyEvidence)}"]`)
          || overlapping(shown.dataset.keyEvidence);
        if (lifted) { lifted.hidden = true; lifted.dataset.keyLifted = ''; }
      }
      const disclosure = list.closest('details');
      if (disclosure) disclosure.hidden = !list.querySelector('li:not([hidden])');
    }
  }
  function fillPanel() {
    if (filling) return;
    filling = true;
    try {
      const records = new Set([...panel.querySelectorAll('[data-judgment-record],[data-judgment-label]')]
        .map(el => el.dataset.judgmentRecord || el.dataset.judgmentLabel));
      records.forEach(fillRecord);
    } finally { filling = false; }
  }
  new MutationObserver(() => fillPanel()).observe(panel, {childList: true});

  // Light-blue selection and hover (owner request 2026-09-28): the whole
  // citation, or only the proposition when the citation has several.
  const citationOverlays = number => document.querySelectorAll(
    `.citation-overlay:not(.member-target)[data-panel-template="citation-panel-${number}"]`);
  function markSelected(number, proposition) {
    document.querySelectorAll('.judgment-overlay.selected').forEach(el => el.classList.remove('selected'));
    document.querySelectorAll('.citation-overlay.proposition-mode').forEach(el => el.classList.remove('proposition-mode'));
    if (number == null) return;
    const props = propositionsOf(number);
    if (props.length < 2) return;
    const chosen = props.find(g => g.key === proposition) || props[0];
    chosen.els.forEach(el => el.classList.add('selected'));
    citationOverlays(number).forEach(el => el.classList.add('proposition-mode'));
  }
  document.addEventListener('sourcefidelity:open', event => markSelected(event.detail.citation, event.detail.proposition));
  function hover(group, on) {
    if (!group) return;
    if (propositionsOf(group.citation).length > 1 && isProposition(group)) group.els.forEach(el => el.classList.toggle('hovered', on));
    else citationOverlays(group.citation).forEach(el => el.classList.toggle('hovered', on));
  }
  document.addEventListener('pointerover', event => {
    const el = event.target.closest?.('.judgment-overlay');
    if (el) hover(groups.get(el.dataset.judgmentGroup), true);
  });
  document.addEventListener('pointerout', event => {
    const el = event.target.closest?.('.judgment-overlay');
    if (el && !el.contains(event.relatedTarget)) hover(groups.get(el.dataset.judgmentGroup), false);
  });
  function openGroup(group) {
    if (!group || group.citation == null) return;
    layout.openCitation(group.citation, isProposition(group) ? group.key : null);
  }
  document.addEventListener('click', event => {
    const retry = event.target.closest('[data-judgment-retry]');
    if (retry) { event.preventDefault(); retryRun(); return; }
    const overlay = event.target.closest('.judgment-overlay');
    if (overlay) { event.preventDefault(); event.stopPropagation(); openGroup(groups.get(overlay.dataset.judgmentGroup)); }
  }, true);
  document.addEventListener('keydown', event => {
    const el = event.target.closest?.('.judgment-overlay');
    if (el && ['Enter', ' '].includes(event.key)) { event.preventDefault(); openGroup(groups.get(el.dataset.judgmentGroup)); }
  });

  async function post(path) {
    const response = await fetch(`${base}/${path}`, {method: 'POST', credentials: 'same-origin', headers: {'Accept': 'application/json'}});
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  }
  function apply(items) {
    for (const item of items) {
      after = Math.max(after, item.seq);
      if (!results.has(item.verification_report_id)) results.set(item.verification_report_id, new Map());
      results.get(item.verification_report_id).set(item.candidate_id, item);
      const mark = byKey.get(`${item.citation_number}:${item.verification_report_id}:${item.candidate_id}`);
      if (mark) { mark.state = item.display_state; paint(groups.get(mark.group)); }
    }
    if (items.length) fillPanel();
  }
  async function poll() {
    try {
      const response = await fetch(`${base}/results?after=${after}`, {credentials: 'same-origin', headers: {'Accept': 'application/json'}});
      if (!response.ok) throw new Error(String(response.status));
      const body = await response.json();
      apply(body.items || []);
      const run = body.run || {};
      const done = data.marks.filter(m => m.state !== 'pending').length;
      if (['completed', 'unavailable', 'failed'].includes(run.status) && !(body.items || []).length) {
        finished = true;
        data.marks.forEach(m => { if (m.state === 'pending') m.state = 'not_judged'; });
        ordered.forEach(paint); fillPanel(); updateCounts();
        say(`Judgment finished. ${done} of ${data.marks.length} statements have a result.`);
        return;
      }
      say(`Judging: ${done} of ${data.marks.length} statements.`);
      delay = (body.items || []).length ? 1000 : Math.min(delay * 2, 5000);
    } catch (error) {
      delay = 5000;
    }
    polling = setTimeout(poll, delay);
  }
  // The header's judged-citation count and the upload estimate follow the results.
  function updateCounts() {
    const judgments = data.marks.filter(m => ANSWERED.has(m.state) && !m.static).length;
    document.querySelectorAll('[data-judgment-count]').forEach(el => {
      el.textContent = `${judgments.toLocaleString()} judgment${judgments === 1 ? '' : 's'}`; });
    // The Sources summary's judgment sentence (owner wording 2026-09-29).
    const summary = document.querySelector('[data-judgment-summary]');
    if (summary) {
      const counts = {};
      data.marks.filter(m => ANSWERED.has(m.state) && !m.static).forEach(m => { counts[m.state] = (counts[m.state] || 0) + 1; });
      const x = Object.values(counts).reduce((a, b) => a + b, 0);
      // "statements" appears once, in the first part shown (owner request 2026-09-29).
      const parts = [['supported', (n, s) => `${n}/${x}${s} are supported by the sources`],
                     ['qualified', (n, s) => `${n}/${x}${s} have qualified or mixed support in the sources`],
                     ['contradicts', (n, s) => `${n}/${x}${s} contradict the sources`],
                     ['insufficient', (n, s) => `${n}/${x}${s} are not supported by the sources`],
                     ['undecided', (n, s) => `${n}/${x}${s} cannot be decided upon by the LLM`]]
        .filter(([key]) => counts[key]).map(([key, text], i) => text(counts[key], i === 0 ? ' statements' : ''));
      const lacking = Number(summary.dataset.withoutFullText || 0);
      if (lacking) parts.push(`${lacking}/${summary.dataset.citationTotal} citations lack full texts and are not judged`);
      summary.textContent = parts.length ? (parts.length === 1 ? parts[0] : parts.slice(0, -1).join(', ') + ', and ' + parts[parts.length - 1]) + '.' : '';
      summary.hidden = !parts.length;
    }
  }
  function startPolling() { clearTimeout(polling); finished = false; delay = 1000; polling = setTimeout(poll, 300); }
  async function retryRun() {
    try {
      await post('retry');
      after = 0; results.clear();
      data.marks.forEach(m => { if (!m.static) m.state = 'pending'; });
      ordered.forEach(paint);
      startPolling();
    } catch (error) { /* the run stays as it was */ }
  }

  draw();
  if (data.static_results) {
    // An exported report: the finished results travel in the file; nothing is fetched.
    apply(data.static_results);
    finished = true;
    data.marks.forEach(m => { if (m.state === 'pending') m.state = 'not_judged'; });
    ordered.forEach(paint); fillPanel(); updateCounts();
  } else {
    post('start').then(() => startPolling(), () => startPolling());   // resumes the check-time run, or starts one
  }
  window.sourceFidelityJudgment = {fill: fillPanel};
})();
