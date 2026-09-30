(() => {
  'use strict';
  const body = document.body;
  const panel = document.getElementById('evidence-panel');
  const status = document.getElementById('report-status');
  const selectText = document.getElementById('select-text');
  let selectedTrigger = null, mode = null;
  // Source navigation touches only the current citation, not every paper word.
  let memberTargetsByPanel = new Map();
  function indexMemberTargets() {
    memberTargetsByPanel = new Map();
    document.querySelectorAll('.member-target').forEach(el => {
      const key = el.dataset.panelTemplate;
      if (!memberTargetsByPanel.has(key)) memberTargetsByPanel.set(key, []);
      memberTargetsByPanel.get(key).push(el);
    });
  }
  indexMemberTargets();
  // Catch only the gesture that crosses into the reading workspace. Release
  // after a short wheel-idle interval; never trap the next gesture or keyboard.
  let entryCaught=false, entryHolding=false, entryRelease=null;
  const readingLayout=document.getElementById('report-layout');
  function releaseEntryAfterGesture() {
    clearTimeout(entryRelease);
    entryRelease=setTimeout(()=>{entryHolding=false;},180);
  }
  document.addEventListener('wheel', event => {
    if (!readingLayout || event.ctrlKey || event.metaKey || !event.cancelable || Math.abs(event.deltaX)>Math.abs(event.deltaY)) return;
    const top=readingLayout.getBoundingClientRect().top;
    if (entryCaught && !entryHolding && top>80) entryCaught=false;
    if (entryHolding) {
      if (event.deltaY>0) {event.preventDefault();releaseEntryAfterGesture();}
      else {entryHolding=false;clearTimeout(entryRelease);}
      return;
    }
    if (entryCaught || event.deltaY<=0 || top<=0) return;
    if (event.target.closest('textarea,input,select,[contenteditable="true"]')) return;
    const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?window.innerHeight:1);
    if (delta<top) return;
    event.preventDefault();
    window.scrollTo({top:window.scrollY+top,behavior:'instant'});
    entryCaught=true;entryHolding=true;releaseEntryAfterGesture();
  },{passive:false});
  document.querySelectorAll('[data-report-step]').forEach(button => button.addEventListener('click', () => step(Number(button.dataset.reportStep))));
  function clearSelection() {
    document.querySelectorAll('[data-panel-template].selected').forEach(el => { el.classList.remove('selected'); el.setAttribute('aria-pressed', 'false'); });
  }
  let openedTemplate = null;
  function markSelected(id) {
    // A citation's light blue is cut around its patchwriting highlights, which
    // are selected with it.
    document.querySelectorAll('.citation-overlay:not(.member-target),.reference-entry-overlay,.patchwriting-overlay,.paper-badge').forEach(el => {
      if (el.dataset.panelTemplate === id) { el.classList.add('selected'); el.setAttribute('aria-pressed','true'); }
    });
  }
  // Open a window that has no mark on the paper (an unplaced citation or an
  // unlocated reference): the window still opens; nothing scrolls.
  // A citation with several judged propositions has one window per
  // proposition (owner request 2026-09-28): "Citation N - A", its wording in bold.
  function showProposition(proposition) {
    const texts = [...panel.querySelectorAll('.selected-citation[data-proposition]')];
    const suffix = panel.querySelector('[data-proposition-suffix]');
    if (!texts.length) { delete panel.dataset.proposition; return null; }
    const chosen = texts.find(el => el.dataset.proposition === proposition) || texts[0];
    texts.forEach(el => { el.hidden = el !== chosen; });
    panel.dataset.proposition = chosen.dataset.proposition;
    if (suffix) suffix.textContent = ' - ' + chosen.dataset.propositionLetter;
    return chosen.dataset.proposition;
  }
  function propositionsIn(template) {
    return [...(template?.content.querySelectorAll('.selected-citation[data-proposition]') || [])].map(el => el.dataset.proposition);
  }
  function announce(id, proposition) {
    const m = /^citation-panel-(\d+)$/.exec(id || '');
    document.dispatchEvent(new CustomEvent('sourcefidelity:open', {detail: {citation: m ? Number(m[1]) : null, proposition}}));
  }
  function openTemplate(id, proposition) {
    const template = document.getElementById(id);
    if (!template) return;
    clearSelection(); selectedTrigger = null; openedTemplate = id;
    markSelected(id);
    panel.replaceChildren(template.content.cloneNode(true));
    const chosen = showProposition(proposition);
    showMember(0); panel.scrollTop = 0;
    announce(id, chosen);
  }
  function chooseCitation(trigger, proposition) {
    const template = document.getElementById(trigger.dataset.panelTemplate);
    if (!template) return;
    clearSelection(); selectedTrigger = trigger; openedTemplate = trigger.dataset.panelTemplate;
    trigger.classList.add('selected'); trigger.setAttribute('aria-pressed', 'true');
    markSelected(trigger.dataset.panelTemplate);
    panel.replaceChildren(template.content.cloneNode(true));
    const chosen = showProposition(proposition);
    showMember(Number(trigger.dataset.memberIndex || 0));
    panel.scrollTop = 0;
    announce(trigger.dataset.panelTemplate, chosen);
  }
  // Light-blue hover on a citation (owner request 2026-09-28) or a patchwriting passage.
  function hoverCitation(el, on) {
    const id = el?.dataset.panelTemplate;
    if (el?.classList.contains('patchwriting-overlay') && !id?.startsWith('citation-panel-')) {
      document.querySelectorAll(`.patchwriting-overlay[data-passage="${el.dataset.passage}"]`)
        .forEach(overlay => overlay.classList.toggle('hovered', on));
      return;
    }
    if (!id || !id.startsWith('citation-panel-')) return;
    // A patchwriting highlight inside a citation opens that citation: both light up.
    document.querySelectorAll(`.citation-overlay:not(.member-target)[data-panel-template="${id}"],.patchwriting-overlay[data-panel-template="${id}"]`)
      .forEach(overlay => overlay.classList.toggle('hovered', on));
  }
  document.addEventListener('pointerover', event => hoverCitation(event.target.closest?.('.citation-overlay,.patchwriting-overlay'), true));
  document.addEventListener('pointerout', event => {
    const el = event.target.closest?.('.citation-overlay,.patchwriting-overlay');
    if (el && !el.contains(event.relatedTarget)) hoverCitation(el, false);
  });
  function showMember(index) {
    const members = [...panel.querySelectorAll('[data-source-member]')];
    if (!members.length) return;
    index = ((index % members.length) + members.length) % members.length;
    members.forEach((member, i) => { member.hidden = i !== index; });
    panel.dataset.memberIndex = String(index);
    const position = panel.querySelector('[data-source-position]');
    if (position) position.textContent = `Source ${index + 1} of ${members.length}`;
    if (selectedTrigger) {
      document.querySelectorAll('.member-target.selected').forEach(el => el.classList.remove('selected'));
      (memberTargetsByPanel.get(selectedTrigger.dataset.panelTemplate) || []).forEach(el => {
        const active = el.dataset.panelTemplate === selectedTrigger.dataset.panelTemplate && Number(el.dataset.memberIndex) === index;
        el.classList.toggle('selected', active); el.setAttribute('aria-pressed', String(active));
      });
    }
  }
  panel.addEventListener('click', event => {
    const arrow = event.target.closest('[data-source-step]');
    if (arrow) showMember(Number(panel.dataset.memberIndex || 0) + Number(arrow.dataset.sourceStep));
  });
  let suppressPaperClick = false;
  // A previous text selection must not swallow a later deliberate click.
  document.addEventListener('pointerdown', event => {
    if (mode !== 'text' && event.target.closest('.page-container')) {
      event.preventDefault();
      document.getSelection()?.removeAllRanges();
    }
    if (event.button !== 0 || !event.target.closest('.page-container')) return;
    suppressPaperClick = false;
    document.getSelection()?.removeAllRanges();
  }, true);
  function citationAt(event) {
    const direct = event.target.closest('[data-panel-template]');
    const elements = document.elementsFromPoint(event.clientX, event.clientY);
    const badge = elements.map(el => el.closest('.paper-badge')).find(Boolean);
    if (badge) return paperTargetFor(badge.dataset.panelTemplate) || badge;
    // A patchwriting highlight opens the window that shows it (its citation at
    // the matched source, or the source's Reference window), also inside a citation.
    const passage = elements.map(el => el.closest('.patchwriting-overlay')).find(Boolean);
    if (passage) return passage;
    const flag = elements.find(el => el.matches('.reference-field-marker,.reference-formatting-hit,.submitted-link-hit,.topical-reference-hit'));
    if (flag) {
      const owner=flag.closest('[data-panel-template]');
      if (flag.matches('.reference-formatting-hit') && owner?.dataset.panelTemplate.startsWith('reference-panel-')) {
        const citation=elements.map(el=>el.closest('.citation-overlay[data-panel-template]')).find(Boolean);
        if (citation) return citation;
      }
      return owner;
    }
    const hits = elements
      .map(el => el.closest('[data-panel-template]'))
      .filter(el => el && !el.matches('.reference-practice-overlay,.reference-entry-overlay,.paper-badge'));
    const entry = elements.map(el => el.closest('.reference-entry-overlay')).find(Boolean);
    return hits.find(el => el.classList.contains('member-target')) || hits[0] || entry ||
      (direct && !direct.matches('.reference-practice-overlay,.paper-badge') ? direct : null);
  }
  // Selecting a paper mark while the window is hidden brings it back first.
  function openFromPaper(trigger) {
    if (view === 'sources') chooseCitation(trigger);
    else setView('sources', {then: () => chooseCitation(trigger)});
  }
  document.addEventListener('click', event => {
    if (!event.target.closest('.page-container,[data-panel-template]')) return;
    if (suppressPaperClick) { suppressPaperClick = false; event.preventDefault(); return; }
    if (document.getSelection()?.toString().trim()) { event.preventDefault(); return; }
    const trigger = citationAt(event);
    if (trigger) { event.preventDefault(); openFromPaper(trigger); }
  });
  document.addEventListener('keydown', event => {
    const trigger = event.target.closest('[data-panel-template]');
    if (trigger && ['Enter', ' '].includes(event.key)) { event.preventDefault(); openFromPaper(trigger); }
  });
  function captureTextSelection() {
    if (mode !== 'text') return;
    const selection = document.getSelection();
    if (!selection || selection.isCollapsed || !selection.rangeCount) return;
    const range = selection.getRangeAt(0);
    const start = (range.startContainer.nodeType === 1 ? range.startContainer : range.startContainer.parentElement)?.closest('.paper-word');
    const end = (range.endContainer.nodeType === 1 ? range.endContainer : range.endContainer.parentElement)?.closest('.paper-word');
    if (!start || !end) return;
    const firstPage = +start.closest('.page-container').dataset.pageIndex;
    const lastPage = +end.closest('.page-container').dataset.pageIndex;
    if (firstPage !== lastPage) { selection.removeAllRanges(); return; }
    const words = [];
    // Walk only the selected interval. Avoid intersectsNode across a long PDF.
    for (let word = start; word; word = word.nextElementSibling) {
      if (word.classList.contains('paper-word')) words.push(word);
      if (word === end) break;
    }
    const pages = new Map();
    for (const word of words) {
      const page = Number(word.closest('.page-container').dataset.pageIndex), index = Number(word.dataset.wordIndex);
      if (!pages.has(page)) pages.set(page, {page_index:page, start:index, end:index});
      else pages.get(page).end = index;
    }
    if (!pages.size) return;
    status.textContent = 'Text selected. You can copy this passage.';
  }
  document.addEventListener('pointerup', event => { if (mode === 'text' && (event.target.closest('.page-container') || textSelectionPage)) setTimeout(captureTextSelection, 0); });
  document.addEventListener('keyup', event => { if (event.target.closest('.page-container')) captureTextSelection(); });
  let textSelectionPage = null;
  let textDrag = null;
  document.addEventListener('pointerdown', event => {
    textSelectionPage = mode === 'text' ? event.target.closest('.page-container') : null;
    const word = event.target.closest('.paper-word');
    textDrag = textSelectionPage && word && event.button === 0 ? {word, x:event.clientX, y:event.clientY,
      bounds:[...textSelectionPage.querySelectorAll('.paper-word')].map(word=>({word,rect:word.getBoundingClientRect()}))} : null;
    // Native selection across absolutely positioned words can jump to the
    // entire page/document. Build a bounded word range instead.
    if (textDrag) { event.preventDefault(); document.getSelection()?.removeAllRanges(); }
  });
  document.addEventListener('pointermove', event => {
    if (!textDrag || !textSelectionPage || !(event.buttons & 1)) return;
    if (Math.hypot(event.clientX-textDrag.x,event.clientY-textDrag.y) < 4) return;
    let nearest = null, distance = Infinity;
    if (textDrag.boundsDirty) {
      textDrag.bounds = textDrag.bounds.map(({word})=>({word,rect:word.getBoundingClientRect()}));
      textDrag.boundsDirty = false;
    }
    for (const {word,rect:r} of textDrag.bounds) {
      const dy = Math.max(r.top-event.clientY,0,event.clientY-r.bottom);
      const dx = Math.max(r.left-event.clientX,0,event.clientX-r.right);
      const score = dy*10000+dx;
      if (score < distance) { distance=score; nearest=word; }
    }
    if (!nearest) return;
    const forward = +nearest.dataset.wordIndex >= +textDrag.word.dataset.wordIndex;
    const range = document.createRange();
    range.setStart((forward?textDrag.word:nearest).firstChild,0);
    const last = forward?nearest:textDrag.word;
    range.setEnd(last.firstChild,last.firstChild.length);
    const selection = document.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    event.preventDefault();
  });
  document.addEventListener('pointerup', () => { textDrag=null; });
  document.addEventListener('pointercancel', () => { textDrag=null; });
  // Reuse word geometry during a drag, but invalidate it when either scroll
  // container moves. Stale viewport coordinates otherwise select the wrong line.
  document.addEventListener('scroll', () => { if (textDrag) textDrag.boundsDirty=true; }, true);
  document.addEventListener('selectionchange', () => {
    const selection = document.getSelection();
    const paperSelection = selection && [selection.anchorNode, selection.focusNode].some(
      node => (node?.nodeType === 1 ? node : node?.parentElement)?.closest('.page-container'));
    if (mode !== 'text' && paperSelection && !selection.isCollapsed) {
      selection.removeAllRanges(); return;
    }
    if (!textSelectionPage || !selection || selection.isCollapsed) return;
    if (!textSelectionPage.contains(selection.anchorNode) || !textSelectionPage.contains(selection.focusNode)) {
      selection.removeAllRanges();
      status.textContent = 'Select text within one page. Use separate selections for another page.';
    }
  });
  document.addEventListener('keydown', event => {
    const paperFocused = event.target.closest('.paper') || textSelectionPage;
    const editable = event.target.closest('input,textarea,[contenteditable="true"]');
    if (paperFocused && !editable && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a') {
      event.preventDefault(); document.getSelection()?.removeAllRanges();
      status.textContent = 'Select individual passages here. Download the paper to select its complete text.';
    }
  });
  function setMode(value) {
    if (mode === value) return;
    suppressPaperClick = false;
    textDrag = null; textSelectionPage = null;
    mode = value; body.classList.toggle('text-selection-off', value !== 'text');
    selectText?.classList.toggle('active', value === 'text');
    selectText?.setAttribute('aria-pressed',String(value === 'text'));
    document.getSelection()?.removeAllRanges();
    status.textContent = '';
  }
  selectText?.addEventListener('click', () => setMode(mode === 'text' ? 'none' : 'text'));
  setMode('none');
  async function awaitSuccessor(url, message, attemptId) {
    try {
      const response=await fetch(url,{credentials:'same-origin'});if(!response.ok)throw new Error('Report update status is unavailable.');const data=await response.json();
      if(data.successor_available){window.location.assign('/report/'+encodeURIComponent(data.latest_report_id)+window.location.search);return;}
      const failure=data.last_reanalysis_failure;
      // A new search is watched for its own attempt only; the upload wording belongs to uploads.
      if(failure&&(attemptId===undefined||failure.attempt_id===attemptId)){message.textContent=attemptId===undefined?'The source was stored, but the report could not be updated. Retry the upload.':'The new search could not update the report.';return;}
      setTimeout(()=>awaitSuccessor(url,message,attemptId),1500);
    }catch(error){message.textContent=error.message;}
  }
  document.addEventListener('click', event => {
    const choose=event.target.closest('[data-choose-source]');
    if(!choose)return;
    const form=choose.closest('form');
    form.querySelector('input[type=file]').click();
  });
  document.addEventListener('change', event => {if(event.target.matches('.source-upload input[type=file]') && event.target.files.length)event.target.closest('form').requestSubmit();});
  document.addEventListener('submit', async event => {
    const form=event.target;if(!form.classList.contains('source-upload'))return;event.preventDefault();
    const message=form.querySelector('.upload-status'),button=form.querySelector('button'),input=form.querySelector('input[type=file]');
    button.disabled=true;message.textContent='Uploading and checking source…';
    try{const response=await fetch(form.action,{method:'POST',body:new FormData(form),credentials:'same-origin'}),data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Source upload could not be completed.');message.textContent=data.message;input.disabled=true;if(data.status_url)awaitSuccessor(data.status_url,message);}
    catch(error){message.textContent=error.message;button.disabled=false;input.value='';}
  });
  document.addEventListener('submit', async event => {
    const form=event.target;if(!form.classList.contains('search-again'))return;event.preventDefault();
    const message=form.querySelector('.upload-status'),button=form.querySelector('button');
    // Owner-approved wording (2026-09-29).
    button.disabled=true;form.setAttribute('aria-busy','true');message.textContent='Searching again…';
    try{const response=await fetch(form.action,{method:'POST',credentials:'same-origin'}),data=await response.json().catch(()=>({}));
      if(response.ok&&data.status_url&&data.attempt_id){awaitSuccessor(data.status_url,message,String(data.attempt_id));return;}
      if(response.status===410){form.remove();return;}   // marks released since the page opened
      if(response.status===429){message.textContent='This reference was already searched again today. Try again tomorrow.';}
      else if(response.status===409){message.textContent='The report is already being updated.';button.disabled=false;}
      else{message.textContent='';button.disabled=false;}}
    catch(error){message.textContent='';button.disabled=false;}
    form.removeAttribute('aria-busy');
  });
  // view-state:start
  // Three layouts. Paper slides the window away and lets the paper take the
  // width; Sources and Judgment bring it back. The reading position is
  // re-anchored so the same passage stays at the top. The key never changes.
  // `layout` is declared with the splitter below; it is only read after load.
  const side = document.querySelector('.side-pane');
  const layoutButtons = {paper: document.getElementById('paper-layout'), sources: document.getElementById('sources-layout'),
                         judgment: document.getElementById('judgment-layout')};
  let view = 'sources', viewToken = 0, savedSelection = null;
  function reducedMotion() { return !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches; }
  function stacked() { return !!window.matchMedia?.('(max-width: 760px)').matches; }
  function paperAnchor() {
    const viewport = document.querySelector('.paper-viewport');
    if (!viewport) return null;
    const frame = viewport.getBoundingClientRect();
    const pages = [...viewport.querySelectorAll('.paper-page')];
    const page = pages.find(el => el.getBoundingClientRect().bottom > frame.top + 1) || pages[0];
    if (!page) return null;
    const box = page.getBoundingClientRect();
    const spare = viewport.scrollWidth - viewport.clientWidth;
    return {page, fraction: (frame.top - box.top) / (box.height || 1), left: spare > 0 ? viewport.scrollLeft / spare : 0};
  }
  function restoreAnchor(anchor) {
    const viewport = document.querySelector('.paper-viewport');
    if (!anchor || !viewport) return;
    const frame = viewport.getBoundingClientRect(), box = anchor.page.getBoundingClientRect();
    viewport.scrollTop += box.top + anchor.fraction * box.height - frame.top;
    viewport.scrollLeft = anchor.left * Math.max(0, viewport.scrollWidth - viewport.clientWidth);
  }
  function applyLayers(next) {
    const sources = next === 'sources';
    ['evidence','relevance','reference','practice'].forEach(name => body.classList.toggle('focus-'+name, sources));
    body.classList.toggle('hide-evidence', !sources);
    body.classList.toggle('hide-reference-practice', !sources);
    ['paper','sources','judgment'].forEach(name => {
      body.classList.toggle('layout-'+name, next === name);
      layoutButtons[name]?.setAttribute('aria-pressed', String(next === name));
    });
    document.querySelectorAll('[data-report-step]').forEach(button => { button.disabled = next === 'paper'; });
  }
  function setView(next, {then} = {}) {
    const token = ++viewToken, showWindow = next !== 'paper';
    if (next === view) { applyLayers(next); if (then) then(); return; }
    const wasHidden = view === 'paper';
    view = next;
    const anchor = paperAnchor();
    const instant = reducedMotion() || stacked() || !side;
    if (!showWindow) {
      savedSelection = openedTemplate ? {id: openedTemplate, member: Number(panel.dataset.memberIndex || 0)} : null;
      clearSelection(); applyLayers(next);
      if (side) side.inert = true;
      let done = false;
      const collapse = () => {
        if (done || token !== viewToken) return;
        done = true; side?.removeEventListener('transitionend', onEnd);
        layout.classList.add('panel-collapsed'); if (side) side.hidden = true;
        layout.classList.remove('panel-sliding'); restoreAnchor(anchor);
        if (then) then();
      };
      const onEnd = event => { if (event.target === side && event.propertyName === 'transform') collapse(); };
      if (instant) { collapse(); return; }
      side.addEventListener('transitionend', onEnd);
      layout.classList.add('panel-sliding');
      setTimeout(collapse, 350);
      return;
    }
    applyLayers(next);
    if (!wasHidden) { if (then) then(); return; }  // Sources <-> Judgment: the window stays.
    if (side) { side.hidden = false; side.inert = false; }
    layout.classList.remove('panel-collapsed');
    if (!instant) layout.classList.add('panel-sliding');
    restoreAnchor(anchor);
    const reveal = () => {
      if (token !== viewToken) return;
      layout.classList.remove('panel-sliding');
      if (then) then();
      else if (savedSelection && view === 'sources') {
        const target = paperTargetFor(savedSelection.id);
        if (target) { chooseCitation(target); showMember(savedSelection.member); }
        else openTemplate(savedSelection.id);
      }
      savedSelection = null;
    };
    if (instant) reveal(); else requestAnimationFrame(() => requestAnimationFrame(reveal));
  }
  layoutButtons.paper?.addEventListener('click', () => setView('paper'));
  applyLayers('sources');
  // view-state:end
  // Navigation shared by the arrows, the summary numbers and window links.
  function paperTargetFor(id) {
    const selectors = ['.citation-overlay:not(.member-target):not(.missing-reference-target)', '.reference-entry-overlay',
                       '.patchwriting-overlay', '.reference-practice-overlay', '.citation-overlay', '.paper-badge'];
    for (const selector of selectors) {
      const el = document.querySelector('.page-container ' + selector + '[data-panel-template="' + id + '"]');
      if (el) return el;
    }
    return null;
  }
  function orderedTargets() {
    const seen = new Set();
    // A patchwriting highlight inside a citation is reached with its citation;
    // one outside any citation is its own step, like the reference entry.
    return [...document.querySelectorAll('.page-container .citation-overlay[data-panel-template],.page-container .reference-practice-overlay[data-panel-template],.page-container .reference-entry-overlay[data-panel-template],.page-container .patchwriting-overlay[data-panel-template]:not([data-panel-template^="citation-panel-"])')]
      .sort((a, b) => {
        const page = Number(a.closest('.page-container').dataset.pageIndex) - Number(b.closest('.page-container').dataset.pageIndex);
        return page || a.getBoundingClientRect().top - b.getBoundingClientRect().top || a.getBoundingClientRect().left - b.getBoundingClientRect().left;
      }).filter(el => {
        const key = el.dataset.passage ? 'passage-' + el.dataset.passage : el.dataset.panelTemplate;
        if (seen.has(key) || !document.getElementById(key)) return false;
        seen.add(key); return true;
      });
  }
  // Scroll only the paper; scrolling the page would undock the workspace.
  function scrollPaperTo(el) {
    const viewport = document.querySelector('.paper-viewport');
    if (!viewport || !el) return;
    const box = el.getBoundingClientRect(), frame = viewport.getBoundingClientRect();
    viewport.scrollTo({top: Math.max(0, viewport.scrollTop + box.top - frame.top - (frame.height - box.height) / 2),
                       left: Math.max(0, viewport.scrollLeft + box.left - frame.left - (frame.width - box.width) / 2),
                       behavior: reducedMotion() ? 'auto' : 'smooth'});
  }
  function dockWorkspace() {
    const top = layout.getBoundingClientRect().top;
    if (Math.abs(top) > 1) window.scrollTo({top: window.scrollY + top, behavior: 'instant'});
  }
  function goToTarget(id, {dock = false, mark = null} = {}) {
    if (!document.getElementById(id)) return;
    const run = () => {
      if (dock) dockWorkspace();
      const target = (mark && document.getElementById(mark)) || paperTargetFor(id);
      if (target) { chooseCitation(target); scrollPaperTo(target); }
      else openTemplate(id);
    };
    if (view !== 'sources') setView('sources', {then: run}); else run();
  }
  function step(direction) {
    if (view !== 'sources') return;
    const targets = orderedTargets();
    if (!targets.length) return;
    const current = targets.includes(selectedTrigger) ? targets.indexOf(selectedTrigger)
      : targets.findIndex(el => el.dataset.panelTemplate === openedTemplate);
    // Step through a citation's propositions before leaving it.
    const here = propositionsIn(document.getElementById(openedTemplate));
    const at = here.indexOf(panel.dataset.proposition || '');
    if (current >= 0 && at >= 0 && at + direction >= 0 && at + direction < here.length) {
      chooseCitation(targets[current], here[at + direction]);
      return;
    }
    const index = current < 0 ? (direction > 0 ? 0 : targets.length - 1) : (current + direction + targets.length) % targets.length;
    const next = propositionsIn(document.getElementById(targets[index].dataset.panelTemplate));
    chooseCitation(targets[index], direction < 0 && next.length ? next[next.length - 1] : undefined);
    scrollPaperTo(targets[index]);
  }
  document.addEventListener('click', event => {
    const link = event.target.closest('[data-go-to]');
    if (!link) return;
    event.preventDefault();
    goToTarget(link.dataset.goTo, {dock: !!link.closest('.summary'), mark: link.dataset.goToMark});
  });
  let zoom=100, zoomFrame=null;const paperPages=document.querySelector('.paper-pages');
  function setZoom(value){
    textDrag = null; textSelectionPage = null; document.getSelection()?.removeAllRanges();
    zoom=Math.max(70,Math.min(200,value));
    document.getElementById('zoom-value').textContent=zoom+'%';
    if (zoomFrame !== null) return;
    zoomFrame=requestAnimationFrame(()=>{paperPages?.style.setProperty('--paper-zoom',zoom+'%');zoomFrame=null;});
  }
  document.getElementById('zoom-out').addEventListener('click',()=>setZoom(zoom-10));document.getElementById('zoom-in').addEventListener('click',()=>setZoom(zoom+10));
  const layout=document.getElementById('report-layout'),splitter=document.getElementById('report-splitter');let dragging=false;
  function share(value){value=Math.max(30,Math.min(80,value));layout.style.setProperty('--paper-share',value+'%');splitter.setAttribute('aria-valuenow',Math.round(value));}
  function resize(x){const box=layout.getBoundingClientRect();share((x-box.left)/box.width*100);}
  splitter.addEventListener('pointerdown',event=>{dragging=true;splitter.setPointerCapture(event.pointerId);resize(event.clientX);});splitter.addEventListener('pointermove',event=>{if(dragging)resize(event.clientX);});splitter.addEventListener('pointerup',event=>{dragging=false;splitter.releasePointerCapture(event.pointerId);});splitter.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();share(parseFloat(getComputedStyle(layout).getPropertyValue('--paper-share'))+(event.key==='ArrowRight'?2:-2));});
  // report_judgment.js opens a citation's window from its claim underline.
  function openCitation(number, proposition) {
    const id = 'citation-panel-' + number, el = paperTargetFor(id);
    if (el) chooseCitation(el, proposition); else openTemplate(id, proposition);
  }
  window.sourceFidelityLayout = {setView, clearSelection, openCitation, get view() { return view; },
    get selectedCitation() { const m = /^citation-panel-(\d+)$/.exec(openedTemplate || ''); return m ? Number(m[1]) : null; }};
})();
