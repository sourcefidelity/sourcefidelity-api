(() => {
  'use strict';
  const body = document.body;
  const panel = document.getElementById('evidence-panel');
  const status = document.getElementById('annotation-status');
  const comment = document.getElementById('add-comment');
  const highlight = document.getElementById('add-highlight');
  const undo = document.getElementById('undo-annotation');
  const redo = document.getElementById('redo-annotation');
  const pen = document.getElementById('pen-tool');
  const selectText = document.getElementById('select-text');
  const selectRegion = document.getElementById('select-region');
  let selectedTrigger = null, selectedAnchor = null, busy = false, mode = 'text';
  const undoStack = [], redoStack = [];
  const svgNS = 'http://www.w3.org/2000/svg';
  function sync() {
    comment.disabled = highlight.disabled = busy || !body.dataset.annotationCreate || !(selectedAnchor || selectedTrigger?.dataset.anchorId);
    undo.disabled = busy || !undoStack.length;
    redo.disabled = busy || !redoStack.length;
  }
  function push(command) { undoStack.push(command); redoStack.length = 0; sync(); }
  async function travel(from, to, operation) {
    const command = from[from.length - 1];
    if (!command || busy) return;
    busy = true; sync();
    try { await command[operation](); from.pop(); to.push(command); }
    catch (error) { status.textContent = error.message; }
    finally { busy = false; sync(); }
  }
  undo.addEventListener('click', () => travel(undoStack, redoStack, 'undo'));
  redo.addEventListener('click', () => travel(redoStack, undoStack, 'redo'));
  function clearSelection() {
    document.querySelectorAll('[data-panel-template].selected').forEach(el => { el.classList.remove('selected'); el.setAttribute('aria-pressed', 'false'); });
    document.querySelectorAll('.region-selection').forEach(el => el.remove());
  }
  function chooseCitation(trigger) {
    const template = document.getElementById(trigger.dataset.panelTemplate);
    if (!template) return;
    clearSelection(); selectedTrigger = trigger; selectedAnchor = null;
    trigger.classList.add('selected'); trigger.setAttribute('aria-pressed', 'true');
    panel.replaceChildren(template.content.cloneNode(true));
    const annotations = document.getElementById(trigger.dataset.annotationTemplate || '');
    if (annotations) panel.appendChild(annotations.content.cloneNode(true));
    panel.scrollTop = 0; sync();
  }
  function chooseAnchor(anchor, message) {
    selectedAnchor = anchor; selectedTrigger = null;
    const heading = document.createElement('h2'), text = document.createElement('p');
    heading.textContent = 'Selected paper text'; text.textContent = message;
    panel.replaceChildren(heading, text); panel.scrollTop = 0;
    status.textContent = body.dataset.annotationCreate ? 'Selection ready. Choose Comment or Highlight.' : 'Switch to Instructor view to add a comment or highlight.';
    sync();
  }
  let suppressPaperClick = false;
  function citationAt(event) {
    return event.target.closest('[data-panel-template]') || document.elementsFromPoint(event.clientX, event.clientY)
      .map(el => el.closest('[data-panel-template]')).find(Boolean);
  }
  document.addEventListener('click', event => {
    if (!event.target.closest('.page-container,[data-panel-template]')) return;
    if (suppressPaperClick) { suppressPaperClick = false; return; }
    if (mode === 'pen' || document.getSelection()?.toString().trim()) return;
    const trigger = citationAt(event);
    if (trigger) { event.preventDefault(); chooseCitation(trigger); }
  });
  document.addEventListener('keydown', event => {
    const trigger = event.target.closest('[data-panel-template]');
    if (trigger && ['Enter', ' '].includes(event.key)) { event.preventDefault(); chooseCitation(trigger); }
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
    const words = [];
    for (let page = firstPage; page <= lastPage; page++) {
      const container = document.querySelector('.page-container[data-page-index="'+page+'"]');
      if (container) words.push(...Array.from(container.querySelectorAll('.paper-word')).filter(word => range.intersectsNode(word)));
    }
    const pages = new Map();
    for (const word of words) {
      const page = Number(word.closest('.page-container').dataset.pageIndex), index = Number(word.dataset.wordIndex);
      if (!pages.has(page)) pages.set(page, {page_index:page, start:index, end:index});
      else pages.get(page).end = index;
    }
    if (!pages.size) return;
    clearSelection();
    chooseAnchor({anchor_kind:'text_selection', selected_words:Array.from(pages.values())}, 'Choose Comment or Highlight. The selection is anchored to the words and lines of the original PDF.');
  }
  document.addEventListener('pointerup', event => { if (mode === 'text' && event.target.closest('.page-container')) setTimeout(captureTextSelection, 0); });
  document.addEventListener('keyup', event => { if (event.target.closest('.page-container')) captureTextSelection(); });
  function setMode(value) {
    if (mode === value) return;
    suppressPaperClick = false;
    mode = value; body.classList.toggle('pen-active', value === 'pen'); body.classList.toggle('text-selection-off', value !== 'text');
    selectText.classList.toggle('active', value === 'text'); selectRegion.classList.toggle('active', value === 'region'); pen.classList.toggle('active', value === 'pen');
    document.getSelection()?.removeAllRanges();
    status.textContent = value === 'pen' ? 'Pen marks are session drafts. Undo and Redo apply to these marks.' : value === 'region' ? 'Drag to select an area, then choose Comment or Highlight.' : 'Select words or lines, then choose Comment or Highlight.';
  }
  selectText.addEventListener('click', () => setMode('text'));
  selectRegion.addEventListener('click', () => setMode('region'));
  pen.addEventListener('click', () => setMode(mode === 'pen' ? 'text' : 'pen'));
  document.querySelectorAll('.page-surface').forEach(svg => {
    let drawing = null, points = [], start = null, downTrigger = null, clientStart = null;
    function point(event) {
      const p = svg.createSVGPoint(); p.x = event.clientX; p.y = event.clientY;
      const local = p.matrixTransform(svg.getScreenCTM().inverse()), bounds = svg.viewBox.baseVal;
      return {x:Math.max(0, Math.min(bounds.width, local.x)), y:Math.max(0, Math.min(bounds.height, local.y))};
    }
    svg.addEventListener('pointerdown', event => {
      if (mode === 'text' || event.button !== 0 || (mode === 'region' && !body.dataset.annotationCreate)) return;
      downTrigger = mode === 'region' ? citationAt(event) : null;
      clientStart = {x:event.clientX,y:event.clientY};
      suppressPaperClick = false;
      event.preventDefault(); start = point(event); points = [start.x + ',' + start.y];
      drawing = document.createElementNS(svgNS, mode === 'pen' ? 'polyline' : 'rect');
      drawing.setAttribute('class', mode === 'pen' ? 'pen-stroke' : 'region-selection');
      if (mode === 'region') { drawing.setAttribute('x', start.x); drawing.setAttribute('y', start.y); }
      else drawing.setAttribute('points', points.join(' '));
      svg.appendChild(drawing); svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener('pointermove', event => {
      if (!drawing) return;
      const p = point(event);
      if (drawing.tagName === 'polyline') { points.push(p.x + ',' + p.y); drawing.setAttribute('points', points.join(' ')); }
      else { drawing.setAttribute('x', Math.min(start.x,p.x)); drawing.setAttribute('y', Math.min(start.y,p.y)); drawing.setAttribute('width', Math.abs(start.x-p.x)); drawing.setAttribute('height', Math.abs(start.y-p.y)); }
    });
    function finish(event, cancelled) {
      if (!drawing) return;
      const mark = drawing; drawing = null;
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
      if (cancelled) { mark.remove(); return; }
      if (mark.tagName === 'polyline') { push({undo:() => mark.remove(), redo:() => svg.appendChild(mark)}); return; }
      const x=+mark.getAttribute('x'), y=+mark.getAttribute('y'), width=+mark.getAttribute('width'), height=+mark.getAttribute('height');
      const clicked = Math.hypot(event.clientX-clientStart.x,event.clientY-clientStart.y) < 4;
      if (clicked) { mark.remove(); if (downTrigger) chooseCitation(downTrigger); suppressPaperClick=true; return; }
      suppressPaperClick = true;
      if (width < 2 || height < 2) { mark.remove(); return; }
      mark.remove(); clearSelection(); svg.appendChild(mark);
      chooseAnchor({anchor_version:'page-region-anchor-v1',anchor_kind:'page_region',localization_level:'exact_rectangle',rectangles:[{page_index:+svg.dataset.pageIndex,x0:x,y0:y,x1:x+width,y1:y+height}]}, 'Choose Comment or Highlight to save this paper area.');
    }
    svg.addEventListener('pointerup', event => finish(event, false));
    svg.addEventListener('pointercancel', event => finish(event, true));
  });
  async function request(url, method, payload) {
    const response = await fetch(url, {method,credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The annotation could not be saved.');
    return data;
  }
  async function refreshAnnotations(annotation) {
    // Refresh only the authored overlays/templates, preserving pen drafts,
    // scroll position, and the undo/redo command stack.
    const response = await fetch(window.location.href, {credentials:'same-origin'});
    if (!response.ok) throw new Error('Annotation saved, but display refresh failed. Reload to see it.');
    const fresh = new DOMParser().parseFromString(await response.text(), 'text/html');
    document.querySelectorAll('.paper-annotation-overlay').forEach(el => el.remove());
    fresh.querySelectorAll('.page-surface').forEach(newSvg => {
      const svg = document.querySelector('.page-surface[data-page-index="'+newSvg.dataset.pageIndex+'"]');
      if (!svg) return;
      newSvg.querySelectorAll('.paper-annotation-overlay').forEach(el => svg.appendChild(el.cloneNode(true)));
      newSvg.querySelectorAll('.citation-overlay').forEach(el => {
        const old = svg.querySelector('.citation-overlay[data-panel-template="'+el.dataset.panelTemplate+'"]');
        if (old) old.replaceWith(el.cloneNode(true));
      });
    });
    document.querySelectorAll('template[id^="annotation-panel-"]').forEach(el => el.remove());
    fresh.querySelectorAll('template[id^="annotation-panel-"]').forEach(el => body.appendChild(el.cloneNode(true)));
    const trigger = annotation && document.querySelector('[data-anchor-id="'+annotation.anchor.anchor_id+'"]');
    if (trigger && annotation.state !== 'deleted') chooseCitation(trigger);
    else { panel.replaceChildren(); selectedTrigger=null; selectedAnchor=null; }
  }
  async function createAnnotation(type, content) {
    if (busy || !body.dataset.annotationCreate || !(selectedTrigger || selectedAnchor)) return;
    const payload = {annotation_type:type,content:content || null,visibility:'private'};
    if (selectedAnchor) payload.anchor = selectedAnchor; else payload.anchor_id=selectedTrigger.dataset.anchorId;
    busy=true; sync(); status.textContent='Saving annotation…';
    try {
      let record = await request(body.dataset.annotationCreate, 'POST', payload);
      const revisionUrl = body.dataset.annotationRevisionTemplate.replace('{annotation_id}', record.annotation_id);
      async function revise(state) { record = await request(revisionUrl,'PATCH',{expected_revision:record.revision,state}); await refreshAnnotations(record); }
      push({undo:() => revise('deleted'),redo:() => revise('active')});
      document.getSelection()?.removeAllRanges();
      await refreshAnnotations(record); status.textContent='Private annotation saved. Release it to include it in the Student view and PDF.';
    } catch(error) { status.textContent=error.message; }
    finally {busy=false;sync();}
  }
  highlight.addEventListener('click', () => createAnnotation('highlight', null));
  comment.addEventListener('click', () => {
    if (comment.disabled) return;
    panel.querySelector('.annotation-editor')?.remove();
    const form=document.createElement('form'), label=document.createElement('label'), textarea=document.createElement('textarea'), save=document.createElement('button');
    form.className='annotation-editor'; label.textContent='Comment on selected paper text or area'; textarea.setAttribute('aria-label','Comment'); textarea.required=true; textarea.maxLength=4000; save.type='submit';save.textContent='Save private comment';
    label.appendChild(textarea);form.append(label,save);panel.prepend(form);panel.scrollTop=0;textarea.focus();
    form.addEventListener('submit', event => {event.preventDefault();const value=textarea.value.trim();if(value)createAnnotation('comment',value);});
  });
  panel.addEventListener('click', async event => {
    const choose=event.target.closest('[data-choose-source]');
    if(choose){choose.closest('form').querySelector('input[type=file]').click();return;}
    const button=event.target.closest('[data-annotation-operation]');
    if(!button || busy)return;
    const note=button.closest('[data-annotation-id]'), payload={expected_revision:+note.dataset.revision,state:'active'}, operation=button.dataset.annotationOperation;
    if(operation==='edit'){const value=window.prompt('Edit comment',note.dataset.content||'');if(!value)return;payload.content=value;}
    if(operation==='visibility')payload.visibility=note.dataset.visibility==='released'?'private':'released';
    if(operation==='delete')payload.state='deleted';
    busy=true;sync();
    try{const record=await request(body.dataset.annotationRevisionTemplate.replace('{annotation_id}',note.dataset.annotationId),'PATCH',payload);await refreshAnnotations(record);undoStack.length=redoStack.length=0;status.textContent='Annotation updated.';}
    catch(error){status.textContent=error.message;}finally{busy=false;sync();}
  });
  async function awaitSuccessor(url, message) {
    try {
      const response=await fetch(url,{credentials:'same-origin'});if(!response.ok)throw new Error('Report update status is unavailable.');const data=await response.json();
      if(data.successor_available){window.location.assign('/report/'+encodeURIComponent(data.latest_report_id)+window.location.search);return;}
      if(data.last_reanalysis_failure){message.textContent='The source was stored, but the report could not be updated. Retry the upload.';return;}
      setTimeout(()=>awaitSuccessor(url,message),1500);
    }catch(error){message.textContent=error.message;}
  }
  panel.addEventListener('change', event => {if(event.target.matches('.source-upload input[type=file]') && event.target.files.length)event.target.closest('form').requestSubmit();});
  panel.addEventListener('submit', async event => {
    const form=event.target;if(!form.classList.contains('source-upload'))return;event.preventDefault();
    const message=form.querySelector('.upload-status'),button=form.querySelector('button'),input=form.querySelector('input');
    button.disabled=true;message.textContent='Uploading and checking source…';
    try{const response=await fetch(form.action,{method:'POST',body:new FormData(form),credentials:'same-origin'}),data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Source upload could not be completed.');message.textContent=data.message;input.disabled=true;if(data.status_url)awaitSuccessor(data.status_url,message);}
    catch(error){message.textContent=error.message;button.disabled=false;input.value='';}
  });
  const evidence=document.getElementById('layer-evidence'),reference=document.getElementById('layer-reference'),key=document.getElementById('active-key');
  function layers(){body.classList.toggle('hide-evidence',!evidence.checked);body.classList.toggle('hide-reference-practice',!reference.checked);body.classList.toggle('paper-only',!evidence.checked&&!reference.checked);key.querySelectorAll('[data-key="evidence"]').forEach(el=>el.hidden=!evidence.checked);key.querySelector('[data-key="reference"]').hidden=!reference.checked;}
  document.getElementById('paper-only').addEventListener('click',()=>{evidence.checked=reference.checked=false;layers();});evidence.addEventListener('change',layers);reference.addEventListener('change',layers);layers();
  let zoom=100;const paperPages=document.querySelector('.paper-pages');
  function setZoom(value){zoom=Math.max(70,Math.min(200,value));paperPages?.style.setProperty('--paper-zoom',zoom+'%');document.getElementById('zoom-value').textContent=zoom+'%';}
  document.getElementById('zoom-out').addEventListener('click',()=>setZoom(zoom-10));document.getElementById('zoom-in').addEventListener('click',()=>setZoom(zoom+10));
  const layout=document.getElementById('report-layout'),splitter=document.getElementById('report-splitter');let dragging=false;
  function share(value){value=Math.max(30,Math.min(80,value));layout.style.setProperty('--paper-share',value+'%');splitter.setAttribute('aria-valuenow',Math.round(value));}
  function resize(x){const box=layout.getBoundingClientRect();share((x-box.left)/box.width*100);}
  splitter.addEventListener('pointerdown',event=>{dragging=true;splitter.setPointerCapture(event.pointerId);resize(event.clientX);});splitter.addEventListener('pointermove',event=>{if(dragging)resize(event.clientX);});splitter.addEventListener('pointerup',event=>{dragging=false;splitter.releasePointerCapture(event.pointerId);});splitter.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();share(parseFloat(getComputedStyle(layout).getPropertyValue('--paper-share'))+(event.key==='ArrowRight'?2:-2));});
  sync();
})();
