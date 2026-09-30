"""Behaviour of the Paper/Report view state in report_interactions.js.

The block between ``// view-state:start`` and ``// view-state:end`` runs in a
small fake DOM. Node is used when ``SOURCEFIDELITY_TEST_NODE`` names a binary;
otherwise macOS JavaScriptCore (``osascript -l JavaScript``) runs the same
harness. The test skips only when neither runtime exists.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path('app/services/report_interactions.js').read_text()
BLOCK = SCRIPT[SCRIPT.index('  // view-state:start'):SCRIPT.index('  // view-state:end')]

HARNESS = r"""
(() => {
  function classList() {
    const s = new Set();
    return {add: c => s.add(c), remove: c => s.delete(c), contains: c => s.has(c),
            toggle(c, force) { const on = force === undefined ? !s.has(c) : force; if (on) s.add(c); else s.delete(c); }};
  }
  function element(id) {
    const listeners = {};
    return {id, classList: classList(), dataset: {}, attrs: {}, hidden: false, inert: false, disabled: false,
      setAttribute(k, v) { this.attrs[k] = String(v); },
      addEventListener(t, f) { (listeners[t] = listeners[t] || []).push(f); },
      removeEventListener(t, f) { listeners[t] = (listeners[t] || []).filter(x => x !== f); },
      fire(t, e) { (listeners[t] || []).slice().forEach(f => f(e)); }};
  }
  const body = element('body'), panel = element('panel'), sidePane = element('side'), layout = element('layout');
  const paperButton = element('paper-layout'), sourcesButton = element('sources-layout'), judgmentButton = element('judgment-layout');
  const steps = [element('prev'), element('next')];
  // Three pages whose height depends on whether the window is collapsed:
  // the paper grows to fill the width in Paper, as the owner decided.
  const frameTop = 100;
  const viewport = {scrollTop: 1500, scrollLeft: 0, scrollWidth: 800, clientWidth: 800,
    getBoundingClientRect() { return {top: frameTop}; },
    querySelectorAll() { return pages; }};
  const pageHeight = () => layout.classList.contains('panel-collapsed') ? 1500 : 1000;
  const pages = [0, 1, 2].map(i => ({getBoundingClientRect() {
    const h = pageHeight(), top = frameTop + i * h - viewport.scrollTop;
    return {top, bottom: top + h, height: h};
  }}));
  const document = {
    querySelector(selector) { return selector === '.side-pane' ? sidePane : selector === '.paper-viewport' ? viewport : null; },
    getElementById(id) { return {'paper-layout': paperButton, 'sources-layout': sourcesButton, 'judgment-layout': judgmentButton}[id]; },
    querySelectorAll(selector) { return selector === '[data-report-step]' ? steps : []; },
  };
  let reduce = false;
  const window = {matchMedia: query => ({matches: query.includes('reduced-motion') ? reduce : false})};
  const timers = [], frames = [];
  const setTimeout = f => { timers.push(f); };
  const requestAnimationFrame = f => { frames.push(f); };
  const flushFrames = () => { while (frames.length) frames.shift()(); };
  let openedTemplate = 'citation-panel-3';
  const log = [];
  const clearSelection = () => log.push('clear');
  const chooseCitation = target => log.push('choose:' + target.id);
  const showMember = index => log.push('member:' + index);
  const openTemplate = id => log.push('open:' + id);
  const paperTargetFor = id => ({id});
  panel.dataset.memberIndex = '2';
/*BLOCK*/
  const results = {};
  results.initial = {stepsEnabled: steps.every(b => !b.disabled), focus: body.classList.contains('focus-evidence'),
                     sources: body.classList.contains('layout-sources')};
  // Starting at scrollTop 1500 with 1000-point pages: halfway down page 2.
  setView('paper');
  results.sliding = {inert: side.inert, sliding: layout.classList.contains('panel-sliding'),
                     collapsed: layout.classList.contains('panel-collapsed'),
                     stepsDisabled: steps.every(b => b.disabled), paperOnly: body.classList.contains('layout-paper')};
  side.fire('transitionend', {target: side, propertyName: 'transform'});
  results.collapsed = {collapsed: layout.classList.contains('panel-collapsed'), hidden: side.hidden,
                       sliding: layout.classList.contains('panel-sliding'), scrollTop: viewport.scrollTop};
  timers.forEach(f => f());                             // the fallback timer must not re-run the collapse
  results.afterFallback = viewport.scrollTop;
  setView('sources');
  results.returning = {hidden: side.hidden, inert: side.inert, collapsed: layout.classList.contains('panel-collapsed'),
                       sliding: layout.classList.contains('panel-sliding'), scrollTop: viewport.scrollTop};
  flushFrames(); flushFrames();
  results.revealed = {sliding: layout.classList.contains('panel-sliding'), stepsEnabled: steps.every(b => !b.disabled),
                      log: log.slice()};
  log.length = 0; reduce = true;
  setView('paper');
  results.reduced = {collapsed: layout.classList.contains('panel-collapsed'), hidden: side.hidden};
  log.length = 0;
  setView('sources', {then: () => log.push('then')});
  results.thenOnly = log.slice();
  // Sources <-> Judgment keeps the window; only the layout class changes.
  reduce = false; log.length = 0;
  setView('judgment');
  results.judgment = {judgment: body.classList.contains('layout-judgment'), sources: body.classList.contains('layout-sources'),
                      hidden: side.hidden, focus: body.classList.contains('focus-evidence'),
                      pressed: judgmentButton.attrs['aria-pressed'], sourcesPressed: sourcesButton.attrs['aria-pressed']};
  return JSON.stringify(results);
})()
"""


def run_harness(source):
    node = os.environ.get('SOURCEFIDELITY_TEST_NODE')
    if node:
        completed = subprocess.run([node, '-e', 'console.log(' + source + ')'], capture_output=True, text=True, timeout=30)
    elif shutil.which('osascript'):
        completed = subprocess.run(['osascript', '-l', 'JavaScript', '-e', source], capture_output=True, text=True, timeout=30)
    else:
        pytest.skip('A JavaScript runtime (Node or macOS JavaScriptCore) is needed')
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip())


def test_paper_view_slides_the_window_away_and_sources_restores_it():
    result = run_harness(HARNESS.replace('/*BLOCK*/', BLOCK))
    assert result['initial'] == {'stepsEnabled': True, 'focus': True, 'sources': True}
    # Leaving Report: inert at once, arrows off, slide first, collapse after.
    assert result['sliding'] == {'inert': True, 'sliding': True, 'collapsed': False,
                                 'stepsDisabled': True, 'paperOnly': True}
    # After the slide the paper takes the width and the same passage stays
    # at the top: page 2 halfway down, now 1500 points tall.
    assert result['collapsed'] == {'collapsed': True, 'hidden': True, 'sliding': False, 'scrollTop': 2250}
    assert result['afterFallback'] == 2250
    # Returning: reflow once, re-anchor, then slide in and restore the window.
    assert result['returning'] == {'hidden': False, 'inert': False, 'collapsed': False, 'sliding': True, 'scrollTop': 1500}
    assert result['revealed']['sliding'] is False and result['revealed']['stepsEnabled'] is True
    assert result['revealed']['log'][-2:] == ['choose:citation-panel-3', 'member:2']
    # Reduced motion collapses immediately; a requested target replaces the restore.
    assert result['reduced'] == {'collapsed': True, 'hidden': True}
    assert result['thenOnly'] == ['then']
    assert result['judgment'] == {'judgment': True, 'sources': False, 'hidden': False, 'focus': False,
                                  'pressed': 'true', 'sourcesPressed': 'false'}
