"""Small, escaped, single-source Personal review interface."""
from html import escape


OPTIONS = (
    ("same_edition_later_printing", "Same cited edition, later printing", "The shown pages identify the cited edition and a later printing, without identifying a new edition. Identify both evidence pages and record the publication and printing dates in your explanation. A later printing alone is not a reference-year error."),
    ("confirmed", "Confirmed alternate edition", "The shown pages establish the same work and an identifiable different edition or reissue. Identify both evidence pages and explain the relationship."),
    ("uncertain", "Uncertain", "The shown pages are unreadable, incomplete, ambiguous or insufficient to establish the relationship."),
    ("rejected", "Relationship not established", "The shown pages give affirmative reasons to reject the proposed same-work alternate-edition relationship. Explain those reasons."),
)


def shell(body):
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personal edition review</title><style>
body{{font:16px/1.5 system-ui,sans-serif;color:#18283b;background:#f3f5f7;margin:0}}
main{{max-width:960px;margin:auto;padding:24px}}section,fieldset{{background:white;padding:18px;border:1px solid #ccd4df;border-radius:8px;margin:16px 0;min-width:0}}
h1{{font-size:1.65rem}}h2{{font-size:1.15rem}}label{{display:block;margin:12px 0}}textarea,input[type=text],input[type=number]{{box-sizing:border-box;max-width:100%;width:100%;padding:10px;font:inherit}}
textarea{{min-height:120px}}button{{font:inherit;padding:10px 16px;background:#214f81;color:white;border:0;border-radius:4px}}
button:disabled{{opacity:.5}}table{{border-collapse:collapse;width:100%;table-layout:fixed}}th,td{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #ccd4df;overflow-wrap:anywhere}}th:first-child{{width:29%}}
img{{display:block;max-width:100%;height:auto;border:1px solid #ccd4df}}pre,.reference{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#214f81}}.notice{{border-left:4px solid #bb751c;padding:10px;background:#fff6e7}}
@media(max-width:500px){{main{{padding:12px}}section,fieldset{{padding:12px}}th,td{{padding:6px}}}}
</style><script src="/edition-reviews/interface.js" defer></script></head><body><main>{body}</main></body></html>'''


def new_review_html(representation_id=""):
    return shell(f'''<h1>Personal edition review</h1>
<p>Compare one source copy with a reference. This records a bibliographic relationship only; it does not admit the source or change a report.</p>
<form method="post" action="/edition-reviews"><section>
<label>Source representation ID<input name="representation_id" type="text" required maxlength="36" value="{escape(representation_id, quote=True)}"></label>
<label>Reference to compare<textarea name="reference_text" required maxlength="4000"></textarea></label>
<p>This reference is supplied by you. It is not automatically linked to a student's paper.</p>
<button>Prepare review</button></section></form>''')


def review_html(snapshot, decision=None):
    payload = snapshot.payload
    if payload.get("version") == "personal-edition-review-v3":
        return simple_review_html(snapshot, decision)
    options = OPTIONS if payload.get("version") == "personal-edition-review-v2" else OPTIONS[1:]
    review_id = str(snapshot.id)
    pages = payload["page_manifest"]["pages"]
    total = payload["page_manifest"]["total_pages"]
    body = f'''<h1>Review this copy's edition</h1>
<p class="notice">Review-only access. This copy is not being approved for citation checking.</p>
<section><h2>Reference supplied by you</h2><p class="reference">{escape(payload['reference_text'])}</p>
<p>PDF pages 1–{len(pages)} of {total}. This is a bounded front-matter review, not a complete-source review. If these pages do not settle the relationship, choose Uncertain.</p>
<p>Recorded source status: identity {escape(payload['source_status']['identity'])}; completeness {escape(payload['source_status']['completeness'])}; admission {escape(payload['source_status']['admission'])}.</p></section>'''
    for page in pages:
        number = page["page_index"] + 1
        body += f'''<section><h2>PDF page {number}</h2><p><a href="/edition-reviews/{review_id}/pages/{number}" target="_blank" rel="noopener noreferrer">Open page {number} at full size</a></p><img src="/edition-reviews/{review_id}/pages/{number}" alt="Source PDF page {number}" data-review-page></section>'''
    body += '<section><h2>Decision guide</h2><table><thead><tr><th>Option</th><th>Select when</th></tr></thead><tbody>'
    for value, label, meaning in options:
        body += f'<tr><td>{escape(label)}</td><td>{escape(meaning)}</td></tr>'
    body += '</tbody></table><p>Confirmation does not establish unchanged text, quotation or paraphrase usability, or matching pagination.</p></section>'
    if decision is not None:
        saved = decision.payload["submission"]
        label = next(label for value, label, _ in OPTIONS if value == saved["decision"])
        body += f'''<section><h2>Saved decision: {escape(label)}</h2><pre>{escape(saved['notes'])}</pre>
<p>This decision is immutable. To reconsider, prepare a new review; this record will remain in the history while the source is retained.</p>
<a href="/edition-reviews/{review_id}/export">Download review record</a><br>
<a href="/edition-reviews/new?representation_id={snapshot.representation_id}">Prepare another review</a></section>'''
    else:
        body += f'''<form method="post" action="/edition-reviews/{review_id}/decision">
<input type="hidden" name="snapshot_sha256" value="{snapshot.snapshot_sha256}"><fieldset><legend>Your decision</legend>'''
        for value, label, _ in options:
            body += f'<label><input type="radio" name="decision" value="{value}" required> {escape(label)}</label>'
        body += f'''<div id="confirmation-pages"><p>For confirmation, identify the PDF pages establishing work identity and the actual edition relationship.</p>
<label>Work identity evidence: PDF page<input type="number" name="work_page" min="1" max="{len(pages)}" disabled></label>
<label>Edition relationship evidence: PDF page<input type="number" name="edition_page" min="1" max="{len(pages)}" disabled></label></div>
<label>Explanation<textarea name="notes" required maxlength="4000"></textarea></label>
<p>Describe the observed title/author and publication or reissue details, or explain what remains unclear.</p>
<label><input type="checkbox" name="acknowledged" value="true" required disabled> I reviewed the reference and all the displayed pages.</label>
<p id="page-status" role="status">Waiting for source pages to load.</p>
<button id="save-review" disabled>Save decision</button></fieldset></form>'''
    return shell(body)


def simple_review_html(snapshot, decision=None):
    p = snapshot.payload
    body = f'<h1>{escape(p["question"])}</h1><p class="reference">{escape(p["reference_text"])}</p>'
    body += '<p>A later printing of the same edition counts as a match. Translations and abridgments require separate retrieval.</p>'
    body += '<p>Compare these front-matter pages. Open any page to read it at full size.</p><div style="display:flex;flex-wrap:wrap;gap:12px">'
    for page in p['page_manifest']['pages']:
        n = page['page_index'] + 1
        url = f'/edition-reviews/{snapshot.id}/pages/{n}'
        body += f'<figure style="margin:0;max-width:100%"><a href="{url}" target="_blank" rel="noopener noreferrer"><img style="max-height:340px;width:auto" src="{url}" alt="PDF page {n}" data-review-page></a><figcaption>PDF page {n}</figcaption></figure>'
    body += '</div><table><thead><tr><th>Option</th><th>Select when</th></tr></thead><tbody>'
    for value, label, meaning in p['options']:
        body += f'<tr><td>{escape(label)}</td><td>{escape(meaning)}</td></tr>'
    body += '</tbody></table>'
    if decision:
        label = 'Yes' if decision.payload['answer'] == 'yes' else 'No'
        body += f'<h2>Saved answer: {label}</h2><p>This answer is recorded without changing a source or report.</p><a href="/edition-reviews/{snapshot.id}/export">Download review record</a>'
    elif p['separate_retrieval_required']:
        body += '<p>This copy has a translation or abridgment signal. It needs separate retrieval and cannot be approved here.</p>'
    else:
        body += f'<form method="post" action="/edition-reviews/{snapshot.id}/answer"><input type="hidden" name="snapshot_sha256" value="{snapshot.snapshot_sha256}"><p id="page-status" role="status">Loading pages…</p><button name="answer" value="yes" data-answer disabled>Yes</button> <button name="answer" value="no" data-answer disabled>No</button></form>'
    body += '<p>This confirms only the bibliographic match, not unchanged text or matching page numbers. No leaves the copy unverified.</p>'
    return shell(body)


INTERFACE_JS = '''"use strict";
const choices=document.querySelectorAll('input[name="decision"]');
function updateDecision(){const confirmed=['confirmed','same_edition_later_printing'].includes(document.querySelector('input[name="decision"]:checked')?.value);
document.querySelectorAll('#confirmation-pages input').forEach(input=>{input.disabled=!confirmed;input.required=confirmed;if(!confirmed)input.value='';});}
choices.forEach(input=>input.addEventListener('change',updateDecision));updateDecision();
const pages=[...document.querySelectorAll('[data-review-page]')];
function checkPages(){const ready=pages.length>0&&pages.every(img=>img.complete&&img.naturalWidth>0);
const ack=document.querySelector('input[name="acknowledged"]');const save=document.querySelector('#save-review');const status=document.querySelector('#page-status');
if(ack)ack.disabled=!ready;if(save)save.disabled=!ready;document.querySelectorAll('[data-answer]').forEach(b=>b.disabled=!ready);if(status)status.textContent=ready?'All review pages loaded.':'A page is still loading or unavailable. Reload before reviewing.';}
pages.forEach(img=>{img.addEventListener('load',checkPages);img.addEventListener('error',checkPages);});checkPages();
document.querySelectorAll('form').forEach(form=>form.addEventListener('submit',async event=>{
event.preventDefault();const button=event.submitter||form.querySelector('button');if(button.disabled)return;
const data=new FormData(form);if(button.name)data.append(button.name,button.value);
form.querySelectorAll('button').forEach(b=>b.disabled=true);let message=form.querySelector('[data-submit-status]');
if(!message){message=document.createElement('p');message.dataset.submitStatus='';message.setAttribute('role','status');form.append(message);}
message.textContent='Saving…';
try{const response=await fetch(form.action,{method:'POST',body:data,credentials:'same-origin'});
if(!response.ok)throw new Error('Review could not be saved. Check the inputs and current source access; your entries remain here.');
const destination=new URL(response.url);if(destination.origin!==location.origin||!response.redirected)throw new Error('Unexpected save response. Reload to check whether a decision was recorded.');
location.assign(destination.href);
}catch(error){message.textContent=error.message;form.querySelectorAll('button').forEach(b=>b.disabled=false);checkPages();}}));
'''
