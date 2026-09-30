"""Neutral display of saved link observations, never new network requests."""
from html import escape

from app.services.submitted_links import SubmittedLink, project_observations


OUTCOME_TEXT = {
    'response': 'The server responded. This alone does not establish source identity or full-text access.',
    'not_found': 'Page not found when checked. This does not establish that the work does not exist.',
    'removed': 'The server reported that the page was removed.',
    'authentication_required': 'Server authentication required.',
    'proxy_authentication_required': 'Proxy authentication required.',
    'access_refused': 'Access refused. The response does not establish why.',
    'rate_limited': 'The server limited requests. This is not a dead-link finding.',
    'server_failure': 'The server encountered an error.',
    'legal_restriction_reported': 'The server reported a legal restriction; its geographic scope is not established.',
    'timeout': 'The request timed out in this environment.',
    'dns_failure': 'The address could not be resolved in this environment.',
    'tls_failure': 'A secure connection could not be established.',
    'connection_failure': 'The connection failed in this environment.',
    'safety_refused': 'The application blocked the request for safety.',
    'size_limit': 'The response exceeded the permitted size.',
    'redirect_limit': 'The redirect limit was reached.',
    'operational_failure': 'The request could not be completed.',
}


def saved_link_view(references, aggregate):
    if 'submitted_link_observations' not in aggregate:
        return []  # Do not retrofit historical reports.
    grouped = {ref.reference_id: [] for ref in references}
    raw = aggregate.get('submitted_link_observations')
    if isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict) and row.get('reference_id') in grouped:
                grouped[row['reference_id']].append(row)
    return project_observations(references, [
        {'reference_id': key, 'submitted_link_observations': value}
        for key, value in grouped.items()])


def render_link_checks(values):
    if not values:
        return ''
    entries = []
    for value in values:
        try:
            row = SubmittedLink.model_validate(value)
        except (ValueError, TypeError):
            entries.append('<p>Historical link-check details are unavailable.</p>')
            continue
        label = 'Submitted DOI' if row.kind == 'doi' else 'Submitted URL'
        if row.address_extraction == 'possible_truncation':
            entries.append('<p>The extracted address may be incomplete where the URL wraps in the reference. '
                           'Compare the complete address in the original paper; this result applies only to the extracted address.</p>')
        if row.state != 'observed':
            text = ('Historical check status was not recorded or could not be validated.'
                if row.state == 'historical_unknown' else {
                    'uncited_reference': 'Not checked: this reference was not cited in the extracted citations.',
                    'resolution_not_run': 'Not checked: source resolution was not run for this reference.',
                    'not_visited_by_resolution': 'Not checked: source resolution did not visit this link.',
                    'authorized_reuse': 'Not checked: an authorized source copy was reused instead.',
                    'capability_disabled': 'Not checked: this retrieval route was not enabled.',
                    'library_locator_only': 'Not checked as source text: this is a library catalog or discovery link.',
                    'candidate_budget_exhausted': 'Not checked: the candidate-checking budget was exhausted.',
                    'unsupported_source_type': 'Not checked: automatic retrieval is not supported for this source type.',
                }.get(row.not_checked_reason, 'Not checked; no request was recorded.'))
            entries.append(f'<p><strong>{label}:</strong> {escape(text)}</p>')
            continue
        for request in row.requests:
            status = f' HTTP {request.http_status}.' if request.http_status else ''
            checked = request.started_at.isoformat(timespec='seconds')
            text = OUTCOME_TEXT[request.outcome]
            if request.safety_reason in {'invalid_connected_peer', 'non_public_connected_peer'}:
                text = 'The connection failed the application’s public-network safety check. The response body was not accepted.'
            page = {
                'pdf_route_required': 'The response was a PDF requiring separate acquisition and validation.',
                'cross_script_title_unresolved': 'The title comparison could not be resolved across writing systems.',
                'page_title_mismatch_unconfirmed': 'The page title did not match; this alone does not establish a different work.',
                'readable_text_unavailable': 'Usable article text could not be extracted from the response.',
                'source_kind_unconfirmed': 'The response did not establish the expected type of source.',
            }.get(request.page_observation, '')
            identity = {
                'not_assessed': 'Destination identity was not assessed.',
                'confirmed': 'The destination’s bibliographic identity was confirmed.',
                'bibliographic_conflict': 'The destination’s bibliographic details conflict with the reference.',
                'unconfirmed': 'The destination’s bibliographic identity remains unconfirmed.',
            }[request.destination_identity]
            origin = (f'<br>Last responding site: {escape(request.destination_origin)}'
                      if request.destination_origin else '')
            hops = ' → '.join(f'{hop.http_status} {hop.destination_origin or "destination withheld"}'
                              for hop in request.hops)
            redirects = f'<br>Response path: {escape(hops)}' if len(request.hops) > 1 else ''
            content = ('A validated copy was obtained from this destination. Available evidence is shown above.'
                if request.admitted_content == 'admitted' else
                'This check did not establish access to usable source text. Evidence may have been obtained by another route.')
            content = {
                'safety_rejected': 'The downloaded file did not pass file-safety checks and was not admitted for use as evidence.',
                'safety_unavailable': 'File-safety checking was unavailable, so the downloaded file was not admitted for use as evidence.',
                'completeness_uncertain': 'The source was downloaded, but its completeness is uncertain. It requires review before use as evidence.',
                'completeness_rejected': 'The downloaded copy was judged incomplete and was not admitted for use as evidence.',
            }.get(request.source_validation, content)
            entries.append(f'<p><strong>{label}:</strong> {escape(text)}{status}'
                f'<br><small>Checked {escape(checked)}; {request.elapsed_seconds:.2f}s.{origin}{redirects}</small>'
                f'<br>{escape(page)} {escape(identity)} {escape(content)}</p>')
        if row.observations_truncated:
            entries.append('<p>Further request details exceeded the recording limit.</p>')
    return '<details class="submitted-link-check"><summary>Submitted link check</summary>' + ''.join(entries) + '</details>'


def reference_link_findings(values, sources):
    """Only completed affirmative outcomes bound to the original supplied link."""
    from app.services.submitted_links import binding, address_may_be_truncated
    findings=homepage_address_findings(sources)
    for value in values or []:
        try:
            row=SubmittedLink.model_validate(value)
        except (ValueError,TypeError):
            continue
        source=sources.get(row.reference_id) or {}
        address=str(source.get(row.kind) or '')
        if not address and row.kind=='url':
            import re
            candidates={part for value in re.findall(r'(?:https?://|www\.)\S+',str(source.get('raw_reference') or ''))
                        for part in (value,value.rstrip('.,;')) if binding(part)==row.submitted_sha256}
            if len(candidates)==1:address=candidates.pop()
        if (not address or binding(address)!=row.submitted_sha256 or row.state!='observed'
                or row.address_extraction=='possible_truncation' or row.observations_truncated):
            continue
        # Older observations predate truncation disclosure. Preserve them, but
        # never promote a prefix-only HTTP failure into a complete-link flag.
        if row.kind=='url' and address_may_be_truncated(address,source.get('raw_reference','')):
            continue
        request=max(row.requests,key=lambda r:r.completed_at)
        missing=(request.outcome,request.http_status) in {('not_found',404),('removed',410)}
        conflict=(request.outcome=='response' and request.http_status is not None and 200<=request.http_status<300
                  and request.destination_identity=='bibliographic_conflict' and bool(request.identity_fields))
        homepage=(request.outcome=='response' and request.http_status is not None and 200<=request.http_status<300
                  and request.page_observation=='site_homepage')
        if not (missing or conflict or homepage):continue
        findings=[f for f in findings if not (f['reference_id']==row.reference_id
                  and f.get('link_outcome')=='website_level_address')]
        message=('The submitted link returned a missing or removed page when checked.' if missing else
                 'The submitted link leads to the website home page, not a page identifying the cited work.' if homepage else
                 'The submitted link’s destination has bibliographic details that conflict with this reference.')
        if conflict and request.identity_differences:
            message = ' '.join(
                f'The reference gives the {d.field} as “{d.submitted}”; the linked page gives “{d.destination}”.'
                for d in request.identity_differences)
        raw=str(source.get('raw_reference') or '')
        if (conflict and set(request.identity_fields) == {'author'}
                and request.identity_differences
                and all(d.field == 'author' for d in request.identity_differences)):
            import re
            author = re.match(r'^(.*?)\s*\((?:18|19|20)\d{2}[a-z]?(?:[),])', raw)
            if author and author.group(1).strip():
                findings.append(dict(finding_type='reference_author_conflict', reference_id=row.reference_id,
                    source=source, finding=message,
                    submitted_link_observations=[row.model_dump(mode='json')],
                    field_difference={'field_name':'author', 'submitted_value':author.group(1).strip()},
                    rectangles=[]))
            # Never relocate a proven author-only difference onto the URL.
            continue
        field=address if address in raw else raw
        findings.append(dict(finding_type='submitted_link_issue',reference_id=row.reference_id,
            source=source,finding=message,link_outcome='missing_page' if missing else 'site_homepage' if homepage else 'destination_conflict',
            submitted_link_observations=[row.model_dump(mode='json')],
            field_difference={'field_name':'submitted_link','submitted_value':field},rectangles=[]))
    return findings


def homepage_address_findings(sources):
    """Report literal address specificity, never reachability or nonexistence."""
    from urllib.parse import urlsplit
    import re
    findings=[]
    for rid, source in sources.items():
        address=str(source.get('url') or '')
        raw=str(source.get('raw_reference') or '')
        title=str(source.get('title') or '').strip()
        author=str(source.get('author') or '').strip().rstrip('.')
        if (not address or address not in raw or len(title.split())<3
                or source.get('source_kind') not in {'report','monograph','journal_article','webpage'}
                or (author and (title.casefold() in author.casefold() or author.casefold() in title.casefold()))
                or re.search(r'\b(?:website|homepage|home page|official site)\b', title, re.I)):
            continue
        try: parsed=urlsplit(address)
        except ValueError: continue
        if (parsed.scheme not in {'https','http'} or not parsed.hostname or parsed.username
                or parsed.path not in {'','/'} or parsed.query or parsed.fragment):
            continue
        findings.append(dict(finding_type='submitted_link_issue', reference_id=rid, source=source,
            rule_id='submitted_website_level_address_v1', link_outcome='website_level_address',
            finding='The submitted URL identifies only the website address, not a specific page for this work.',
            field_difference=dict(field_name='submitted_link',submitted_value=address),rectangles=[]))
    return findings


def render_link_summary(values):
    if not values:
        return ''
    rows = []
    for value in values:
        try:
            rows.append(SubmittedLink.model_validate(value))
        except (ValueError, TypeError):
            return ''
    checked = sum(row.state == 'observed' for row in rows)
    unknown = sum(row.state == 'historical_unknown' for row in rows)
    return (f'<p class="muted submitted-link-summary">Submitted links: {checked} checks attempted; '
            f'{len(rows) - checked - unknown} not checked; {unknown} historical status unknown. '
            'These are access observations, not reference-error counts.</p>')


def without_repeated_identifier_conflicts(link_findings, credibility_findings):
    """One visible wrong-DOI issue, preserving distinct URLs and missing pages.

    The underlying observations stay intact. Registry evidence does not invent
    an HTTP result or replace an independently observed missing-page outcome.
    """
    from app.services.submitted_links import binding, request_url
    covered = set()
    for finding in credibility_findings:
        if finding.get('finding_type') not in {'reference_identifier_conflict', 'potentially_fabricated_reference'}:
            continue
        doi = (finding.get('source') or {}).get('doi')
        if doi:
            covered.add((finding.get('reference_id'), binding(request_url(doi, 'doi'))))
    return [finding for finding in link_findings
            if finding.get('link_outcome') != 'destination_conflict'
            or not any((finding.get('reference_id'), row.get('request_sha256')) in covered
                       for row in finding.get('submitted_link_observations') or [])]
