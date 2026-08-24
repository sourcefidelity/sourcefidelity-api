"""Phase 3.8 evidence artifact and authorization-bound passage regressions."""

from datetime import datetime, timedelta, timezone
import hashlib

import fitz
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.schemas import CitationMarkerMember, InTextCitation
from app.services.source_repository import AdmissionRequest, WorkIdentity, admit_representation
from app.services.storage.backend import StorageBackend
from app.services.verification_evidence import (
    _PassageCandidate,
    _consolidate_nested_passage_entries,
    _SourcePage,
    _source_blocks,
    ClaimAntecedentDependency,
    ClaimEvidence,
    CoverageLevel,
    EvidenceAuthorizationError,
    RelationshipStatus,
    VerificationVerdict,
    authorize_representation,
    build_passage_evidence,
    claim_evidence_from_citation,
    passage_role_from_text,
)
def test_nested_same_page_passages_merge_channels_and_keep_broader_span() -> None:
    shorter = _PassageCandidate(
        page_index=0,
        page_label="10",
        start=100,
        end=500,
        text="a" * 400,
        method="whole_citation_context",
        score=0.91,
    )
    broader = _PassageCandidate(
        page_index=0,
        page_label="10",
        start=100,
        end=650,
        text="a" * 400 + "b" * 150,
        method="lexical_overlap",
        score=0.83,
    )
    other_page = _PassageCandidate(
        page_index=1,
        page_label="11",
        start=100,
        end=650,
        text="c" * 550,
        method="lexical_overlap",
        score=0.80,
    )

    consolidated = _consolidate_nested_passage_entries(
        [
            (shorter, {"whole_citation_context"}),
            (broader, {"candidate_lexical"}),
            (other_page, {"candidate_lexical"}),
        ]
    )

    assert len(consolidated) == 2
    merged, channels = consolidated[0]
    assert (merged.page_index, merged.start, merged.end) == (0, 100, 650)
    assert merged.text == broader.text
    assert merged.score == shorter.score
    assert channels == {"whole_citation_context", "candidate_lexical"}
    assert consolidated[1][0].page_index == 1


def test_passage_roles_exclude_references_and_citation_only_notes_but_keep_body_prose():
    body = (
        "Examination of media representations suggests inaccurate portrayals. "
        "Several studies reach related conclusions (Jones, 2009; Smith, 2011; "
        "Draaisma, 2012). The present discussion then explains the substantive "
        "difference between those findings."
    )
    references = (
        "References\n"
        "Jones, A. (2009). Media representations and disability. Journal One.\n"
        "Smith, B. (2011). Public understanding. Journal Two.\n"
        "Draaisma, C. (2012). Film stereotypes. Journal Three."
    )
    notes = (
        "1 Smith (2009) discusses the earlier statute.\n"
        "2 Jones (2010) supplies the historical citation.\n"
        "3 Brown (2011) records the parallel proceeding.\n"
        "4 White (2012) provides another citation."
    )

    assert passage_role_from_text(body) == "body_prose"
    assert passage_role_from_text(references) == "reference_list"
    assert passage_role_from_text(notes) == "citation_notes"


def test_reference_heading_excludes_later_blank_line_blocks_and_pages():
    pages = [
        _SourcePage(
            index=0,
            label="1",
            text=(
                "Substantive language discussion appears in the article body. "
                "It contains several complete explanatory sentences.\n"
                "References\n"
                "Jones, A. (2009). Language and film."
            ),
        ),
        _SourcePage(
            index=1,
            label="2",
            text="Smith, B. (2011). Translation and culture.\n\nBrown, C. (2012). Loneliness.",
        ),
    ]

    blocks = _source_blocks(pages)
    roles = [role for _page, _start, _end, _text, role in blocks]

    assert roles[0] == "body_prose"
    assert roles[1:] == ["reference_list", "reference_list", "reference_list"]


class MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload(self, file_bytes: bytes, key: str) -> str:
        self.objects.setdefault(key, file_bytes)
        return key

    def download(self, key: str) -> bytes:
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def delete(self, key: str) -> bool:
        self.objects.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list_keys(self, prefix: str) -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


def _pdf(*pages: str) -> bytes:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(60, 60, 540, 760), text, fontsize=11)
    content = document.tobytes()
    document.close()
    return content


def _admit(
    session: Session,
    storage: MemoryStorage,
    content: bytes,
    *,
    scope_type: str = "personal_owner",
    scope_id: str = "owner-1",
    expires_at: datetime | None = None,
    kind: RepresentationKind = RepresentationKind.PDF,
    media_type: str = "application/pdf",
):
    record = admit_representation(
        session,
        storage,
        AdmissionRequest(
            work=WorkIdentity(
                title="A Verified Source",
                work_type="journal_article",
                doi="10.1234/verified-source",
                author="Scholar, A.",
                year="2025",
            ),
            representation=SourceRepresentation(
                kind=kind,
                media_type=media_type,
                content=content,
            ),
            provenance="instructor_upload",
            license_class="commercial_user_upload",
            scope_type=scope_type,
            scope_id=scope_id,
            identity_verdict="verified",
            identity_confidence=0.98,
            completeness_verdict="complete",
            cleanliness_verdict="clean",
            text_quality="digital",
            expires_at=expires_at,
            admitted_by="owner-1",
        ),
    )
    session.commit()
    return record


def _claim(
    text: str,
    *,
    claim_type: str = "paraphrase",
    page_locator: str = "",
) -> ClaimEvidence:
    return ClaimEvidence(
        claim_id="claim-1",
        paper_version_id="paper-v1",
        text=text,
        claim_type=claim_type,
        reference_ids=["ref-1"],
        page_locator=page_locator,
        passage_start=20,
        passage_end=20 + len(text),
    )


def test_exact_quotation_builds_scope_bound_inspectable_artifact(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "Opening material about another subject.",
        "The evidence states that careful verification improves accuracy.\n\n"
        "A neighboring sentence provides necessary context.",
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            'The author writes, "careful verification improves accuracy."',
            claim_type="quotation",
            page_locator="p. 2",
        ),
    )

    assert artifact.source_identity.status.value == "verified"
    assert artifact.coverage.level is CoverageLevel.FULL_TEXT
    assert artifact.relationship.status is RelationshipStatus.NOT_ASSESSED
    assert artifact.verdict is VerificationVerdict.INCONCLUSIVE
    assert artifact.passages[0].retrieval_method == "exact_quotation"
    assert artifact.passages[0].page_index == 1
    assert artifact.passages[0].representation_id == str(record.id)
    assert artifact.passages[0].authorization_scope_type == "personal_owner"
    assert artifact.passages[0].authorization_scope_id == "owner-1"
    assert artifact.passages[0].content_sha256 == hashlib.sha256(content).hexdigest()
    assert "necessary context" in artifact.passages[0].text
    payload = artifact.report_payload()
    assert payload["verdict"] == "inconclusive"
    assert "content" not in payload
    assert "candidate_passages_located_relation_not_assessed" in payload["reason_codes"]


def test_lexical_retrieval_ranks_relevant_passage_and_does_not_claim_support(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "Vampire cinema and adaptation are discussed in this paragraph.",
        "Anime diplomacy connects popular culture, national image, and foreign policy.\n\n"
        "The authors examine cultural influence in international relations.",
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            "Anime can influence national image and foreign policy through popular culture.",
            page_locator="2",
        ),
        top_k=2,
    )

    assert artifact.passages
    assert artifact.passages[0].page_index == 1
    assert artifact.passages[0].retrieval_method == "lexical_overlap"
    assert "foreign policy" in artifact.passages[0].text
    assert artifact.relationship.status is RelationshipStatus.NOT_ASSESSED
    assert artifact.verdict is VerificationVerdict.INCONCLUSIVE


def test_resolved_antecedent_expands_retrieval_without_rewriting_claim(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "The report explains that competition law creates market rivalry among firms."
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )
    claim = _claim("This change improves outcomes.")
    dependency = ClaimAntecedentDependency(
        mention_text="This change",
        mention_local_start=0,
        mention_local_end=len("This change"),
        mention_paper_start=20,
        mention_paper_end=20 + len("This change"),
        resolution_status="resolved",
        confidence="high",
        antecedent_context_index=0,
        antecedent_text="competition law creates market rivalry among firms",
        antecedent_paper_start=0,
        antecedent_paper_end=51,
        method="test",
    )
    claim = claim.model_copy(
        update={
            "granularity": "atomic_claim",
            "antecedent_dependencies": [dependency],
            "context_dependency_status": "resolved",
        }
    )

    artifact = build_passage_evidence(source, claim=claim)

    assert artifact.claim.text == "This change improves outcomes."
    assert artifact.passages
    assert "competition law" in artifact.passages[0].text


def test_no_located_passage_abstains_with_explicit_reason(session: Session) -> None:
    storage = MemoryStorage()
    record = _admit(session, storage, _pdf("This source concerns botanical taxonomy."))
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim("Quantum processors improve cryptographic throughput."),
    )

    assert artifact.passages == []
    assert artifact.verdict is VerificationVerdict.INCONCLUSIVE
    assert artifact.reason_codes == [
        "no_passage_located_in_available_evidence",
        "claim_not_atomized",
    ]


def test_exact_scope_is_required_even_when_same_bytes_exist_elsewhere(
    session: Session,
) -> None:
    storage = MemoryStorage()
    record = _admit(
        session,
        storage,
        _pdf("Authorized course evidence."),
        scope_type="assessment",
        scope_id="assessment-1",
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )

    with pytest.raises(EvidenceAuthorizationError, match="requesting scope"):
        authorize_representation(
            session,
            storage,
            representation_id=record.id,
            scope_type="assessment",
            scope_id="assessment-2",
        )


def test_expired_or_nonaccepted_representation_fails_closed(session: Session) -> None:
    storage = MemoryStorage()
    expired = _admit(
        session,
        storage,
        _pdf("Expired evidence."),
        scope_type="assessment",
        scope_id="assessment-1",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    with pytest.raises(EvidenceAuthorizationError, match="expired"):
        authorize_representation(
            session,
            storage,
            representation_id=expired.id,
            scope_type="assessment",
            scope_id="assessment-1",
        )

    expired.expires_at = datetime.now(timezone.utc) + timedelta(days=1)
    expired.admission_state = "needs_review"
    session.commit()
    with pytest.raises(EvidenceAuthorizationError, match="not accepted"):
        authorize_representation(
            session,
            storage,
            representation_id=expired.id,
            scope_type="assessment",
            scope_id="assessment-1",
        )


def test_content_hash_mismatch_fails_closed(session: Session) -> None:
    storage = MemoryStorage()
    record = _admit(session, storage, _pdf("Original verified bytes."))
    storage.objects[record.content_object.storage_key] = _pdf("Tampered bytes.")

    with pytest.raises(EvidenceAuthorizationError, match="immutable-object"):
        authorize_representation(
            session,
            storage,
            representation_id=record.id,
            scope_type="personal_owner",
            scope_id="owner-1",
        )


def test_plain_text_representation_uses_same_evidence_contract(session: Session) -> None:
    storage = MemoryStorage()
    content = (
        b"First page unrelated.\f"
        b"The study reports a relationship between cultural exports and national image."
    )
    record = _admit(
        session,
        storage,
        content,
        kind=RepresentationKind.PLAIN_TEXT,
        media_type="text/plain",
    )
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim("Cultural exports shape national image.", page_locator="2"),
    )

    assert artifact.coverage.representation_kind == "plain_text"
    assert artifact.passages[0].page_index == 1
    assert artifact.verdict is VerificationVerdict.INCONCLUSIVE


def test_exact_quotation_normalizes_line_end_hyphenation(session: Session) -> None:
    storage = MemoryStorage()
    content = b"Careful verifi-\ncation improves the reliability of academic evidence."
    record = _admit(
        session,
        storage,
        content,
        kind=RepresentationKind.PLAIN_TEXT,
        media_type="text/plain",
    )
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            '"Careful verification improves the reliability of academic evidence."',
            claim_type="quotation",
        ),
    )

    assert artifact.passages[0].retrieval_method == "exact_quotation"


def test_claim_conversion_preserves_application_identity_and_coordinates() -> None:
    citation = InTextCitation(
        reference_ids=["paper-v1-ref-0001"],
        link_status="linked",
        text="A verified cited claim.",
        claim_type="paraphrase",
        citation_marker="(Scholar, 2025, p. 14)",
        page_number="14",
        passage_start=100,
        passage_end=123,
    )

    first = claim_evidence_from_citation(citation, paper_version_id="paper-v1")
    second = claim_evidence_from_citation(citation, paper_version_id="paper-v1")

    assert first.claim_id == second.claim_id
    assert first.reference_ids == ["paper-v1-ref-0001"]
    assert first.passage_start == 100
    assert first.passage_end == 123
    assert first.page_locator == "14"
    assert first.granularity == "citation_unit"
    assert first.atomization_method == "not_run"


@pytest.mark.parametrize(
    "citation, message",
    [
        (
            InTextCitation(
                reference_ids=["ref-1"],
                text="Rejected model text.",
                passage_start=0,
                passage_end=20,
                drop_reason="text_not_in_original",
            ),
            "Rejected citation",
        ),
        (
            InTextCitation(
                reference_ids=[],
                candidate_reference_ids=["ref-1", "ref-2"],
                link_status="ambiguous",
                text="Ambiguous claim.",
                passage_start=0,
                passage_end=16,
            ),
            "uniquely linked",
        ),
        (
            InTextCitation(
                reference_ids=["ref-1"],
                text="Unlocated claim.",
                passage_start=-1,
                passage_end=-1,
            ),
            "coordinates",
        ),
    ],
)
def test_claim_conversion_rejects_untrusted_or_ambiguous_spans(
    citation: InTextCitation,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        claim_evidence_from_citation(citation, paper_version_id="paper-v1")


def test_image_only_pdf_is_not_reported_as_full_text_coverage(session: Session) -> None:
    storage = MemoryStorage()
    document = fitz.open()
    document.new_page()
    content = document.tobytes()
    document.close()
    record = _admit(session, storage, content)
    record.text_quality = "pure_scan"
    session.commit()
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim("A claim that requires source text."),
    )

    assert artifact.coverage.level is CoverageLevel.UNAVAILABLE
    assert artifact.verdict is VerificationVerdict.NOT_ASSESSED
    assert "source_text_unavailable" in artifact.reason_codes


def test_collective_claim_requires_and_preserves_source_specific_binding(
    session: Session,
) -> None:
    storage = MemoryStorage()
    record = _admit(
        session,
        storage,
        _pdf("Evidence matters when sources are checked independently."),
    )
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )
    text = "Smith (2020) and Jones (2021) argue that evidence matters."
    smith_text = "Smith (2020)"
    jones_text = "Jones (2021)"
    jones_start = text.index(jones_text)
    claim = ClaimEvidence(
        claim_id="collective-claim",
        paper_version_id="paper-v1",
        text=text,
        reference_ids=["ref-smith", "ref-jones"],
        citation_marker="Smith (2020) and Jones (2021)",
        citation_markers=[
            CitationMarkerMember(
                text=smith_text,
                local_start=0,
                local_end=len(smith_text),
                reference_ids=["ref-smith"],
                marker_type="narrative",
            ),
            CitationMarkerMember(
                text=jones_text,
                local_start=jones_start,
                local_end=jones_start + len(jones_text),
                reference_ids=["ref-jones"],
                marker_type="narrative",
            ),
        ],
        citation_marker_type="narrative",
        passage_start=100,
        passage_end=100 + len(text),
    )

    with pytest.raises(ValueError, match="active reference binding"):
        build_passage_evidence(source, claim=claim)

    smith = build_passage_evidence(
        source,
        claim=claim,
        active_reference_id="ref-smith",
        cited_author_label="Smith",
    )
    jones = build_passage_evidence(
        source,
        claim=claim,
        active_reference_id="ref-jones",
        cited_author_label="Jones",
    )

    assert smith.source_binding.reference_id == "ref-smith"
    assert smith.source_binding.marker_text == smith_text
    assert smith.source_binding.cited_author_label == "Smith"
    assert jones.source_binding.reference_id == "ref-jones"
    assert jones.source_binding.marker_text == jones_text
    assert jones.source_binding.cited_author_label == "Jones"
    assert smith.verification_id != jones.verification_id
