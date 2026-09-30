import io
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pytest
from app.services.submitted_locator_inventory import _docx_entry, inventory_submitted_locators, SubmittedLocatorInventory
from app.services.reference_layout import extract_reference_layout_from_bytes
from app.services.schemas import ParsedReference
from app.services.assessment_configuration import assessment_link_omissions


def field(p, instruction=' HYPERLINK "https://example.org/PRIVATE_SENTINEL" ', *, simple=False, close=True):
    if simple:
        node=OxmlElement('w:fldSimple');node.set(qn('w:instr'),instruction)
        r=OxmlElement('w:r');t=OxmlElement('w:t');t.text='Source';r.append(t);node.append(r);p._p.append(node)
        return
    for kind in ('begin','instruction','separate','text','end'):
        if kind=='end' and not close:continue
        run=p.add_run()
        if kind=='instruction':
            # Word can split an instruction across runs.
            for part in (instruction[:8],instruction[8:]):
                node=OxmlElement('w:instrText');node.text=part;run._r.append(node)
        elif kind=='text':run.text='Source'
        else:
            node=OxmlElement('w:fldChar');node.set(qn('w:fldCharType'),kind);run._r.append(node)


@pytest.mark.parametrize('simple',[False,True])
def test_complete_hyperlink_fields_are_presence_only(simple):
    doc=Document();p=doc.add_paragraph('Writer, W. (2020). A book. Publisher. ')
    field(p,simple=simple)
    channels,error=_docx_entry(doc,[0])
    assert error is None and 'native_hyperlink' in channels
    stream=io.BytesIO();doc.save(stream);content=stream.getvalue()
    # Match the same visible text used by the existing extractor.
    raw=p.text
    refs=[ParsedReference(reference_id='ref',raw_ref=raw)]
    layout=extract_reference_layout_from_bytes(content,'p.docx',references=refs,citation_format='apa')
    inventory=inventory_submitted_locators(content,references=refs,layout=layout)
    assert inventory.entries[0].status=='supplied'
    assert 'PRIVATE_SENTINEL' not in inventory.model_dump_json()
    assert SubmittedLocatorInventory.model_validate_json(inventory.model_dump_json())==inventory
    assert not assessment_link_omissions({'require_reference_links':True},inventory.model_dump())


@pytest.mark.parametrize('instruction', ['INCLUDETEXT "https://example.org"',
    'HYPERLINK "file:///private/example"', 'HYPERLINK "https://example.org" INCLUDETEXT "x"',
    'HYPERLINK "https://example.org" \\x "unknown"', 'HYPERLINK "https://example.org'])
def test_unsupported_or_compound_fields_abstain(instruction):
    doc=Document();p=doc.add_paragraph('Reference.');field(p,instruction)
    assert _docx_entry(doc,[0])[1]


def test_field_cannot_borrow_next_reference_paragraph():
    doc=Document();p=doc.add_paragraph('First reference.');field(p,close=False)
    q=doc.add_paragraph('Second reference.');end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end');q.add_run()._r.append(end)
    assert _docx_entry(doc,[0])[1] and _docx_entry(doc,[1])[1]


def test_neighboring_reference_does_not_inherit_a_link():
    doc=Document();first=doc.add_paragraph('First reference.');field(first)
    doc.add_paragraph('Second reference.')
    assert _docx_entry(doc,[1])==([],None)


def test_nested_field_is_unresolved():
    doc=Document();p=doc.add_paragraph('Reference.')
    begin=OxmlElement('w:fldChar');begin.set(qn('w:fldCharType'),'begin');p.add_run()._r.append(begin)
    field(p,simple=True)
    assert _docx_entry(doc,[0])[1]


def test_legacy_inventory_version_remains_readable():
    old={'version':'submitted-locator-inventory-v1','paper_sha256':'x','submitted_snapshot_sha256':'y',
         'entries':[],'total_entries':0,'assessed_entries':0,'counts':{'supplied':0,'not_observed':0,'unknown':0}}
    assert SubmittedLocatorInventory.model_validate(old).version=='submitted-locator-inventory-v1'


@pytest.mark.parametrize('option',['l','t'])
def test_bookmark_and_target_window_do_not_erase_external_link_presence(option):
    doc=Document();p=doc.add_paragraph('Reference.')
    field(p,'HYPERLINK "https://example.org" \\'+option+' "destination"')
    assert _docx_entry(doc,[0])==(['native_hyperlink'],None)


def test_link_label_dates_and_dois_do_not_start_new_apa_entries():
    from app.services.parsers.apa_parser import ApaParser
    lines=['Screen Journal. (1932, January). Annual volume. Sample Publisher.',
           'Screen journal. (Vol. 1, Nov. 1931-Apr. 1932) - Catalog',
           'Writer, W. (2019). A journal article. Review, 42(4), 51–61.',
           'Archive | A journal article | 10.1353/example.2019.0065',
           'Other, A. (2020). Another article. Review, 20, 20–30.']
    refs=ApaParser.split_references('\n'.join(lines))
    assert refs==[' '.join(lines[:2]),' '.join(lines[2:4]),lines[4]]
