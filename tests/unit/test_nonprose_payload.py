"""Retrieved text must be a document's prose, not script or feed output.

Found by audit on 2026-09-23. Two `text/plain` representations filed as *EMI
in higher education: An ELF perspective* held a cookie-consent JavaScript
bundle, and one filed as *Towards a Game Theory of Game* held a WordPress RSS
feed. The only check on retrieved text was a 500-character minimum, so all
three were stored as full text of works they have nothing to do with.

Thresholds come from the stored corpus, not from taste: script markers appear
2.20 times per 1,000 characters in the JavaScript and at most 0.05 in every
genuine source; tags appear 5.15 per 1,000 in the feed against at most 0.24.
"""
import pytest

from app.services.source_validator import detect_nonprose_payload

_JS = (
    "self.airgap = { overrides: [], cookieOverrides: [], ...self.airgap, }; "
    "const allowGcmAdvanced = (event) => { if (event.purposes.has('GcmAdvanced')) "
    "{ event.allow(); } }; self.airgap.overrides.push({ override: allowGcmAdvanced }); "
    '"use strict"; function _typeof(t) { return (_typeof = "function" == typeof Symbol '
    "? function (t) { return typeof t } : function (t) { return t === Symbol.prototype "
    '? "symbol" : typeof t })(t) } !function () { self.airgap?.ready }();'
) * 3

_RSS = (
    "<rss version='2.0'><channel><title>Divided Kingdom, 561</title>"
    "<link>https://dividedkingdom561.wordpress.com</link>"
    "<description>Blog for my M.A. process</description>"
    "<item><title>Historical Monumentalism</title><pubDate>Tue, 18 May 2021</pubDate>"
    "<guid>https://dividedkingdom561.wordpress.com/2021/05/18/x/</guid></item>"
    "<item><title>Wallenstein</title><link>https://example.org/a</link></item>"
    "</channel></rss>"
)

_PROSE = (
    "Digital narratives are immersed in a new media environment that maintains a "
    "close relationship with the construction and comprehension of contemporary "
    "society. The emergence of a new generation of media based on an innovative "
    "model founded on networks has changed how audiences encounter storytelling. "
    "This article examines three cases and argues that the shift is not merely "
    "technological but institutional, with consequences for how works circulate."
)


def test_a_javascript_bundle_is_refused() -> None:
    assert detect_nonprose_payload(_JS) == "executable script rather than document text"


def test_a_feed_is_refused() -> None:
    assert detect_nonprose_payload(_RSS) == "a feed or structured data payload"


def test_ordinary_prose_is_kept() -> None:
    assert detect_nonprose_payload(_PROSE) is None


def test_prose_that_quotes_code_is_kept() -> None:
    """A work discussing software is still the work."""
    article = (
        _PROSE + " The authors illustrate the point with a short listing, "
        "function render(node) { return node.value; }, which shows the pattern "
        "under discussion. " + _PROSE
    )

    assert detect_nonprose_payload(article) is None


def test_prose_that_shows_a_markup_example_is_kept() -> None:
    article = (
        _PROSE + " Encoding practice varies: some editions mark this as "
        "<title>Heavenly Bodies</title> while others use <name>. " + _PROSE
    )

    assert detect_nonprose_payload(article) is None


def test_the_alphabetic_ratio_is_not_used_as_a_signal() -> None:
    """A genuine stored source sits at 0.67, below the JavaScript's 0.72.

    Rejecting on that ratio would refuse real documents, which is why the rule
    keys on script and markup density instead.
    """
    dense = ("Table 3. 12,441 (14.2%) 8,013 (9.1%) 2,214 (2.5%) 1,002 (1.1%) "
             "n = 87,412; p < .001; 95% CI [1.02, 1.44]. ") * 12

    assert detect_nonprose_payload(dense) is None


@pytest.mark.parametrize("text", ["", "short", None])
def test_a_payload_too_small_to_judge_is_not_refused(text) -> None:
    """Absence of evidence is not a rejection; length checks live elsewhere."""
    assert detect_nonprose_payload(text) is None
