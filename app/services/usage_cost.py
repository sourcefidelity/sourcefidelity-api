"""Versioned gross API estimates, not invoices or account-credit balances."""
from datetime import datetime, timezone
from urllib.parse import urlsplit
import math

PRICING_VERSION = 'public-api-prices-2026-09-28'
PRICING_SOURCES = ['https://api-docs.deepseek.com/quick_start/pricing/',
                   'https://brave.com/search/api/',
                   'https://docs.bigmodel.cn/cn/guide/start/pricing',
                   'https://z.ai/pricing',
                   'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml']
# ECB reference rates of 2026-09-25: 1 EUR = 1.1403 USD = 7.6551 CNY.
USD_PER_CNY = 1.1403 / 7.6551
# GLM list prices per million tokens as [cache hit, input, output], checked
# 2026-09-28. BigModel charges in CNY at its standard price (its limited-time
# half price is not assumed); Z.ai international charges in USD.
_GLM_RATES = {
    'open.bigmodel.cn': {'glm-5.3-flash': tuple(r * USD_PER_CNY for r in (0.23, 0.8, 2.8))},
    'api.z.ai': {'glm-5.3-flash': (0.03, 0.15, 0.5)},
}


def llm_price_context(model: str, base_url: str, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    host = urlsplit(base_url).hostname
    if host in _GLM_RATES:
        rates = _GLM_RATES[host].get(model)
        if rates is None:
            return {}
        return dict(pricing_version=PRICING_VERSION, model=model,
                    requested_at=now.astimezone(timezone.utc).isoformat(), usd_per_million=list(rates))
    if host != 'api.deepseek.com':
        return {}
    rates = {'deepseek-flash':(.003,.15,.6), 'deepseek-v4-flash':(.003,.15,.6),
             'deepseek-v4-pro':(.022,.66,1.98)}.get(model)
    if rates is None:
        return {}
    utc = now.astimezone(timezone.utc)
    peak = utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)
    return dict(pricing_version=PRICING_VERSION, model=model, requested_at=utc.isoformat(),
                usd_per_million=[r*(2 if peak else 1) for r in rates])


def llm_cost(usage: dict) -> float | None:
    rates = usage.get('usd_per_million')
    if (not isinstance(rates,list) or len(rates)!=3
            or not all(type(r) in (int,float) and math.isfinite(r) and r>=0 for r in rates)):
        return None
    values = [usage.get(k) for k in ('prompt_cache_hit_tokens','prompt_cache_miss_tokens','completion_tokens')]
    if not all(type(v) is int and v >= 0 for v in values):
        return None
    if sum(values[:2]) != usage.get('prompt_tokens'):
        return None
    return sum(v*r for v,r in zip(values,rates))/1_000_000


def search_trace_cost(sources: list[dict]) -> dict:
    calls = {}; cost = 0.; unknown = 0
    # Only actual query execution receipts, not attempts + candidates + traces.
    for source in sources:
        trace = source.get('reference_discovery_trace') or {}
        for query in trace.get('queries', []):
            provider = str(query.get('execution_provider') or '').casefold()
            count = query.get('provider_calls')
            if type(count) is not int or count < 0 or query.get('cache_hit'):
                continue
            calls[provider] = calls.get(provider, 0)+count
            observed = query.get('cost_usd')
            if type(observed) in (int,float) and math.isfinite(observed) and observed >= 0:
                cost += observed
            elif provider == 'brave':
                cost += count*.005
            # 'duckduckgo' appears only in receipts predating its removal;
            # it was uncosted, so historical totals stay correct.
            elif provider not in ('searxng','duckduckgo'):
                unknown += count
    return dict(search_api_calls=sum(calls.values()), search_calls_by_provider=calls,
                search_cost_usd=round(cost,8), search_cost_unknown_calls=unknown,
                search_cost_scope='Retained execution receipts; before monthly credits; not an invoice.',
                pricing_version=PRICING_VERSION)


def search_cost_by_provider(records: list[dict]) -> list[dict]:
    """Per-provider calls and cost, using the same rules as the total.

    ``cost_basis`` says where a figure comes from: the provider's own reported
    cost, the versioned tariff, a free route, or unknown. Unknown is never 0.
    """
    rows = []
    providers = sorted({str(r.get('execution_provider') or '').casefold() for r in records or []})
    for provider in providers:
        subset = [r for r in records if str(r.get('execution_provider') or '').casefold() == provider]
        totals = search_trace_cost([{'reference_discovery_trace': {'queries': subset}}])
        if not totals['search_api_calls']:
            continue
        observed = any(type(r.get('cost_usd')) in (int, float) and math.isfinite(r['cost_usd'])
                       and r['cost_usd'] >= 0 for r in subset)
        # SearXNG receipts carry an explicit 0.0; a free route is not a
        # provider-reported charge.
        basis = ('free' if provider in ('searxng', 'duckduckgo') else 'observed' if observed
                 else 'tariff' if provider == 'brave' else 'unknown')
        rows.append({'provider': provider or 'unnamed', 'calls': totals['search_api_calls'],
                     'cost_usd': None if basis == 'unknown' else totals['search_cost_usd'],
                     'cost_basis': basis, 'unpriced_calls': totals['search_cost_unknown_calls']})
    return rows
