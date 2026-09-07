from app.services.retrieval_trace_audit import audit_retrieval_funnel


def test_audit_separates_web_provider_conversion_and_marks_old_trace_gap():
    result = audit_retrieval_funnel(
        [
            {
                "status": "durable_authorized",
                "reference_discovery_trace": {
                    "queries": [
                        {
                            "provider": "web_search",
                            "execution_provider": "exa",
                            "execution_outcome": "results",
                        }
                    ],
                    "attempts": [
                        {"provider": "web_search", "outcome": "candidate_found"}
                    ],
                    "candidates": [
                        {
                            "provider": "web_search",
                            "discovery_provider": "exa",
                            "location_available": True,
                            "acquisition_outcome": "acquired",
                        },
                        {
                            "provider": "openalex",
                            "location_available": True,
                            "acquisition_outcome": "metadata_only",
                        },
                    ],
                },
            }
        ]
    )

    assert result["web_provider_conversion"]["exa"]["acquired_documents"] == 1
    assert result["web_provider_conversion"]["exa"]["candidate_conversion_rate"] == 1.0
    assert result["structured_provider_locations"]["openalex"][
        "location_conversion_trace_missing"
    ] is True
    assert "normalized_query" not in str(result)
