"""Private development artifacts must not enter the runtime build context."""
from pathlib import Path


def test_docker_context_excludes_private_artifacts():
    root = Path(__file__).resolve().parents[2]
    patterns = set((root / '.dockerignore').read_text().splitlines())
    required = {
        '.git', '.git-rewrite/', '.env', '.env.*', 'private_guiding_history/',
        'STATE.md', 'PLAN.md', 'ARCHITECTURE.md', 'ROADMAP.md', 'AGENTS.md',
        'RESEARCH.md', 'REVIEW*.md', 'Review*.md', 'SYSTEM_PROMPT.md', 'COSTS.md',
        'HPC_SURVEY_ANSWERS.md', 'CAMPUS_TEST_INSTRUCTIONS.md',
        'PHASE_3_5_IMPLEMENTATION.md', 'tests/', 'test_data/', 'test_data_synthetic/',
        'outputs/', 'output/', 'tmp/', 'mocks/', 'Users/', 'Development',
        'HiAgent YAML/', '.sourcefidelity/', '.continue/', '.zcode/', '.vscode/',
        '*.pdf', '*.docx', '*.log', '.last_retrieval_run', 'url_web_results.json',
        'campus_test.py', 'run_retrieval_profile_test.sh',
    }
    assert required <= patterns
    # No later broad negation may reopen an excluded private directory.
    assert {line for line in patterns if line.startswith('!')} == {'!.env.example'}
