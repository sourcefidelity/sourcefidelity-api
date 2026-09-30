"""Private development artifacts must not enter the runtime build context."""
from pathlib import Path


def test_docker_context_excludes_private_artifacts():
    root = Path(__file__).resolve().parents[2]
    patterns = set((root / '.dockerignore').read_text().splitlines())
    required = {
        '.git', '.git-rewrite/', '.env', '.env.*', 'private_guiding_history/',
        'private_guiding_contracts/', 'private_guiding_tools/', 'sources/',
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


def test_git_excludes_supplied_sources_regardless_of_file_format():
    import subprocess

    root = Path(__file__).resolve().parents[2]
    paths = ['sources/example.html', 'sources/editions/example.epub',
             'sources/example.txt', 'sources/example.pdf',
             'private_guiding_contracts/example.md',
             'private_guiding_tools/example.py']
    result = subprocess.run(
        ['git', 'check-ignore', '--stdin'], input='\n'.join(paths) + '\n',
        text=True, capture_output=True, cwd=root, check=True,
    )
    assert set(result.stdout.splitlines()) == set(paths)
