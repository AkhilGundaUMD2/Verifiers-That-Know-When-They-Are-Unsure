"""Put page_facts and the with_page_facts flag back into baselines/llm_judge.py.

Run from the repo root:  python restore_page_facts.py
Every edit asserts that its anchor appears exactly once, so a file that has
changed since this was written is left untouched instead of half-patched.
"""
from pathlib import Path

path = Path("baselines/llm_judge.py")
text = path.read_text()
assert "page_facts" not in text, "page_facts is already in this file"

PAGE_FACTS = '''def page_facts(steps, limit=MAX_FACT_LINES):
    """Page lines with a number, plus the line above each, listed once at the step they first appeared.

    If there are more than limit lines, the newest are kept, because the pages
    nearest the agent's answer matter most.
    """
    seen, facts = set(), []
    for number, step in enumerate(steps, start=1):
        lines = (step.get("axtree") or "").splitlines()
        for index, line in enumerate(lines):
            # Look at the element's text, not its bid, which is always a number.
            if not re.search(r"\\d", " ".join(re.findall(r"'([^']*)'", line))):
                continue
            # The label for a price or count is usually the line just above it.
            for candidate in lines[max(0, index - 1):index + 1]:
                key = " ".join(re.findall(r"'([^']*)'", candidate)) or candidate.strip()
                if key in seen:
                    continue
                seen.add(key)
                facts.append(f"  step {number}: {candidate.strip()[:160]}")
    return facts[-limit:]


'''

FACTS_BLOCK = '''    if with_page_facts:
        facts = page_facts(steps)
        if facts:
            lines += ["", "Numbers and values that appeared on pages the agent visited:"] + facts
'''

edits = [
    ("import json\n\nfrom .browser_checker",
     "import json\nimport re\n\nfrom .browser_checker"),
    ("MAX_TREE_CHARS = 8000\n",
     "MAX_TREE_CHARS = 8000\n\n# Cap on page-fact lines so one trajectory stays a modest request.\nMAX_FACT_LINES = 300\n"),
    ("def build_prompt(record):\n",
     PAGE_FACTS + "def build_prompt(record, with_page_facts=False):\n"),
    ('    tree = steps[-1].get("axtree") if steps else None\n',
     FACTS_BLOCK + '    tree = steps[-1].get("axtree") if steps else None\n'),
]
for old, new in edits:
    assert text.count(old) == 1, "expected exactly one match for: " + repr(old[:50])
    text = text.replace(old, new, 1)
path.write_text(text)
print("page_facts restored in", path)
