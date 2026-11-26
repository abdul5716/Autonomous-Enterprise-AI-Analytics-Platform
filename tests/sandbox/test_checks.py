import pytest

from analytics_platform.tools.sandbox.checks import check_code


@pytest.mark.parametrize(
    "code",
    [
        "import numpy as np\nresult = np.mean([1, 2])",
        "from scipy import stats\nresult = stats.pearsonr([1, 2, 3], [2, 4, 7])[0]",
        "result = df.groupby('region').amount.sum().reset_index()",
        "import json\nresult = json.loads('{\"a\": 1}')",
        "if __name__ == '__main__':\n    pass\nresult = 1",
    ],
)
def test_allows_ordinary_analysis(code: str) -> None:
    assert check_code(code) == []


@pytest.mark.parametrize(
    ("code", "problem"),
    [
        ("import os\nresult = 1", "import of 'os'"),
        ("import subprocess\nresult = 1", "import of 'subprocess'"),
        ("from urllib import request\nresult = 1", "import of 'urllib'"),
        ("result = open('/etc/passwd').read()", "calling 'open'"),
        ("result = eval('1+1')", "calling 'eval'"),
        ("result = __import__('os')", "calling '__import__'"),
        ("result = ().__class__.__base__.__subclasses__()", "attribute '__class__'"),
        ("result = getattr(df, 'to_csv')", "calling 'getattr'"),
        ("result = pd.read_csv('http://x/y.csv')", "file I/O ('read_csv')"),
        ("df.to_csv('/tmp/x')\nresult = 1", "file I/O ('to_csv')"),
        ("import numpy as np\nresult = np.load('x.npy')", "file I/O ('load')"),
        ("x = __builtins__\nresult = 1", "name '__builtins__'"),
        ("total = 1", "must assign its answer"),
        ("result = (", "Syntax error"),
    ],
)
def test_rejects_unsafe_or_incomplete_code(code: str, problem: str) -> None:
    assert any(problem in p for p in check_code(code))


def test_rejects_oversized_code() -> None:
    assert "too long" in check_code("result = 1\n" + "#" * 9000)[0]
