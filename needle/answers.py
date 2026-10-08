"""Pull the final answer out of a sample and write it in one canonical form, so equal answers vote together."""

import ast
import operator
import re
import warnings
from fractions import Fraction
from functools import lru_cache

# Explicit answer lines, most specific first. The first match wins: text after it is usually the
# model inventing the next question.
_FORMATS = [
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"[Tt]he final answer is:?\s*(.+?)\s*(?:\.?\s*I hope|\.\s*\n|\n|\.?\s*$)"),
    re.compile(r"[Ff]inal [Aa]nswer:\s*(.+?)\s*(?:\.?\s*I hope|\.\s*\n|\n|\.?\s*$)"),
    re.compile(r"[Tt]he answer is:?\s*(.+?)\s*(?:\.\s*\n|\n|\.?\s*$)"),
]

_UNITS = (
    "square|ways|integers|dollars|mph|inches|ft|hours|km|units|points|feet|minutes|digits|cents|"
    "degrees|cm|gm|pounds|meters|meals|edges|students|childrentickets|multiples|sue"
)
_REMOVE = [
    "\\ldots", "\\dots", "\\text{s}", "\\text{.}", "\\text{}^2", "\\text{}^3", "\\text{}",
    "\\mathrm{th}", "^\\circ", "^{\\circ}", "\\;", ",\\!", "\\!", "\\,", "{,}", '"',
    "\\displaystyle", "\\left", "\\right", "\\%", "%",
]
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def extract(text: str) -> tuple[str | None, bool]:
    """The final answer in a sample, and whether it came from an explicit answer line (strict)."""
    for pattern in _FORMATS:
        found = pattern.search(text)
        if found and found.group(1).strip():
            return found.group(1).strip(), True
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed, False
    return None, False


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, braces balanced."""
    start = text.rfind("\\boxed")
    i = text.find("{", start) if start >= 0 else -1
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1 : j]
    return None


def normalize(answer: str) -> str:
    """Canonical form: same value, same string. '$57,500' and '57500.00' both become '57500'."""
    a = answer.strip()
    boxed = last_boxed(a)
    if boxed is not None:
        a = boxed
    a = re.split(r"(?<![<>!])=", a)[-1]
    a = re.sub(r"\\text\{\s*([^{}]*)\}", r"\1", a)
    a = re.sub(r"\\(?:textbf|mathbf|mathrm|overline)\{([^{}]*)\}", r"\1", a)
    a = re.sub(r"\\[dt]frac", r"\\frac", a)
    for token in _REMOVE:
        a = a.replace(token, "")
    a = re.sub(r"(?<![a-zA-Z\\])(" + _UNITS + r")\b", "", a)
    a = a.replace("\\$", "").replace("$", "").replace("\\ ", "").replace(" ", "")
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", a)
    a = a.rstrip(".").strip()
    # A number followed by a word is a number with a unit: '48grams', '5miles', '6th', '14,400/year'.
    m = re.fullmatch(r"(-?[\d,]*\.?\d+)(?:/?[a-zA-Z]{2,}.*|\([a-zA-Z].*\))", a)
    if m:
        a = m.group(1)
    a = re.sub(r"(?<=[\d}])[a-zA-Z]{2,}$", "", a)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", a):
        a = a.replace(",", "")
    value = _evaluate(a)
    return value if value is not None else a


def _evaluate(a: str) -> str | None:
    """Exact value of plain arithmetic: '0.50', '1/2', '\\frac{2^7}{3}', '18^2', '2\\cdot3'. None otherwise."""
    if not a or not re.fullmatch(r"[\d.+\-*/^(){}\\fraccdotimes]+", a):
        return None
    expr = a.replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
    expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", expr)
    if "\\" in expr or "{" in expr or "}" in expr:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # '1.5.2' and friends
            tree = ast.parse(expr, mode="eval")
        value = _eval(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, OverflowError):
        return None
    if value is None:
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _eval(node: ast.AST) -> Fraction | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _eval(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right) > 64 or abs(left) > 10**6:
                return None
            return left ** int(right)
        op = _OPS.get(type(node.op))
        return op(left, right) if op else None
    return None


@lru_cache(maxsize=1 << 16)
def answer_of(text: str) -> tuple[str | None, bool]:
    """Extract then normalize. (None, False) when the sample states no answer."""
    raw, strict = extract(text)
    if raw is None:
        return None, False
    value = normalize(raw)
    return (value, strict) if value else (None, False)


_CALC = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
_INLINE = re.compile(
    r"(?<![\w.^/])(-?\d[\d,]*(?:\.\d+)?(?:\s*[-+*/×]\s*\d[\d,]*(?:\.\d+)?)+)\s*=\s*\$?(-?\d[\d,]*(?:\.\d+)?)(?![\d^/])"
)


@lru_cache(maxsize=1 << 16)
def slips(text: str) -> int:
    """How many plain arithmetic statements in the sample are wrong ('15 * 50 * 30 = 45000').
    A sample that gets its own arithmetic wrong is far less likely to be right."""
    statements = set()  # '15 * 50 = <<15*50=750>>750' says one thing twice: count it once
    for pattern in (_CALC, _INLINE):
        for lhs, rhs in pattern.findall(text):
            statements.add((lhs.replace(",", "").replace("×", "*").replace(" ", ""), rhs.replace(",", "").strip()))
    wrong = 0
    for lhs, rhs in statements:
        a, b = _evaluate(lhs), _evaluate(rhs)
        if a is not None and b is not None:
            x, y = Fraction(a), Fraction(b)
            wrong += abs(x - y) > max(1, abs(y)) / 10000
    return wrong


def prompt_answers(prompt: str) -> frozenset[str]:
    """Answers already written in the prompt (worked examples). A sample that repeats one is suspect."""
    found = set()
    for pattern in _FORMATS[:3]:
        for raw in pattern.findall(prompt):
            value = normalize(raw)
            if value:
                found.add(value)
    return frozenset(found)
