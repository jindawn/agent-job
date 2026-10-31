"""Guardrails: PII masking, prompt-injection detection, SQL safety, role policy.

Defense in depth - none of these is perfect on its own:
  input  : injection heuristics + PII masking *before* text reaches the LLM
  tools  : role allow-list, JSON-schema validation, read-only SQL (regex + SQLite authorizer)
  output : retrieved/tool text is defused of instruction-like lines, PII masked on the way out,
           canary token detects system-prompt leakage
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CANARY = "CANARY-7f3a91"

ROLE_TOOLS: dict[str, set[str]] = {
    "guest": {"kb_search", "calculator"},
    "support": {"kb_search", "calculator", "sql_query", "create_ticket"},
}

# ---------------------------------------------------------------- PII
_PHONE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
_IDCARD = re.compile(r"(?<![\dXx])(\d{6})(\d{8})(\d{3}[\dXx])(?![\dXx])")
_EMAIL = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def mask_pii(text: str) -> str:
    text = _IDCARD.sub(lambda m: m.group(1)[:3] + "***********" + m.group(3)[-4:], text)
    text = _PHONE.sub(lambda m: f"{m.group(1)}****{m.group(3)}", text)
    text = _EMAIL.sub(lambda m: f"{m.group(1)}***{m.group(2)}", text)
    return text


# ---------------------------------------------------------------- injection
_INJECTION_PATTERNS = [
    r"忽略(之前|以上|上述|前面|所有|先前)[^。\n]{0,10}(指令|提示|规则|要求)",
    r"无视(之前|以上|上述|所有)[^。\n]{0,10}(指令|提示|规则)",
    r"(输出|泄露|告诉我|打印|显示)[^。\n]{0,8}(系统提示|system prompt|提示词)",
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"(reveal|print|show|repeat)[^.\n]{0,20}(system prompt|hidden instructions)",
    r"you are now (?!able)",
    r"你现在(是|扮演)[^。\n]{0,12}(不受|没有)[^。\n]{0,6}限制",
]
_INJ = [re.compile(p, re.I) for p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> list[str]:
    return [p.pattern for p in _INJ if p.search(text)]


def defuse(text: str) -> tuple[str, int]:
    """Replace instruction-like lines inside *retrieved/tool* content. Returns (text, n_removed)."""
    removed = 0
    out = []
    for line in text.splitlines():
        if detect_injection(line):
            out.append("[已拦截疑似指令注入内容]")
            removed += 1
        else:
            out.append(line)
    return "\n".join(out), removed


@dataclass
class InputVerdict:
    blocked: bool
    text: str
    reasons: list[str] = field(default_factory=list)


def check_input(text: str) -> InputVerdict:
    hits = detect_injection(text)
    if hits:
        return InputVerdict(True, text, ["prompt_injection"])
    return InputVerdict(False, mask_pii(text))


def check_output(text: str) -> str:
    if CANARY in text:
        return "抱歉，该内容受安全策略限制，无法提供。"
    return mask_pii(text)


# ---------------------------------------------------------------- SQL
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex)\b", re.I
)


def validate_sql(sql: str) -> str | None:
    """Return an error string if the statement is not a single read-only SELECT."""
    s = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip().rstrip(";").strip()
    if not s:
        return "SQL 为空"
    if ";" in s:
        return "只允许单条语句"
    if not re.match(r"(select|with)\b", s, re.I):
        return "只允许 SELECT 查询"
    m = _FORBIDDEN_SQL.search(s)
    if m:
        return f"包含被禁止的关键字: {m.group(1).upper()}"
    return None
