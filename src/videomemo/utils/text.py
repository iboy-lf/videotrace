from __future__ import annotations

import re
from collections import Counter


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) <= 2:
                tokens.append(token)
            else:
                tokens.extend(token[idx : idx + 2] for idx in range(len(token) - 1))
        elif len(token) > 1:
            tokens.append(token)
    return tokens


def keyword_set(text: str) -> set[str]:
    return set(tokenize_text(text))


GENERIC_QUERY_TERMS = keyword_set(
    "这个 视频 主要 内容 总结 请 给出 时间戳 证据 回答 问题 哪里 哪段 是否 有没有 "
    "出现 明确 直接 对应 位置 讲了 讲到 说明 指出 列出 每一步 什么 怎么样"
)

GENERIC_QUERY_PHRASES = [
    "这个视频",
    "视频中",
    "视频里",
    "请分别给出时间戳",
    "请给出带时间戳的证据",
    "请给出时间戳",
    "哪里展示了",
    "哪里讲到了",
    "有没有提到",
]


def informative_keyword_set(text: str) -> set[str]:
    cleaned = text.lower()
    for phrase in GENERIC_QUERY_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    return keyword_set(cleaned) - GENERIC_QUERY_TERMS


def top_keywords(text: str, limit: int = 8) -> list[str]:
    counter = Counter(tokenize_text(text))
    return [token for token, _ in counter.most_common(limit)]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s*|[\n\r]+", text)
    return [part.strip() for part in parts if part and part.strip()]
