#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parallel_to_sharegpt.py

병렬 말뭉치 JSON([{source_text, reference_text}...])를
ShareGPTDataset가 읽을 수 있는 형식으로 변환.

예시 프롬프트 형식 (vLLM/LLama 계열):
<s>[INST]Translate this from {src} to {tgt}:
{src}: {text}
{tgt}:[/INST]

사용법:
  python parallel_to_sharegpt.py \
      --input parallel.json \
      --output sharegpt.json \
      --src vi --tgt en
옵션:
  --no-reference       # assistant 응답을 빈 문자열로 (TPS 전용)
  --limit 1000         # 앞에서부터 N개만 변환
  --shuffle            # 셔플 후 --limit 적용
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

PROMPT_TEMPLATE = (
    "<s>[INST]Translate this from {src} to {tgt}:\n{src}: {text}\n{tgt}:[/INST]"
)


def load_parallel(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            "Input JSON must be a list of {source_text, reference_text} objects."
        )
    return data


def to_sharegpt_item(
    src_lang: str,
    tgt_lang: str,
    source_text: str,
    reference_text: str,
    include_reference: bool,
) -> Dict[str, Any]:
    user_prompt = PROMPT_TEMPLATE.format(src=src_lang, tgt=tgt_lang, text=source_text)
    assistant_reply = reference_text if include_reference else ""
    return {
        "conversations": [
            {"from": "user", "value": user_prompt},
            {"from": "assistant", "value": assistant_reply},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="병렬 JSON 경로 (list of {source_text, reference_text})",
    )
    ap.add_argument("--output", required=True, help="출력 ShareGPT JSON 경로")
    ap.add_argument("--src", required=True, help="원문 언어 코드/라벨 (예: vi, ko, en)")
    ap.add_argument("--tgt", required=True, help="목표 언어 코드/라벨 (예: en, ko)")
    ap.add_argument(
        "--no-reference",
        action="store_true",
        help="assistant 응답을 빈 문자열로. (TPS 전용, 정답 불필요)",
    )
    ap.add_argument("--limit", type=int, default=None, help="변환할 최대 개수")
    ap.add_argument("--shuffle", action="store_true", help="셔플 후 limit 적용")
    args = ap.parse_args()

    data = load_parallel(args.input)

    # 선택적으로 섞기 + 제한
    if args.shuffle:
        random.shuffle(data)
    if args.limit is not None:
        data = data[: args.limit]

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        src_text = row.get("source_text")
        ref_text = row.get("reference_text", "")
        if not src_text:
            # source_text 없는 항목은 스킵
            continue
        item = to_sharegpt_item(
            src_lang=args.src,
            tgt_lang=args.tgt,
            source_text=src_text,
            reference_text=ref_text,
            include_reference=(not args.no_reference),
        )
        out.append(item)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✔ Converted {len(out)} items → {args.output}")


if __name__ == "__main__":
    main()
