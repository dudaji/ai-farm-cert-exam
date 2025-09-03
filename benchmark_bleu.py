import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
from sacrebleu import corpus_bleu, sentence_bleu
from tqdm import tqdm

# --- Constants ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_SAMPLES = 1000
CONCURRENT_WORKERS = 100
TRANSLATE_BASE_URL = "http://localhost:8000"
TRANSLATE_URL = f"{TRANSLATE_BASE_URL}/v1/chat/completions"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
LANGUAGE_MAP = {"en": "English", "es": "Spanish", "vi": "Vietnamese", "ko": "Korean"}
SEED = 42

random.seed(SEED)


def _norm_role(d: dict) -> str:
    role = (d.get("role") or d.get("from") or "").lower()
    if role in ("user", "human"):
        return "user"
    if role in ("assistant", "gpt", "bot", "model"):
        return "assistant"
    return role


def _msg_text(d: dict) -> str:
    return (d.get("value") or d.get("content") or "").strip()


def load_data_from_json(filepath: str) -> Tuple[List[str], List[str]]:
    """
    ShareGPT JSON을 로드해 [user_prompt], [assistant_reference] 리스트 반환.
    모든 항목에 ref(assistant 첫 메시지)가 '존재한다'는 가정으로 검증합니다.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Top-level JSON must be a list.")

    random.shuffle(data)

    user_prompts: List[str] = []
    references: List[str] = []

    for idx, entry in enumerate(data):
        convs = entry.get("conversations")
        if not isinstance(convs, list) or len(convs) < 2:
            raise ValueError(
                f"[{filepath}] entry {idx}: conversations must have >= 2 turns."
            )
        user_idx = next(
            (
                i
                for i, m in enumerate(convs)
                if _norm_role(m) == "user" and _msg_text(m)
            ),
            None,
        )
        if user_idx is None:
            raise ValueError(f"[{filepath}] entry {idx}: missing user message.")
        user_msg = _msg_text(convs[user_idx])

        assistant_idx = next(
            (
                i
                for i in range(user_idx + 1, len(convs))
                if _norm_role(convs[i]) == "assistant" and _msg_text(convs[i])
            ),
            None,
        )
        if assistant_idx is None:
            raise ValueError(f"[{filepath}] entry {idx}: missing assistant reference.")
        assistant_msg = _msg_text(convs[assistant_idx])

        user_prompts.append(user_msg)
        references.append(assistant_msg)

    print(f"Successfully loaded {len(user_prompts)} entries from {filepath}")
    return user_prompts, references


def build_openai_request_body_from_prompt(
    user_prompt: str, source_lang: str, target_lang: str
) -> dict:
    src = LANGUAGE_MAP.get(source_lang, source_lang)
    tgt = LANGUAGE_MAP.get(target_lang, target_lang)
    system_prompt = (
        f"You are an expert translator. "
        f"Your task is to accurately translate text from {src} to {tgt}. "
        f"Provide only the translated text, without any additional explanations or introductions."
    )
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
        "stream": False,
    }


def translate_prompt(user_prompt: str, src_lang: str, tgt_lang: str) -> Dict[str, Any]:
    """
    단일 요청 번역 + 지연시간/상태/usage 포함 결과 반환.
    """
    body = build_openai_request_body_from_prompt(user_prompt, src_lang, tgt_lang)
    t0 = time.time()
    try:
        resp = requests.post(TRANSLATE_URL, headers=HEADERS, json=body, timeout=60)
        latency = time.time() - t0
        status = resp.status_code
        resp.raise_for_status()
        payload = resp.json()
        text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        usage = payload.get("usage", {})
        return {
            "ok": True,
            "translation": text,
            "latency_sec": latency,
            "status_code": status,
            "usage": usage,
            "error": "",
        }
    except requests.exceptions.RequestException as e:
        latency = time.time() - t0
        status = getattr(e.response, "status_code", None)
        return {
            "ok": False,
            "translation": "",
            "latency_sec": latency,
            "status_code": status,
            "usage": {},
            "error": str(e),
        }


def evaluate_language_pair(
    json_filepath: str, src_lang: str, tgt_lang: str, output_path: Optional[str] = None
):
    pair = f"{src_lang}-{tgt_lang}"

    # 1) ShareGPT JSON 로드 (ref 존재 가정 + 검증)
    user_prompts, ref_texts = load_data_from_json(json_filepath)
    if NUM_SAMPLES and len(user_prompts) > NUM_SAMPLES:
        user_prompts = user_prompts[:NUM_SAMPLES]
        ref_texts = ref_texts[:NUM_SAMPLES]

    n = len(user_prompts)
    print(
        f"Evaluating {pair} on {n} samples with {CONCURRENT_WORKERS} concurrent workers..."
    )

    # 결과 수집 버퍼
    predictions: List[str] = [""] * n
    latencies: List[Optional[float]] = [None] * n
    statuses: List[Optional[int]] = [None] * n
    errors: List[str] = [""] * n
    usages: List[Dict[str, Any]] = [{}] * n

    # 2) 병렬 번역 + 완료 시마다 로그 출력
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        future_to_index = {
            executor.submit(translate_prompt, prompt, src_lang, tgt_lang): i
            for i, prompt in enumerate(user_prompts)
        }

        progress = tqdm(
            as_completed(future_to_index),
            total=n,
            desc=f"Translating {pair}",
        )
        for future in progress:
            index = future_to_index[future]
            try:
                result = future.result()
                predictions[index] = result["translation"]
                latencies[index] = result["latency_sec"]
                statuses[index] = result["status_code"]
                errors[index] = result["error"]
                usages[index] = result["usage"] or {}
            except Exception as e:
                # 이 경우엔 future 자체 실패
                predictions[index] = ""
                latencies[index] = None
                statuses[index] = None
                errors[index] = f"worker-exception: {e}"
                usages[index] = {}

    # 3) BLEU 계산 (ref 존재 가정)
    valid_preds = [p for p in predictions if p]
    valid_refs = [r for p, r in zip(predictions, ref_texts) if p]
    sentence_bleus = []
    if valid_preds:
        for p, r in zip(predictions, ref_texts):
            if p:
                sb = sentence_bleu(p, [r], tokenize="13a")
                sentence_bleus.append(round(sb.score, 2))
            else:
                sentence_bleus.append(None)
        _bleu = corpus_bleu(valid_preds, [valid_refs], tokenize="13a")
        metrics = {"bleu": round(_bleu.score, 2)}
    else:
        metrics = {
            "bleu": None,
            "note": "No valid predictions; BLEU not computed.",
        }

    # 4) JSON 결과 저장 (BLEU 메트릭 포함)
    results_data = [
        {
            "index": i,
            "pair": pair,
            "prompt": prompt,
            "reference_translation": ref,
            "model_translation": pred,
            "latency_sec": latencies[i],
            "status_code": statuses[i],
            "usage": usages[i],
            "error": errors[i],
            "sentence_bleu": sentence_bleus[i],
        }
        for i, (prompt, ref, pred) in enumerate(
            zip(user_prompts, ref_texts, predictions)
        )
    ]
    output_payload = {
        "pair": pair,
        "num_samples": n,
        "metrics": metrics,
        "results": results_data,
    }
    json_out = output_path if output_path else f"translation_results_{pair}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(f"Translation results saved to {json_out}")

    # 5) 콘솔 출력 (BLEU)
    if metrics.get("bleu") is not None:
        print(f"{pair} BLEU score: {metrics['bleu']:.2f}\n")
    else:
        print(f"No valid translations for {pair}. BLEU score skipped.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to ShareGPT JSON file")
    parser.add_argument("--src", required=True, help="Source language code (e.g., vi)")
    parser.add_argument("--tgt", required=True, help="Target language code (e.g., en)")
    parser.add_argument(
        "--output",
        help="Output JSON filepath (default: translation_results_{src}-{tgt}.json)",
    )
    args = parser.parse_args()

    t0 = time.time()
    evaluate_language_pair(args.input, args.src, args.tgt, args.output)
    print(f"Total execution time: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    main()
