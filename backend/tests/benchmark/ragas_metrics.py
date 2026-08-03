"""
ragas_metrics.py — Đánh giá chất lượng câu trả lời kiểu RAGAS bằng LLM-as-judge.

Đây KHÔNG phải thư viện `ragas` chính thức (tránh thêm dependency LangChain wrapper
nặng nề) mà là bản triển khai rút gọn. Đo 2 metric cốt lõi:

  - Faithfulness      : câu trả lời có bám sát (không bịa) so với context đã retrieve không.
  - Answer Relevancy   : câu trả lời có thực sự giải quyết đúng câu hỏi không.

Judge dùng Gemini (free tier) thay vì model Ollama local. Lý do: đã thử cả
llama3.2:1b (quá dễ dãi — chấm cao cho câu bịa) lẫn qwen2.5:3b-instruct (quá khắt
khe — tự mâu thuẫn, nói "context không chứa X" ngay cả khi context có nguyên văn X)
làm judge, và cả 2 đều không đáng tin ở quy mô 1-3B tham số. Thêm nữa, judge cùng
model với generator (qwen2.5:3b) là "đá bóng kiêm thổi còi". Gemini chỉ dùng CHO
JUDGE (~92 call/lần benchmark = 46 câu × 2 metric) — thấp hơn nhiều so với việc
embed hàng nghìn chunk từng làm hết quota free tier — nên nhiều khả năng vẫn nằm
trong free tier dù generator/embedding vẫn giữ nguyên local (LocalGenerator +
LocalEmbedder/BGE-M3).

Nếu sau này cần đúng thuật toán RAGAS gốc (decomposition câu thành statements,
NLI scoring...), có thể thay thế bằng package `ragas` mà không cần đổi interface
của 2 hàm bên dưới.
"""
import json
import logging
import re
import time

logger = logging.getLogger(__name__)

GEMINI_JUDGE_MODEL_DEFAULT = "gemini-3.1-flash-lite"
CLAUDE_JUDGE_MODEL_DEFAULT = "claude-haiku-4-5"

# Import google.genai / anthropic LAZY (trong từng hàm judge) thay vì ở top-level:
# cho phép chạy benchmark với JUDGE_PROVIDER=claude mà KHÔNG cần cài/cấu hình Google,
# và ngược lại — không ép cả 2 SDK phải import được cùng lúc.

# Free tier gemini-3.1-flash-lite: 15 RPM. Benchmark gọi ~2 request/câu hỏi (faithfulness
# + answer_relevancy), với 46 câu = ~92 request — chắc chắn vượt 15 RPM nếu chạy dồn dập.
# Giãn cách chủ động (thay vì chỉ retry khi đã bị 429) để tránh bị tạm khóa quota giữa chừng.
_MIN_SECONDS_BETWEEN_CALLS = 4.2  # 60s / 15 RPM = 4.0s, cộng thêm biên an toàn
_last_call_start = 0.0


def _rate_limit_wait() -> None:
    """Đảm bảo khoảng cách tối thiểu giữa 2 lần gọi Gemini, tính từ lúc BẮT ĐẦU
    request trước (không phải lúc kết thúc) — thời gian round-trip của request
    trước cũng được tính vào khoảng giãn cách, tránh sleep thừa."""
    global _last_call_start
    now = time.monotonic()
    elapsed = now - _last_call_start
    if elapsed < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_start = time.monotonic()

# Câu mở đầu refusal chuẩn theo đúng SYSTEM_INSTRUCTION của Generator/LocalGenerator
# (app/core/generator.py, app/core/local_generator.py). Dùng rule-based string match
# thay vì hỏi LLM judge, vì đây là logic nhị phân đơn giản mà judge nhỏ đã chứng minh
# không đáng tin (Sprint 0: chấm sai cả 3/3 câu negative, kể cả khi prompt đã dặn rõ
# is_negative_expected).
_REFUSAL_MARKERS = [
    "tôi không tìm thấy thông tin",
    "không tìm thấy thông tin này",
]


def is_refusal_answer(answer: str) -> bool:
    """Phát hiện câu trả lời kiểu 'Tôi không tìm thấy thông tin...' bằng string match,
    không phụ thuộc LLM. Chỉ xét ~200 ký tự đầu để tránh match nhầm câu văn nhắc lại
    cụm này ở giữa một câu trả lời dài nhưng thực chất không phải từ chối."""
    normalized = answer.strip().lower()[:200]
    return any(marker in normalized for marker in _REFUSAL_MARKERS)


_FAITHFULNESS_PROMPT = """Bạn là một giám khảo (judge) chấm điểm hệ thống RAG. Nhiệm vụ: đánh giá xem CÂU TRẢ LỜI có "faithful" (trung thực, không bịa đặt thông tin ngoài CONTEXT) hay không.

Quy tắc chấm điểm:
- Tách câu trả lời thành các claim (nhận định) độc lập.
- Với mỗi claim, kiểm tra xem nó có được CONTEXT hỗ trợ (support) hay không — support bao gồm cả diễn giải/paraphrase, không chỉ trích dẫn nguyên văn.
- Nếu câu trả lời là lời từ chối hợp lệ (ví dụ "không tìm thấy thông tin") và CONTEXT thực sự không chứa câu trả lời, đó là faithful (score cao).
- score = (số claim được support) / (tổng số claim). Nếu không có claim nào (câu từ chối), score = 1.0.

CÂU HỎI:
{question}

CONTEXT (các đoạn tài liệu đã retrieve):
{context}

CÂU TRẢ LỜI CẦN CHẤM:
{answer}

Trả về JSON với đúng các key sau:
{{"score": <float 0.0-1.0>, "reasoning": "<giải thích ngắn gọn 1-2 câu>", "unsupported_claims": ["<claim không được context hỗ trợ, nếu có>"]}}
"""

_ANSWER_RELEVANCY_PROMPT = """Bạn là một giám khảo (judge) chấm điểm hệ thống RAG. Nhiệm vụ: đánh giá xem CÂU TRẢ LỜI có thực sự liên quan và giải quyết đúng CÂU HỎI hay không (answer relevancy).

Quy tắc chấm điểm:
- Điểm cao nếu câu trả lời đi thẳng vào vấn đề được hỏi, không lan man, không thiếu ý chính.
- Nếu CÂU HỎI được kỳ vọng là không có câu trả lời trong tài liệu (is_negative_expected=true) và câu trả lời đã từ chối một cách rõ ràng, đúng cách, đó là relevancy CAO (score gần 1.0) — vì từ chối đúng là hành vi mong muốn.
- Điểm thấp nếu câu trả lời lạc đề, chung chung, hoặc trả lời sai câu hỏi khác.

CÂU HỎI:
{question}

is_negative_expected: {is_negative_expected}

CÂU TRẢ LỜI CẦN CHẤM:
{answer}

Trả về JSON với đúng các key sau:
{{"score": <float 0.0-1.0>, "reasoning": "<giải thích ngắn gọn 1-2 câu>"}}
"""


def _parse_judge_json(raw_text: str) -> dict:
    """Bóc tách + parse JSON từ output judge (dùng chung cho Gemini và Claude).
    Chuẩn hóa score string→float. Trả về dict an-toàn nếu parse hỏng."""
    if not raw_text:
        return {"score": None, "reasoning": "Judge trả về rỗng."}
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    json_str = match.group(0) if match else raw_text
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Không parse được JSON từ judge output: {raw_text[:200]}")
        return {"score": None, "reasoning": "Không parse được output của judge.", "raw": raw_text}

    # Một số model trả score dạng string ("0.8") thay vì float.
    if isinstance(parsed.get("score"), str):
        try:
            parsed["score"] = float(parsed["score"])
        except ValueError:
            parsed["score"] = None
    return parsed


def _call_judge_gemini(api_key: str, model_name: str, prompt: str, max_retries: int = 4) -> dict:
    """
    Gọi Gemini làm judge, ép JSON output qua response_mime_type (constrained decoding
    ở tầng API). Có giãn cách chủ động (_rate_limit_wait) + retry backoff nếu dính 429.
    """
    from google import genai  # import lazy — chỉ cần khi thực sự dùng judge Gemini
    from google.genai import types

    raw_text = None
    for attempt in range(max_retries):
        _rate_limit_wait()
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.candidates[0].content.parts[0].text.strip()
            break
        except Exception as exc:
            if attempt == max_retries - 1:
                logger.error(f"Judge (Gemini) call thất bại sau {max_retries} lần thử: {exc}")
                return {"score": None, "reasoning": f"Judge call lỗi sau {max_retries} lần thử: {exc}"}
            wait_time = 15 * (attempt + 1)
            logger.warning(
                f"Judge (Gemini) lỗi (có thể do rate limit 15 RPM free tier). "
                f"Thử lại sau {wait_time}s... (Lỗi: {exc})"
            )
            time.sleep(wait_time)

    return _parse_judge_json(raw_text)


def _call_judge_claude(api_key: str, model_name: str, prompt: str, max_retries: int = 3) -> dict:
    """
    Gọi Claude (Anthropic API) làm judge. SDK tự retry 429/5xx nên không cần vòng lặp
    thủ công. Haiku 4.5 là model pre-4.6 → KHÔNG truyền `thinking`. Yêu cầu JSON qua
    prompt + system, rồi parse bằng _parse_judge_json (cùng fallback với Gemini).
    """
    import anthropic  # import lazy — chỉ cần khi thực sự dùng judge Claude

    try:
        client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            temperature=0.0,
            system="Bạn là giám khảo chấm điểm hệ thống RAG. CHỈ trả về một JSON hợp lệ, không kèm text nào khác.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as exc:
        logger.error(f"Judge (Claude) call thất bại: {exc}")
        return {"score": None, "reasoning": f"Judge (Claude) call lỗi: {exc}"}

    return _parse_judge_json(raw_text)


def _call_judge(provider: str, api_key: str, model_name: str, prompt: str) -> dict:
    """Điều phối judge theo provider ("gemini" | "claude")."""
    if provider == "claude":
        return _call_judge_claude(api_key, model_name, prompt)
    return _call_judge_gemini(api_key, model_name, prompt)


def score_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    api_key: str,
    model_name: str = GEMINI_JUDGE_MODEL_DEFAULT,
    provider: str = "gemini",
) -> dict:
    """
    Đo mức độ câu trả lời bám sát context (không hallucinate).
    Trả về {"score": float | None, "reasoning": str, "unsupported_claims": list}
    """
    context_str = "\n\n---\n\n".join(contexts) if contexts else "(không có context nào được retrieve)"
    prompt = _FAITHFULNESS_PROMPT.format(question=question, context=context_str, answer=answer)
    return _call_judge(provider, api_key, model_name, prompt)


def score_answer_relevancy(
    question: str,
    answer: str,
    api_key: str,
    is_negative_expected: bool = False,
    model_name: str = GEMINI_JUDGE_MODEL_DEFAULT,
    provider: str = "gemini",
) -> dict:
    """
    Đo mức độ câu trả lời có thực sự giải quyết đúng câu hỏi.
    Trả về {"score": float | None, "reasoning": str}
    """
    prompt = _ANSWER_RELEVANCY_PROMPT.format(
        question=question,
        is_negative_expected=str(is_negative_expected).lower(),
        answer=answer,
    )
    return _call_judge(provider, api_key, model_name, prompt)
