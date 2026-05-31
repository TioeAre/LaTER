import os

import re
import asyncio
from typing import Tuple, List, Dict, Optional
from loguru import logger
from openai import AsyncOpenAI


def extract_content_outside_think(text):
    """split think and response

    Parameters
    ----------
    text : _type_
        origin response

    Returns
    -------
    _type_
        reasoning content, result
    """
    inside_pattern = r"<think>(.*?)</think>"
    inside_content = re.findall(inside_pattern, text, flags=re.DOTALL)
    outside_pattern = r"<think>.*?</think>"
    outside_content_raw = re.split(outside_pattern, text, flags=re.DOTALL)
    cleaned_parts = (part.strip() for part in outside_content_raw if part and part.strip())
    outside_content_string = " ".join(cleaned_parts)
    return inside_content, outside_content_string


def _build_judge_messages(question, predict_answer, ground_truth, answer_type=None) -> List[Dict[str, str]]:
    prompt = """
You are required to determine if a predicted answer is correct or can reasonably answer the question compared to the ground truth. The question will be placed within <question></question> tags, answer type will be placed within <type></type> tags, predicted answer will be placed within <predict></predict> tags, and the ground truth answer will be placed within <gt></gt> tags.

You must output the final score as a floating-point number (0.0 to 1.0) enclosed within `<answer>` tags.

Output Format: Output **only** the final numerical score inside the tags. Do not provide reasoning.

Example:
<answer>1.0</answer>

Question: <question>{{question}}</question>

Answer Type: <type>{{answer_type}}</type>

Predict Answer: <predict>{{sys_ans}}</predict>

Ground Truth: <gt>{{ref_ans}}</gt>
"""
    cur_prompt = (
        prompt.replace("{{question}}", str(question))
        .replace("{{answer_type}}", str(answer_type or "general"))
        .replace("{{sys_ans}}", str(predict_answer))
        .replace("{{ref_ans}}", str(ground_truth))
    )
    return [
        {"role": "system", "content": "You are a helpful and objective evaluator."},
        {"role": "user", "content": cur_prompt},
    ]


async def _gpt_acc_with_client(
    client: AsyncOpenAI,
    question,
    predict_answer,
    ground_truth,
    answer_type=None,
) -> Tuple[float, str]:
    messages = _build_judge_messages(question, predict_answer, ground_truth, answer_type)
    response_content = ""
    try:
        completion = await client.chat.completions.create(
            model=os.getenv("JUDGE_MODEL", "Qwen3-32B"),
            messages=messages,  # ty:ignore[invalid-argument-type]
            max_tokens=int(os.getenv("STAGE1_MAX_TOKENS", 8192)),
        )
        _, response_content = extract_content_outside_think(str(completion.choices[0].message.content))
        match = re.search(r"<answer>\s*([\d\.]+)\s*</answer>", response_content)

        if match:
            try:
                score = float(match.group(1))
                if score > 1.0:
                    score = 1.0
                elif score < 0.0:
                    score = 0.0
            except ValueError:
                logger.warning(f"Failed to parse float from extracted content: {match.group(1)}")
                score = 0.0
        else:
            logger.warning(f"No <answer> tags found in Judge response: {response_content}")
            fallback_match = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", response_content)
            score = float(fallback_match.group(0)) if fallback_match else 0.0
    except Exception as e:
        logger.error(f"Error during LLM Judge evaluation: {e}")
        score = 0.0

    return score, response_content


async def gpt_acc_async(
    question,
    predict_answer,
    ground_truth,
    answer_type=None,
    client: Optional[AsyncOpenAI] = None,
    base_url: str = "",
) -> Tuple[float, str]:
    try:
        owned_client = client is None
        if base_url == "":
            base_url = os.getenv("OPENAI_BASE_URL", "")
        judge_client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=os.getenv("API_KEY", ""),
            timeout=3600,
        )
        result = await _gpt_acc_with_client(
            judge_client,
            question,
            predict_answer,
            ground_truth,
            answer_type,
        )
        if owned_client:
            close = getattr(judge_client, "close", None)
            if callable(close):
                await close()
        return result
    except Exception as e:
        logger.error(f"gpt_acc_async failed: {type(e).__name__}: {e}")
        return 0.0, ""


def gpt_acc(question, predict_answer, ground_truth, answer_type=None) -> Tuple[float, str]:
    return asyncio.run(gpt_acc_async(question, predict_answer, ground_truth, answer_type))


async def batch_gpt_acc_async(
    requests: List[Dict[str, str]],
    base_url: str = "",
    max_concurrency: int = 8,
) -> Dict[int, Tuple[float, str]]:
    """Run multiple gpt_acc calls concurrently with bounded concurrency."""
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    if base_url == "":
        base_url = os.getenv("OPENAI_BASE_URL", "")
    judge_client = AsyncOpenAI(
        base_url=base_url,
        api_key=os.getenv("API_KEY", ""),
        timeout=3600,
    )

    async def _run_one(idx: int, req: Dict[str, str]) -> Tuple[int, Tuple[float, str]]:
        async with semaphore:
            try:
                score, response = await gpt_acc_async(
                    req.get("question", ""),
                    req.get("predict_answer", ""),
                    req.get("ground_truth", ""),
                    req.get("answer_type", None),
                    client=judge_client,
                )
                return idx, (score, response)
            except Exception as e:
                logger.error(f"batch_gpt_acc_async request failed at idx={idx}: {type(e).__name__}: {e}")
                return idx, (0.0, "")

    try:
        tasks = [_run_one(i, req) for i, req in enumerate(requests)]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        results: Dict[int, Tuple[float, str]] = {}
        for i, item in enumerate(gathered):
            if isinstance(item, BaseException):
                logger.error(f"batch_gpt_acc_async gather exception at idx={i}: {type(item).__name__}: {item}")
                results[i] = (0.0, "")
                continue
            if not isinstance(item, tuple) or len(item) != 2:
                logger.error(f"batch_gpt_acc_async got unexpected gather item at idx={i}: {item}")
                results[i] = (0.0, "")
                continue
            idx, score_response = item
            results[idx] = score_response
        return results
    finally:
        close = getattr(judge_client, "close", None)
        if callable(close):
            await close()
