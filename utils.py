import os
import random
import re, regex
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


# def extract_boxed_answer(text: str) -> Optional[str]:
#     boxes = regex.findall(r"\\boxed\{((?:[^{}]+|(?1))*)\}", text)
#     if boxes:
#         return boxes[-1].strip()


#     numbers = regex.findall(r"[-+]?\d+(?:\.\d+)?", text)
#     if numbers:
#         return numbers[-1]
#     return None
def extract_boxed_answer(text: str) -> Optional[str]:
    matches = list(re.finditer(r"\\boxed\{", text))
    valid_boxes = []
    for match in matches:
        start_idx = match.end()
        brace_count = 1
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            if brace_count == 0:
                valid_boxes.append(text[start_idx:i].strip())
                break
    if valid_boxes:
        return valid_boxes[-1]
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)
