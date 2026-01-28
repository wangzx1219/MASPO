"""
工具函数：答案提取、标准化、代码执行等
"""
import re
import html
import asyncio
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Optional
from collections import Counter

from config import TaskType

DS_API_CONCURRENCY_LIMIT = 60 
ds_semaphore = asyncio.Semaphore(DS_API_CONCURRENCY_LIMIT)
def async_retry(tries: int = 5, delay: float = 0.5, max_delay: float = 30):
    def deco(func):
        async def wrapper(*args, **kwargs):
            d = delay
            for i in range(1, tries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if i == tries:
                        raise
                    await asyncio.sleep(min(max_delay, d))
                    d *= 2
            return None
        return wrapper
    return deco

@async_retry()
async def async_call_llm(client, prompt: str, temperature: float = 0.0, 
                         max_tokens: int = 4096, use_ds_api: bool = False) -> str:
    # 1. 确定模型
    model = "Qwen3-8B" if not use_ds_api else "gemini-2.5-pro"
    
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    }
    
    if use_ds_api:
        async with ds_semaphore:
            resp = await client.chat.completions.create(** request_kwargs)
    else:
        resp = await client.chat.completions.create(** request_kwargs)
    
    first_choice = resp.choices[0]

    
    return first_choice.message.content

def extract_answer(raw: str) -> str:
    raw = raw.strip()
    for _ in range(3):
        new_raw = html.unescape(raw)
        if new_raw == raw:
            break
        raw = new_raw

    matches = re.findall(r"<answer>(.*?)</answer>", raw, re.S)
    if matches:
        return matches[-1].strip()

    box_pat = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.S)
    m = box_pat.search(raw)
    if m:
        return m.group(1).strip()

    sentences = re.split(r'[。\n;]+', raw)
    last = sentences[-1].strip()
    return last[-30:] if len(last) > 30 else last

def extract_code(raw: str) -> str:
    raw = raw.strip()

    candidates = []
    seen = set()

    python_pattern = r'```\s*python\s*(.*?)\s*```'
    for m in re.findall(python_pattern, raw, flags=re.DOTALL | re.IGNORECASE):
        code = m.strip()
        if code and code not in seen:
            seen.add(code)
            candidates.append(code)

    general_pattern = r'```\s*(.*?)\s*```'
    for m in re.findall(general_pattern, raw, flags=re.DOTALL):
        code = m.strip()
        if code and code not in seen:
            code = re.sub(r'^python\s*', '', code, flags=re.IGNORECASE).strip()
            seen.add(code)
            candidates.append(code)

    code_tag_pattern = r'<code>(.*?)</code>'
    for m in re.findall(code_tag_pattern, raw, flags=re.DOTALL | re.IGNORECASE):
        code = m.strip()
        if code and code not in seen:
            seen.add(code)
            candidates.append(code)

    for code in reversed(candidates):
        if 'def ' in code:
            return code

    if 'def ' in raw:
        def_pattern = r'def\s+[\w_]+\(.*?\):\s*\n?(.|\n)*?(?=\n\s*\n|```|\Z)'
        def_match = re.search(def_pattern, raw, flags=re.IGNORECASE)
        if def_match:
            return def_match.group(0).strip()

    return ''

def extract_output(raw: str, task_type: TaskType) -> str:
    if task_type == TaskType.CODE:
        return extract_code(raw)
    else:
        return extract_answer(raw)

def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r'\$(.*?)\$', r'\1', answer)
    answer = re.sub(r'\\\[(.*?)\\\]', r'\1', answer, flags=re.S)
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\boxed\s*{((?:[^{}]|{[^}]*})*?)}', r'\1', answer)
    answer = re.sub(r'\\\((.*?)\\\)', r'\1', answer)
    answer = re.sub(r'\\?°', '', answer)
    answer = re.sub(r'\^?\\?circ', '', answer)
    answer = re.sub(r'\s+', '', answer)
    answer = re.sub(r'\\sqrt\s*{([^}]*)}', r'sqrt(\1)', answer)
    answer = re.sub(r'√(\d+)', r'sqrt(\1)', answer)
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)
    answer = re.sub(r'\\dfrac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)
    answer = re.sub(r'\\pi', 'π', answer)
    answer = re.sub(r'\\left|\\right', '', answer)
    answer = answer.replace('[', '(').replace(']', ')')
    if re.fullmatch(r'[,\s\-0-9]+', answer):
        nums = [int(x) for x in re.findall(r'-?\d+', answer)]
        return ','.join(map(str, sorted(nums)))
    return answer.lower()

def execute_code_with_tests(code: str, test_list: List[str], timeout: int = 5) -> Dict[str, Any]:
    
    required_imports = set()
    for test in test_list:
        if 'math.' in test:
            required_imports.add('import math')
    
    existing_imports = set()
    for imp in required_imports:
        if imp in code:
            existing_imports.add(imp)
    
    missing_imports = required_imports - existing_imports
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        temp_file = f.name
        
        for imp in sorted(missing_imports):
            f.write(imp + '\n')
        
        if missing_imports:
            f.write('\n')
        
        f.write(code + '\n\n')
        
        for test in test_list:
            f.write(test + '\n')
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        os.unlink(temp_file)
        
        if result.returncode == 0:
            return {"success": True, "passed": len(test_list), "total": len(test_list), "errors": []}
        else:
            return {"success": False, "passed": 0, "total": len(test_list), "errors": [result.stderr]}
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return {"success": False, "passed": 0, "total": len(test_list), "errors": ["Execution timeout"]}
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return {"success": False, "passed": 0, "total": len(test_list), "errors": [str(e)]}

def extract_function_name_from_tests(test_list: List[str]) -> Optional[str]:
    BUILTIN_SKIP = {
        'assert', 'math', 'set', 'list', 'dict', 'tuple', 'str', 'int', 'float',
        'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum',
        'max', 'min', 'abs', 'round', 'all', 'any', 'isinstance', 'type',
        'print', 'input', 'open', 'isclose', 'sqrt', 'pow', 'ceil', 'floor'
    }
    
    function_candidates = []
    
    for test in test_list:
        matches = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', test)
        for match in matches:
            if match not in BUILTIN_SKIP:
                function_candidates.append(match)
    
    if not function_candidates:
        return None
    
    most_common = Counter(function_candidates).most_common(1)
    return most_common[0][0] if most_common else None

def majority_vote(answers: List[str]) -> str:
    from collections import defaultdict
    
    if not answers:
        return ""
    
    counts = defaultdict(int)
    for a in answers:
        if a:
            normalized = normalize_answer(a)
            counts[normalized] += 1
    
    if not counts:
        return answers[0] if answers else ""
    
    return max(counts.items(), key=lambda x: x[1])[0]

def code_vote(codes: List[str]) -> str:
    def normalize_code(code: str) -> str:
        lines = code.split('\n')
        normalized = []
        for line in lines:
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if line:
                normalized.append(line)
        return '\n'.join(normalized)
    
    if not codes:
        return ""
    
    normalized_codes = [normalize_code(c) for c in codes if c]
    
    if not normalized_codes:
        return codes[0] if codes else ""
    
    code_hashes = [hash(nc) for nc in normalized_codes]
    most_common_hash = Counter(code_hashes).most_common(1)[0][0]
    
    for i, h in enumerate(code_hashes):
        if h == most_common_hash:
            return codes[i]
    
    return codes[0]

def parse_comparison_result(raw: str) -> bool:
    if "<choose>" in raw and "</choose>" in raw:
        try:
            choice = raw.split("<choose>")[1].split("</choose>")[0]
            return "A" in choice
        except:
            pass
    for c in reversed(raw.strip().upper()):
        if c.isalpha():
            return c == 'A'
    return True
