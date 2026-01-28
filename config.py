from enum import Enum
from openai import AsyncOpenAI

class GraphType(Enum):
    SUMMARIZE = "summarize"
    AGGREGATE = "aggregate"
    REFLECT = "reflect"
    DEBATE = "debate"
    LLM_AGG = "llm_agg"
    DEBATE_LLM_AGG = "debate_llm_agg"

    SELF_REFINE = "self_refine"
    HIERARCHICAL = "hierarchical"

class AgentType(Enum):
    PREDICTOR = "predictor"
    SUMMARIZER = "summarizer"
    AGGREGATOR = "aggregator"
    REFLECTOR = "reflector"
    DEBATOR = "debator"
    LLM_AGG = "llm_agg"
    MATH_EXPERT = "math_expert"
    MATH_ANALYST = "math_analyst"
    PROGRAMMING_EXPERT = "programming_expert"
    PROJECT_MANAGER = "project_manager"
    PROGRAMMER = "programmer"
    TEST_ANALYST = "test_analyst"
    BUG_FIXER = "bug_fixer"
    KNOWLEDGEABLE_EXPERT = "knowledgeable_expert"
    CRITIC = "critic"
    PSYCHOLOGIST = "psychologist"
    HISTORIAN = "historian"
    REFINER = "refiner"  
    MATH_AGENT = "math_agent"
    SCIENCE_AGENT = "science_agent"
    CODE_AGENT = "code_agent"
    TASK_SUMMARIZER = "task_summarizer"

class TaskType(Enum):
    MATH = "math"               
    MATH_CHOICE = "math_choice"    
    REASONING_CHOICE = "reasoning_choice"
    CODE = "code"              

DATASET_CONFIG = {
    "math": {
        "path": "dataset/math-500/math_500.jsonl",
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False, 
    },
    "aime": {
        "path": "dataset/aime-2025/aime2025.jsonl",
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "agi": {
        "path": "dataset/AGIEval_MATH/test.jsonl",
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "aqua": {
        "path": "dataset/aqua/aqua.jsonl",
        "task_type": TaskType.MATH_CHOICE,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "gpqa": {
        "path": "dataset/gpqa/gpqa_diamond.jsonl",
        "task_type": TaskType.REASONING_CHOICE,
        "problem_key": "question",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "mbpp": {
        "path": "dataset/mbpp/sanitized-mbpp.jsonl",
        "task_type": TaskType.CODE,
        "problem_key": "prompt",
        "answer_key": None,
        "default_use_judge": True,
    },
    "humaneval": {
        "path": "dataset/humaneval-et/HumanEval_ET.jsonl",
        "task_type": TaskType.CODE,
        "problem_key": "prompt",
        "answer_key": None,
        "default_use_judge": True,
    },
}

def create_main_client():
    return AsyncOpenAI(
        api_key="API_KEYS",
        base_url="url",
        default_headers={}
    )

def create_judge_client():
    return AsyncOpenAI(
        api_key="API_KEYS",
        base_url="url",
        default_headers={}
    )

def create_evaluator_client():
    return AsyncOpenAI(
        api_key="API_KEYS",
        base_url="url",
        default_headers={}
    )

aclient = create_main_client()
bclient = create_judge_client()


def get_default_use_judge(dataset: str) -> bool:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        return False
    return config.get("default_use_judge", False)
