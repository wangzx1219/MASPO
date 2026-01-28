from config import AgentType, TaskType

# ------------------ Agent角色描述 ------------------
ROLE_DESCRIPTIONS = {
    AgentType.PREDICTOR: {
        TaskType.MATH: "Primary solver that provides step-by-step mathematical reasoning and final answer.",
        TaskType.MATH_CHOICE: "Primary solver that provides step-by-step mathematical reasoning and final answer.",
        TaskType.REASONING_CHOICE: "Primary solver that provides step-by-step reasoning and final answer.",
        TaskType.CODE: "Primary solver that provides step-by-step reasoning and final answer.",
    },
    AgentType.SUMMARIZER: "Extractor of relevant information from context to assist problem solving.",
    AgentType.REFLECTOR: "Critical reviewer that identifies errors in previous solutions and provides corrected reasoning.",
    AgentType.DEBATOR: "Evaluator of multiple solutions that synthesizes the best approach from competing answers.",
    AgentType.LLM_AGG: "Integrator that compares multiple candidate solutions and produces a single authoritative answer.",
    AgentType.AGGREGATOR: "Voting mechanism that selects the most frequent answer from multiple predictors.",
    # AgentDropout 数学题角色
    AgentType.MATH_EXPERT: "You are a math expert.",
    AgentType.MATH_ANALYST: "You are a mathematical analyst.",
    AgentType.PROGRAMMING_EXPERT: "You are a programming expert.",
    # AgentDropout 代码生成角色
    AgentType.PROJECT_MANAGER: "You are a project manager.",
    AgentType.PROGRAMMER: "You are a programming expert.",
    AgentType.TEST_ANALYST: "You are a test analyst.",
    AgentType.BUG_FIXER: "You are a bug fixer.",

    AgentType.KNOWLEDGEABLE_EXPERT: "You are a knowledgeable expert in question answering.",
    AgentType.CRITIC: "You are an excellent critic.",
    AgentType.PSYCHOLOGIST: "You are a psychologist.",
    AgentType.HISTORIAN: "You research and analyze cultural, economic, political, and social events in the past.",
}

def get_role_description(agent_type: AgentType, task_type: TaskType = None) -> str:
    desc = ROLE_DESCRIPTIONS.get(agent_type, "General reasoning agent")
    if isinstance(desc, dict) and task_type:
        return desc.get(task_type, list(desc.values())[0])
    return desc if isinstance(desc, str) else list(desc.values())[0]

AGENT_TEMPLATES = {
    TaskType.MATH: {
        AgentType.PREDICTOR: (
            "Let's think step by step. Show your final answer bracketed between <answer> and </answer> tags.\n"
            "{context}\n"
            "Question: {question}\nReasoning:"
        ),
        AgentType.SUMMARIZER: (
            "Based on the math problem, retrieve relevant information from the provided context that is ONLY "
            "helpful in solving the problem. Do not repeat irrelevant context. Start with 'Summary: '.\n"
            "Question: {question}\nContext: {context}\nSummary:"
        ),
        AgentType.REFLECTOR: (
            "Please review the solution above and criticize where it might be wrong. "
            "Show your final answer bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolution: {context}\nFeedback: indicating the reflection of the solution given the question.\nCorrect answer:"
        ),
        AgentType.DEBATOR: (
            "These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer. "
            "Show your final answer bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolutions: {context}\nAnswer:"
        ),
        AgentType.LLM_AGG: (
            "You are given several candidate solutions to the following math problem. "
            "Carefully compare their reasoning and answers, then produce a single final solution. "
            "Show your final answer bracketed between <answer> and </answer> tags.\n"
            "Question: {question}\n\n"
            "Candidate solutions:\n{context}\n"
            "Answer:"
        ),
    },
    
    TaskType.MATH_CHOICE: {
        AgentType.PREDICTOR: (
            "Let's think step by step. Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
            "{context}\n"
            "Question: {question}\nReasoning:"
        ),
        AgentType.SUMMARIZER: (
            "Based on the math problem, retrieve relevant information from the provided context that is ONLY "
            "helpful in solving the problem. Do not repeat irrelevant context. Start with 'Summary: '.\n"
            "Question: {question}\nContext: {context}\nSummary:"
        ),
        AgentType.REFLECTOR: (
            "Please review the solution above and criticize where it might be wrong. "
            "Show your final option with one letter bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolution: {context}\nFeedback: indicating the reflection of the solution given the question.\nCorrect answer:"
        ),
        AgentType.DEBATOR: (
            "These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer. "
            "Show your final option with one letter bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolutions: {context}\nAnswer:"
        ),
        AgentType.LLM_AGG: (
            "You are given several candidate solutions to the following math problem. "
            "Carefully compare their reasoning and answers, then produce a single final solution. "
            "Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
            "Question: {question}\n\n"
            "Candidate solutions:\n{context}\n"
            "Answer:"
        ),
    },
    
    TaskType.REASONING_CHOICE: {
        AgentType.PREDICTOR: (
            "Let's think step by step. Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
            "{context}\n"
            "Question: {question}\nReasoning:"
        ),
        AgentType.SUMMARIZER: (
            "Based on the multi-choice problem, retrieve relevant information from the provided context that is ONLY "
            "helpful in solving the problem. Do not repeat irrelevant context. Start with 'Summary: '.\n"
            "Question: {question}\nContext: {context}\nSummary:"
        ),
        AgentType.REFLECTOR: (
            "Please review the solution above and criticize where it might be wrong. "
            "Show your final option with one letter bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolution: {context}\nFeedback: indicating the reflection of the solution given the question.\nCorrect answer:"
        ),
        AgentType.DEBATOR: (
            "These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer. "
            "Show your final option with one letter bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolutions: {context}\nAnswer:"
        ),
        AgentType.LLM_AGG: (
            "You are given several candidate solutions to the following multi-choice problem. "
            "Carefully compare their reasoning and answers, then produce a single final solution. "
            "Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
            "Question: {question}\n\n"
            "Candidate solutions:\n{context}\n"
            "Answer:"
        ),
    },
    
    TaskType.CODE: {
        AgentType.PREDICTOR: (
            "You are an expert Python programmer. Write clean, efficient, and well-documented code.\n"
            "Problem: {question}\n"
            "{context}\n"
            "Provide a complete and correct code implementation in python.\n"
            "Code:"
        ),
        AgentType.SUMMARIZER: (
            "Based on the programming problem, extract key requirements and constraints.\n"
            "Problem: {question}\n"
            "Context: {context}\n"
            "Summary of key requirements:"
        ),
        AgentType.REFLECTOR: (
            "Review the following Python code and identify potential bugs or improvements.\n"
            "Problem: {question}\n"
            "Code: {context}\n"
            "Provide an improved version in python at the end of your response.\n"
            "Improved Code:"
        ),
        AgentType.DEBATOR: (
            "Compare these Python solutions and synthesize the best approach.\n"
            "Problem: {question}\n"
            "Solutions:\n{context}\n"
            "Provide the best solution in python at the end of your response.\n"
            "Final Code:"
        ),
        AgentType.LLM_AGG: (
            "Analyze several Python solutions and produce the most robust one.\n"
            "Problem: {question}\n\n"
            "Candidate solutions:\n{context}\n"
            "Provide your final solution in python at the end of your response.\n"
            "Final Code:"
        ),
    },
}

def get_agent_template(task_type: TaskType, agent_type: AgentType) -> str:
    """获取Agent模板"""
    return AGENT_TEMPLATES.get(task_type, {}).get(agent_type)

COMPRESS_PROMPTS = {
    TaskType.MATH: (
        "Below is a solution to a math problem. "
        "Summarize the key reasoning steps and the final answer in at most 30 words.\n\n"
        "{raw}\n\nSummary:"
    ),
    TaskType.MATH_CHOICE: (
        "Below is a solution to a math problem. "
        "Summarize the key reasoning steps and the final answer in at most 30 words.\n\n"
        "{raw}\n\nSummary:"
    ),
    TaskType.REASONING_CHOICE: (
        "Below is a solution to a multi-choice problem. "
        "Summarize the key reasoning steps and the final answer in at most 30 words.\n\n"
        "{raw}\n\nSummary:"
    ),
    TaskType.CODE: (
        "Summarize the key logic of this Python code in at most 50 words:\n\n"
        "{raw}\n\nSummary:"
    ),
}

PROMPT_OPTIMIZE_TEMPLATE = {
    TaskType.MATH: """
You are optimizing a prompt for a specific agent in a multi-agent mathematical reasoning system.
CRITICAL: The agent's core role and responsibilities MUST be preserved in the optimized prompt.

Agent Type: {agent_type}
Current System Role: {role_description}

Sample Execution Traces (Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```
Provide your analysis, optimization points, and the complete optimized prompt using the following XML format:
<analyse>Analyse what drawbacks exist in the results produced by the reference prompt and how to improve them.</analyse>
<modification>One sentence summary of the key improvement</modification>
<prompt>Provide the complete optimized prompt</prompt>
""",
    
    TaskType.CODE: """
You are optimizing a prompt for a specific agent in a multi-agent code generation system.
CRITICAL: The agent's core role and responsibilities MUST be preserved in the optimized prompt.
Agent Type: {agent_type}
Current System Role: {role_description}
Sample Execution Traces (Problem + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

Your Task:
1. Analyze the "Agent Output" in the samples above.
2. Identify specific **coding errors** (e.g., off-by-one errors, wrong library usage, syntax errors) or **format violations**.
3. Determine if the current prompt is too vague, causing these errors.
4. Propose a new prompt that explicitly instructs the agent to avoid these specific pitfalls.

Format your response exactly as follows:
<analyse>Detailed analysis of the bugs found in the sample outputs and how the prompt failed to prevent them.</analyse>
<modification>One sentence summary of the specific instruction added/changed.</modification>
<prompt>The complete, optimized prompt (keep it under 300 words). Ensure it retains the core task description.</prompt>
""",

    TaskType.MATH_CHOICE: """
You are optimizing a prompt for a specific agent in a multi-agent mathematical reasoning system.
CRITICAL: The agent's core role and responsibilities MUST be preserved in the optimized prompt.

Agent Type: {agent_type}
Current System Role: {role_description}

Sample Execution Traces (Question + Context + Agent Output)
```
{samples}
```

Requirements:
```
{requirements}
```

Reference prompt:
```
{prompt}
```

Provide your analysis, optimization points, and the complete optimized prompt using the following XML format:
<analyse>Analyse what drawbacks exist in the results produced by the reference prompt and how to improve them.</analyse>
<modification>One sentence summary of the key improvement</modification>
<prompt>Provide the complete optimized prompt</prompt>
""",

    TaskType.REASONING_CHOICE: """
You are optimizing a prompt for a specific agent in a multi-agent problem-solving system.
CRITICAL: The agent's core role and responsibilities MUST be preserved in the optimized prompt.

Agent Type: {agent_type}
Current System Role: {role_description}

Sample Execution Traces (Question + Context + Agent Output)
{samples}

Requirements:
{requirements}

Reference prompt:
{prompt}

### Analysis Focus:
1. **Truncation Check:** Are outputs cut off? If yes, the prompt is too demanding.
2. **Reasoning Quality:** Are the answers correct? If not, what's missing in the guidance?
3. **Efficiency Check:** Is the prompt asking for unnecessary steps that don't improve accuracy?
### Optimization Principles:
**DO:**
- Give clear, actionable guidance in 2-3 sentences
- Focus on WHAT to think about, not HOW to format
- The prompt must be CONCISE (under 250 words) and DOMAIN-AGNOSTIC
**DON'T:**
- Create multi-section templates with headers
- Require "verify each step" or "solve independently first"
- Add meta-instructions about the reasoning process
- Use bullet-point structures that the model must fill

Provide your analysis, optimization points, and the complete optimized prompt using the following XML format:
<analyse>Analyse what drawbacks exist in the results produced by the reference prompt and how to improve them.</analyse>
<modification>One sentence summary of the key improvement</modification>
<prompt>Provide the complete optimized prompt</prompt>
""",
}

ANSWER_EVALUATE_TEMPLATE = {
    TaskType.MATH: """
You are evaluating two outputs (A and B) from an agent of type {agent_type}.
Based on the input, requirements and the agent's role, evaluate the two responses, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Input to the Agent
Question: {question}

# Requirement
{requirement}

# A
{Answer_A}

# B
{Answer_B}

Guidelines:
- If one contains a clearly correct final answer and the other does not, choose the correct one — even if its explanation is shorter.
- If both answers appear correct, prefer the one with clearer, logically sound reasoning.
- Do NOT favor verbose, or confident-sounding outputs if they are wrong.
- Pay attention to the format of final answer (<answer>...</answer>).

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Some analysis</analyse>
<choose>A/B</choose>
""",
    
    TaskType.MATH_CHOICE: """
You are evaluating two outputs (A and B) from an agent of type {agent_type}.
Based on the input, requirements and the agent's role, evaluate the two responses, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Input to the Agent
Question: {question}

# Requirement
{requirement}

# A
{Answer_A}

# B
{Answer_B}

Guidelines:
- If one contains a clearly correct final answer and the other does not, choose the correct one — even if its explanation is shorter.
- If both answers appear correct, prefer the one with clearer, logically sound reasoning.
- Do NOT favor verbose, or confident-sounding outputs if they are wrong.
- Pay attention to the format of final answer (only one letter between <answer> and </answer> representing the correct option).

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Some analysis</analyse>
<choose>A/B</choose>
""",
    
    TaskType.CODE: """
You are evaluating two code outputs (A and B) from an agent of type {agent_type}.
Based on the problem requirements and the agent's role, evaluate the two code solutions, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Programming Problem
{question}

# Requirements
{requirement}

# Code Solution A
{Answer_A}

# Code Solution B
{Answer_B}

Guidelines:
- If one solution's code is syntactically correct and passes all possible test cases while the other doesn't, choose the correct one.
- If both solutions' code appear correct, prefer the one with:
  * Better code quality (readability, maintainability)
  * More efficient algorithm or implementation
  * Clearer variable names and comments
  * Clearer reasoning before final code
- Do NOT favor verbose or over-engineered solutions if a simpler one is correct.
- Pay attention to whether the code is properly enclosed in ```python ... ``` blocks.

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Some analysis focusing on correctness, code quality, and efficiency</analyse>
<choose>A/B</choose>
""",

    TaskType.REASONING_CHOICE: """
You are an expert judge evaluating two reasoning outputs (A and B) from an agent of type {agent_type}.
Based on the input, requirements and the agent's role, evaluate the two responses, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Input to the Agent
Question: {question}

# Requirement
{requirement}

# A
{Answer_A}

# B
{Answer_B}

### Evaluation Criteria (Priority Order):
1. **Completeness:** Is the output truncated? Complete outputs are always preferred.
2. **Correctness:** Is the reasoning sound and the answer likely correct?
3. **Efficiency:** 
   - Concise, focused reasoning is BETTER than verbose, over-structured reasoning.
   - Unnecessary section headers and formatting reduce quality.
   - The goal is correct answers, not beautiful formatting.
### Decision:
- If one is truncated, choose the complete one.
- If both complete, choose the one with better reasoning (not better formatting).
- Penalize outputs that waste tokens on structure instead of thinking.

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>
Compare Reasoning (if both complete).
</analyse>
<choose>A/B</choose>
""",
}

INTERMEDIATE_COMPARE_TEMPLATE = {
    TaskType.MATH: """
You are comparing two intermediate outputs from an agent in a multi-agent mathematical reasoning system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FINAL ANSWER.

Prefer the output that:
- Contains mathematically accurate reasoning (no factual/logical errors)
- Clearly states intermediate results or assumptions
- Avoids misleading statements or ambiguous conclusions
- Provides enough detail for downstream agents to verify or build upon

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct final answer? Respond ONLY with "A" or "B".
""",
    
    TaskType.CODE: """
You are comparing two outputs from a 'Predictor' agent in a code generation pipeline.
The next agent in the pipeline is a 'Reflector' or 'Debugger' that will try to fix any errors.

Your goal is to select the output that provides the **best starting point** for the Reflector.

Criteria for selection (in order of importance):
1. **Correctness**: If one output appears to be functionally correct python code while the other has logic errors, choose the correct one.
2. **Adherence to Requirements**: The code must NOT contain exception handling, input validation, or type hints if the prompt forbids them. It must be a standalone function.
3. **Format**: The code must be easily extractable (e.g., inside ```python``` blocks).
4. **Logic Clarity**: If both have bugs, choose the one with the clearer algorithmic logic that is easier to debug.

Problem: {question}

Output A:
{output_a}

Output B:
{output_b}

Which output is better? Respond ONLY with "A" or "B".
""",

    TaskType.MATH_CHOICE: """
You are comparing two intermediate outputs from an agent in a multi-agent mathematical reasoning system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FINAL ANSWER.

Prefer the output that:
- Contains mathematically accurate reasoning (no factual/logical errors)
- Clearly states intermediate results or assumptions
- Avoids misleading statements or ambiguous conclusions
- Provides enough detail for downstream agents to verify or build upon

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct final answer? Respond ONLY with "A" or "B".
""",

    TaskType.REASONING_CHOICE: """
You are comparing two intermediate outputs from an agent in a multi-agent problem-solving reasoning system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FINAL ANSWER.

### Evaluation Priority:
**1. Completeness:**
- Is the output truncated (ends mid-sentence, incomplete thought)?
- A truncated output cannot effectively help downstream agents.
- If one is truncated and one is not, select the complete one.
**2. Reasoning Correctness (If both complete):**
- Are scientific facts and principles correctly stated?
- Is the logical reasoning valid?
- Are option eliminations properly justified?
**3. Usefulness for Next Agent:**
- Does it provide clear intermediate conclusions?
- Can the next agent build upon this reasoning?
### Decision Rules:
- Truncated output = automatic disqualification (unless both truncated)
- If both complete: prioritize scientific accuracy over verbosity
- If both truncated: choose the one with more valid reasoning content

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct final answer? Respond ONLY with "A" or "B".
""",
}


FINAL_ANSWER_COMPARE_TEMPLATE = {
    TaskType.MATH: """
You are comparing two FINAL answers to a math problem.
Your SOLE task is to determine which answer is more likely to be **mathematically correct**.

Do NOT favor longer, more detailed, or better-formatted responses unless they are also correct.
If one answer is clearly correct and the other is wrong, choose the correct one—even if its reasoning is minimal.
If both seem plausible, prefer the one with clearer, error-free reasoning.

Problem: {question}
Requirement: {requirement}

Answer A:
{Answer_A}

Answer B:
{Answer_B}

Respond with only "A" or "B".
""",
    
    TaskType.CODE: """
You are comparing two FINAL Python code solutions.
Your SOLE criterion is **Functional Correctness**.

Problem: {question}
Requirement: {requirement}

Answer A:
{Answer_A}

Answer B:
{Answer_B}

Evaluation Steps:
1. Mentally trace both codes with edge case inputs.
2. If Answer A is correct and Answer B is buggy (infinite loop, wrong logic, syntax error), choose A.
3. If Answer B is correct and Answer A is buggy, choose B.
4. If both are correct, choose the one that is **more concise** and follows standard Pythonic practices.
5. If both are buggy, choose the one that is "less broken" (closer to the solution).

Respond with only "A" or "B".
""",

    TaskType.MATH_CHOICE: """
You are comparing two FINAL answers to a math problem.
Your SOLE task is to determine which answer is more likely to be **mathematically correct**.

Do NOT favor longer, more detailed, or better-formatted responses unless they are also correct.
If one answer is clearly correct and the other is wrong, choose the correct one—even if its reasoning is minimal.
If both seem plausible, prefer the one with clearer, error-free reasoning.

Problem: {question}
Requirement: {requirement}

Answer A:
{Answer_A}

Answer B:
{Answer_B}

Respond with only "A" or "B".
""",

    TaskType.REASONING_CHOICE: """
You are comparing two FINAL answers to a problem.
Your SOLE task is to determine which answer is more likely to be **correct**.

### Evaluation Process:
**Step 1: Check Completeness**
- Does each output have a clear final answer (preferably in <answer>X</answer> format)?
- If one is truncated/incomplete and the other is complete, choose the complete one.
**Step 2: Extract and Compare Answers (if both complete)**
- Identify the final letter choice from each output
- If answers are the same, choose the one with better reasoning
- If answers differ, proceed to Step 3
**Step 3: Evaluate Reasoning Quality (if answers differ)**
- Which output correctly applies relevant scientific principles?
- Which output properly eliminates incorrect options?
- Which reasoning chain is more logically sound?
**Step 4: Make Decision**
Priority: Complete > Truncated
Among complete outputs: Correct reasoning > Incorrect reasoning > Better format

Problem: {question}
Requirement: {requirement}

Answer A:
{Answer_A}

Answer B:
{Answer_B}

Respond with only "A" or "B".
""",
}


OPTIMIZATION_REQUIREMENTS = {
    TaskType.MATH: (
        "The agent must produce a clear, step-by-step reasoning process. "
        "Crucially, the **final answer must appear ONLY once**, enclosed strictly between <answer> and </answer> tags, "
        "with **no additional text, explanation, unit, or reasoning inside these tags**. "
        "The content within `<answer>...</answer>` must be the minimal, canonical mathematical answer."
    ),
    TaskType.MATH_CHOICE: (
        "The agent must produce a clear, step-by-step reasoning process. "
        "Crucially, the **final answer must appear ONLY once**, enclosed strictly between <answer> and </answer> tags, "
        "with **no additional text, explanation, unit, or reasoning inside these tags**. "
        "The content within `<answer>...</answer>` must be only one letter representing the correct option among the given option."
    ),
    TaskType.REASONING_CHOICE: (
        "Crucially, the **final answer must appear ONLY once**, enclosed strictly between <answer> and </answer> tags, "
        "with **no additional text, explanation, unit, or reasoning inside these tags**. "
        "The content within `<answer>...</answer>` must be only one letter representing the correct option among the given option."
    ),
    TaskType.CODE: (
        "The agent must produce a clear, step-by-step reasoning process followed by syntactically correct Python code that passes all possible test cases. "
        "The format of code must be like ```python\ndef function_name(input_arguments): \n#code here\nreturn output```without any exception detection, input validation, type annotations, or edge-case handling. "
    ),
}

AGENT_DROPOUT_MATH_PROMPTS = {
    "0": (
        "You are a math expert. "
        "You will be given a multiple-choice question and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final choice with only a capital letter, for example: <answer>A</answer>\n"
        "Question: {question}\n"
        "{context}"
    ),
    "1": (
        "You are a mathematical analyst. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results. "
        "The last line of your output contains only the final choice with only a capital letter, for example: <answer>A</answer>\n"
        "Question: {question}\n"
        "{context}"
    ),
    "2": (
        "You are a programming expert. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve multiple-choice question. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the answer variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n    x = 10\n    y = 20\n    return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "3": (
        "You are a math expert. "
        "You will be given a multiple-choice question and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final choice with only a capital letter, for example: <answer>A</answer>\n"
        "Question: {question}\n"
        "{context}"
    ),
}


AGENT_DROPOUT_MATH_OPEN_PROMPTS = {
    "0": (
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "Show your final answer bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "1": (
        "You are a mathematical analyst. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results. "
        "Show your final answer bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "2": (
        "You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve the math problem. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the answer variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n    x = 10\n    y = 20\n    return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "3": (
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "Show your final answer bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
}

AGENT_DROPOUT_CODE_PROMPTS = {
    "0": (
        "You are a project manager. "
        "You will be given a function signature and its docstring by the user. "
        "You are responsible for overseeing the overall structure of the code, ensuring that the code is structured to complete the task. Implement code concisely and correctly without pursuing over-engineering. "
        "You need to suggest optimal design patterns to ensure that the code follows best practices for maintainability and flexibility. "
        "You can specify the overall design of the code, including the classes that need to be defined (maybe none) and the functions used (maybe only one function). "
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points.\n"
        "Problem: {question}\n"
        "{context}"
    ),
    "1": (
        "You are a programming expert. "
        "You will be given a function signature and its docstring by the user. "
        "You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```\n"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks. "
        "Please think step by step.\n"
        "Problem: {question}\n"
        "{context}"
    ),
    "2": (
        "You are a test analyst. "
        "You will be given a function signature and its docstring by the user. "
        "You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. "
        "You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. "
        "You can point out any potential errors in the code. "
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points.\n"
        "Problem: {question}\n"
        "{context}"
    ),
    "3": (
        "You are a bug fixer. "
        "You will be given a function signature and its docstring by the user. "
        "You need to provide modified and improved python code based on the current overall code design, algorithm framework, code implementation or test problems. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```\n"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks.\n"
        "Problem: {question}\n"
        "{context}"
    ),
}

AGENT_DROPOUT_REASONING_PROMPTS = {
    "0": (
        "You are a knowledgeable expert in question answering. "
        "Please give several key entities that need to be searched in wikipedia to solve the problem. "
        "Key entities that need to be searched are included between two '@' when output, for example: @catfish effect@, @broken window effect@, @Shakespeare@. "
        "If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "1": (
        "You are an excellent critic. "
        "Please point out potential issues in other agent's analysis point by point.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "2": (
        "You are a psychologist. "
        "You are good at psychology, sociology, and philosophy. "
        "You give people scientific suggestions that will make them feel better. "
        "Based on the question and hints from other agents, provide your analysis and reasoning. "
        "Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "3": (
        "You are a historian. "
        "You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history. "
        "Based on the question and hints from other agents, provide your final analysis and answer. "
        "Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
}



HIERARCHICAL_PROMPTS = {
    TaskType.MATH: {
        "0": ( # Planner
            "You are a math agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "1": ( # Critic
            "You are a science agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "2": ( # Refiner
            "You are a code agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "3": ( # Judger
            "You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>.\n\n"
            "Content from Previous Agent:\n{context}\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
    },
    
    TaskType.MATH_CHOICE: {
        "0": "You are a math agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "1": "You are a science agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "2": "You are a code agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "3": "You are a task summarizer. Given the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nContent from Previous Agent:\n{context}\n\nInput Question: {question}\n\nYour response:",
    },
    TaskType.REASONING_CHOICE: {
        "0": "You are a math agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "1": "You are a science agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "2": "You are a code agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "3": "You are a task summarizer. Given the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nContent from Previous Agent:\n{context}\n\nInput Question: {question}\n\nYour response:",
    },
    TaskType.CODE: {
        "0": ( # Planner
            "You are a math agent. Given the input question, reason step by step and provide an efficient and self-contained Python function that solves the following problem in a markdown code block. You must put all python code as self-contained Python function in markdown code blocks. "
            "For example ```python\nimport needed_library\ndef FUNC_NAME(a, b):\n    return a + b```. "
            "Do not add any other contents inside the markdown code block.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "1": ( # Critic
            "You are a science agent. Given the input question, reason step by step and provide an efficient and self-contained Python function that solves the following problem in a markdown code block. You must put all python code as self-contained Python function in markdown code blocks. "
            "For example ```python\nimport needed_library\ndef FUNC_NAME(a, b):\n    return a + b```. "
            "Do not add any other contents inside the markdown code block.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "2": ( # Refiner
            "You are a code agent. Given the input question, reason step by step and provide an efficient and self-contained Python function that solves the following problem in a markdown code block. You must put all python code as self-contained Python function in markdown code blocks. "
            "For example ```python\nimport needed_library\ndef FUNC_NAME(a, b):\n    return a + b```. "
            "Do not add any other contents inside the markdown code block.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "3": ( # Judger
            "You are a task summarizer. Given the input question and responses from previous agents as reference, reason step by step and provide an efficient and self-contained Python function that solves the following problem in a markdown code block.\n\n"
            "For example ```python\nimport needed_library\ndef FUNC_NAME(a, b):\n    return a + b```. "
            "Do not add any other contents inside the markdown code block.\n\n"
            "Content from Previous Agent:\n{context}\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
    },
}
ROLE_DESCRIPTIONS.update({
    AgentType.MATH_AGENT: "You are a math agent.",
    AgentType.SCIENCE_AGENT: "You are a science agent.",
    AgentType.CODE_AGENT: "You are a code agent.",
    AgentType.TASK_SUMMARIZER: "You are a task summarizer.",
})
