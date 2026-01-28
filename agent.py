
import asyncio
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple, Set

from config import GraphType, AgentType, TaskType, aclient
from prompts import get_agent_template, COMPRESS_PROMPTS
from utils import (
    async_call_llm, extract_answer, extract_code, extract_output,
    majority_vote, code_vote
)
from judges import LLMJudgeAgent, CodeJudgeAgent


class InferenceCache:

    
    def __init__(self):
        self.question: str = ""
        self.node_inputs: Dict[int, str] = {}      # agent_id -> context input
        self.node_outputs_raw: Dict[int, str] = {} # agent_id -> raw output  
        self.node_outputs_short: Dict[int, str] = {} # agent_id -> compressed/extracted output
    
    def set_question(self, question: str):
        self.question = question
    
    def set_node_data(self, agent_id: int, context: str, raw: str, short: str):
        """一次性设置节点的输入和输出"""
        self.node_inputs[agent_id] = context
        self.node_outputs_raw[agent_id] = raw
        self.node_outputs_short[agent_id] = short
    
    def get_context_for_node(self, agent_id: int, predecessors: List[int], 
                              use_short: bool = True) -> str:
        if not predecessors:
            return ""
        
        outputs = self.node_outputs_short if use_short else self.node_outputs_raw
        ctx_list = [outputs.get(p, "") for p in predecessors if p in outputs]
        return "\n---\n".join(ctx_list)
    
    def has_all_predecessors(self, predecessors: List[int]) -> bool:
        return all(p in self.node_outputs_raw for p in predecessors)
    
    def clone(self) -> 'InferenceCache':
        new_cache = InferenceCache()
        new_cache.question = self.question
        new_cache.node_inputs = self.node_inputs.copy()
        new_cache.node_outputs_raw = self.node_outputs_raw.copy()
        new_cache.node_outputs_short = self.node_outputs_short.copy()
        return new_cache
    
    def clone_up_to(self, keep_nodes: Set[int]) -> 'InferenceCache':

        new_cache = InferenceCache()
        new_cache.question = self.question
        new_cache.node_inputs = {k: v for k, v in self.node_inputs.items() if k in keep_nodes}
        new_cache.node_outputs_raw = {k: v for k, v in self.node_outputs_raw.items() if k in keep_nodes}
        new_cache.node_outputs_short = {k: v for k, v in self.node_outputs_short.items() if k in keep_nodes}
        return new_cache
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "node_inputs": self.node_inputs,
            "node_outputs_raw": self.node_outputs_raw,
            "node_outputs_short": self.node_outputs_short,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceCache':
        cache = cls()
        cache.question = data.get("question", "")
        cache.node_inputs = {int(k): v for k, v in data.get("node_inputs", {}).items()}
        cache.node_outputs_raw = {int(k): v for k, v in data.get("node_outputs_raw", {}).items()}
        cache.node_outputs_short = {int(k): v for k, v in data.get("node_outputs_short", {}).items()}
        return cache


class Agent:
    
    def __init__(self, agent_id: int, agent_type: AgentType, task_type: TaskType):
        self.id = agent_id
        self.type = agent_type
        self.task_type = task_type
        self.template = get_agent_template(task_type, agent_type)
        self.last_raw = ""

    async def arun(self, question: str, context: Optional[Any] = None, *, extract: bool = True) -> str:
        if self.type is AgentType.AGGREGATOR:
            return self._aggregate(context or [])

        try:
            prompt = self.template.format(
                question=question, 
                context=context if context is not None else ""
            )
        except (KeyError, IndexError) as e:
            print(f"⚠️ Format error: {e}")
            prompt = self.template.replace("{question}", str(question))
            prompt = prompt.replace("{context}", str(context) if context else "")
        
        raw = await async_call_llm(aclient, prompt, temperature=0.0)
        self.last_raw = raw
        
        if self.type is AgentType.PREDICTOR and not extract:
            compress_template = COMPRESS_PROMPTS.get(self.task_type, COMPRESS_PROMPTS[TaskType.MATH])
            compress_prompt = compress_template.format(raw=raw)
            short = await async_call_llm(aclient, compress_prompt, 0.0)
            return short
        
        return extract_output(raw, self.task_type) if extract else raw

    async def arun_full(self, question: str, context: Optional[str] = None) -> Tuple[str, str]:

        if self.type is AgentType.AGGREGATOR:
            result = self._aggregate(context.split("\n---\n") if context else [])
            return result, result

        try:
            prompt = self.template.format(
                question=question, 
                context=context if context is not None else ""
            )
        except (KeyError, IndexError):
            prompt = self.template.replace("{question}", str(question))
            prompt = prompt.replace("{context}", str(context) if context else "")
        
        raw = await async_call_llm(aclient, prompt, temperature=0.0)
        self.last_raw = raw
        
        compress_template = COMPRESS_PROMPTS.get(self.task_type, COMPRESS_PROMPTS[TaskType.MATH])
        compress_prompt = compress_template.format(raw=raw)
        short = await async_call_llm(aclient, compress_prompt, 0.0)
        
        return raw, short

    def _aggregate(self, answers: List[str]) -> str:
        if not answers:
            return ""
        
        valid_answers = [a.strip() for a in answers if a and a.strip()]
        if not valid_answers:
            return ""
        
        extracted_answers = []
        for ans in valid_answers:
            extracted = extract_output(ans, self.task_type)
            if extracted:
                extracted_answers.append(extracted)
        
        if not extracted_answers:
            return extract_output(valid_answers[0], self.task_type)
        
        if self.task_type == TaskType.CODE:
            return code_vote(extracted_answers)
        return majority_vote(extracted_answers)



class MAS:
    
    def __init__(self, graph_type: GraphType, task_type: TaskType,
                 Ns: int = 1, Na: int = 1, Nr: int = 1, Nd: int = 1,
                 use_judge: bool = False, judge_client=None):
        self.gtype = graph_type
        self.task_type = task_type
        self.Ns, self.Na, self.Nr, self.Nd = Ns, Na, Nr, Nd
        self.agents: List[Agent] = []
        self.edges: Dict[int, List[int]] = {}
        self._build()
        
        self.use_judge = use_judge
        if use_judge:
            if task_type == TaskType.CODE:
                self.judge = CodeJudgeAgent()
            else:
                problem_type = "multi-choice problem" if task_type in [TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE] else "math problem"
                self.judge = LLMJudgeAgent(judge_client, problem_type)
        else:
            self.judge = None

    def _build(self):
        a = self.agents
        tt = self.task_type
        
        if self.gtype is GraphType.SUMMARIZE:
            a.extend([Agent(0, AgentType.SUMMARIZER, tt), Agent(1, AgentType.PREDICTOR, tt)])
            self.edges = {0: [1], 1: []}
            
        elif self.gtype is GraphType.AGGREGATE:
            for i in range(self.Na):
                a.append(Agent(i, AgentType.PREDICTOR, tt))
            agg_id = self.Na
            a.append(Agent(agg_id, AgentType.AGGREGATOR, tt))
            self.edges = {i: [agg_id] for i in range(self.Na)}
            self.edges[agg_id] = []
            
        elif self.gtype is GraphType.REFLECT:
            for i in range(self.Nr):
                pred_id = len(a)
                a.append(Agent(pred_id, AgentType.PREDICTOR, tt))
                refl_id = len(a)
                a.append(Agent(refl_id, AgentType.REFLECTOR, tt))
                
                self.edges[pred_id] = [refl_id]
                self.edges[refl_id] = []
                
                if i > 0:
                    prev_refl_id = pred_id - 1
                    self.edges[prev_refl_id] = [pred_id]
            
        elif self.gtype is GraphType.DEBATE:
            a.extend([
                Agent(0, AgentType.PREDICTOR, tt),
                Agent(1, AgentType.PREDICTOR, tt),
                Agent(2, AgentType.DEBATOR, tt)
            ])
            self.edges = {0: [2], 1: [2], 2: []}
            
        elif self.gtype is GraphType.LLM_AGG:
            for i in range(self.Na):
                a.append(Agent(i, AgentType.PREDICTOR, tt))
            llm_agg_id = self.Na
            a.append(Agent(llm_agg_id, AgentType.LLM_AGG, tt))
            self.edges = {i: [llm_agg_id] for i in range(self.Na)}
            self.edges[llm_agg_id] = []
            
        elif self.gtype is GraphType.DEBATE_LLM_AGG:
            for rnd in range(self.Nd):
                p0 = len(self.agents)
                self.agents.append(Agent(p0, AgentType.PREDICTOR, tt))
                p1 = len(self.agents)
                self.agents.append(Agent(p1, AgentType.PREDICTOR, tt))
                d = len(self.agents)
                self.agents.append(Agent(d, AgentType.DEBATOR, tt))
                self.edges[p0] = [d]
                self.edges[p1] = [d]
                self.edges[d] = []
            
            llm_agg = len(self.agents)
            self.agents.append(Agent(llm_agg, AgentType.LLM_AGG, tt))
            for rnd in range(self.Nd):
                self.edges[rnd*3 + 2] = [llm_agg]
            self.edges[llm_agg] = []
        elif self.gtype is GraphType.HIERARCHICAL:
            from prompts import HIERARCHICAL_PROMPTS
            
            agent_types = [
                AgentType.MATH_AGENT,  # 0
                AgentType.SCIENCE_AGENT,   # 1
                AgentType.CODE_AGENT,  # 2
                AgentType.TASK_SUMMARIZER    # 3
            ]
            
            current_prompts = HIERARCHICAL_PROMPTS.get(tt, HIERARCHICAL_PROMPTS[TaskType.MATH])
            for i in range(4):
                agent = Agent(i, agent_types[i], tt)
                # 直接注入 Prompt
                agent.template = current_prompts[str(i)]
                a.append(agent)
            
            self.edges = {
                0: [3],
                1: [3],
                2: [3],
                3: []
            }

    
    def _topo_order(self) -> List[int]:
        in_degree = defaultdict(int)
        for src, dsts in self.edges.items():
            for d in dsts:
                in_degree[d] += 1
        
        queue = deque([a.id for a in self.agents if in_degree[a.id] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.edges.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    def _get_levels(self) -> List[List[int]]:
        in_degree = defaultdict(int)
        for src, dsts in self.edges.items():
            for d in dsts:
                in_degree[d] += 1

        queue = deque([a.id for a in self.agents if in_degree[a.id] == 0])
        levels = []
        while queue:
            level = list(queue)
            queue.clear()
            for node in level:
                for child in self.edges.get(node, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            levels.append(level)
        return levels

    def get_predecessors(self, agent_id: int) -> List[int]:
        return [k for k, v in self.edges.items() if agent_id in v]

    def get_successors(self, agent_id: int) -> List[int]:
        return self.edges.get(agent_id, [])

    def get_all_predecessors(self, agent_id: int) -> Set[int]:
        predecessors = set()
        queue = deque(self.get_predecessors(agent_id))
        while queue:
            node = queue.popleft()
            if node not in predecessors:
                predecessors.add(node)
                queue.extend(self.get_predecessors(node))
        return predecessors

    def get_all_successors(self, agent_id: int) -> Set[int]:
        successors = set()
        queue = deque(self.get_successors(agent_id))
        while queue:
            node = queue.popleft()
            if node not in successors:
                successors.add(node)
                queue.extend(self.get_successors(node))
        return successors

    def get_terminal_id(self) -> int:
        terminal_nodes = [i for i, v in self.edges.items() if not v]
        return terminal_nodes[0] if terminal_nodes else -1

    def inject_prompt_map(self, prompt_map: Dict[int, str]):
        for aid, p in prompt_map.items():
            if aid < len(self.agents) and self.agents[aid].type != AgentType.AGGREGATOR:
                self.agents[aid].template = p


    async def arun(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        result, _ = await self.arun_with_cache(question, context)
        return result

    async def arun_with_cache(self, question: str, 
                               context: Optional[str] = None) -> Tuple[Dict[str, Any], InferenceCache]:
        cache = InferenceCache()
        cache.set_question(question)
        
        levels = self._get_levels()
        terminal_id = self.get_terminal_id()
        
        for level in levels:
            tasks = []
            for agent_id in level:
                agent = self.agents[agent_id]
                predecessors = self.get_predecessors(agent_id)
                
                if predecessors:
                    if agent.type == AgentType.AGGREGATOR:
                        ctx = cache.get_context_for_node(agent_id, predecessors, use_short=False)
                    else:
                        ctx = cache.get_context_for_node(agent_id, predecessors, use_short=True)
                else:
                    ctx = context
                
                tasks.append(self._run_single_agent(agent, question, ctx, agent_id == terminal_id))
            
            results = await asyncio.gather(*tasks)
            
            for agent_id, (raw, short) in zip(level, results):
                predecessors = self.get_predecessors(agent_id)
                ctx = cache.get_context_for_node(agent_id, predecessors) if predecessors else (context or "")
                cache.set_node_data(agent_id, ctx, raw, short)
        
        final_raw = cache.node_outputs_raw.get(terminal_id, "")
        final_answer = extract_output(final_raw, self.task_type)
        
        return {
            "final": final_answer,
            "raw_trace": cache.node_outputs_raw.copy(),
            "short_trace": cache.node_outputs_short.copy(),
        }, cache

    async def _run_single_agent(self, agent: Agent, question: str, 
                                 context: Optional[str], is_terminal: bool) -> Tuple[str, str]:
        raw, short = await agent.arun_full(question, context)
        

        if agent.type == AgentType.AGGREGATOR:
            answers = context.split("\n---\n") if context else []
            result = agent._aggregate(answers)
            return result, result
        if is_terminal:
            short = extract_output(raw, self.task_type)
        
        return raw, short

    async def arun_from_node(self, start_node: int, 
                              base_cache: InferenceCache,
                              new_prompt: Optional[str] = None) -> Tuple[Dict[str, Any], InferenceCache]:
        question = base_cache.question
        terminal_id = self.get_terminal_id()
        
        
        nodes_to_run = {start_node} | self.get_all_successors(start_node)

        all_nodes = {a.id for a in self.agents}
        nodes_to_keep = all_nodes - nodes_to_run
        
        new_cache = base_cache.clone_up_to(nodes_to_keep)
        
        original_template = None
        if new_prompt is not None:
            original_template = self.agents[start_node].template
            self.agents[start_node].template = new_prompt
        
        try:
            topo_order = self._topo_order()
            nodes_to_run_sorted = [n for n in topo_order if n in nodes_to_run]
            
            for agent_id in nodes_to_run_sorted:
                agent = self.agents[agent_id]
                predecessors = self.get_predecessors(agent_id)
                
                ctx = new_cache.get_context_for_node(agent_id, predecessors, use_short=True)
                
                raw, short = await agent.arun_full(question, ctx if ctx else None)
                
                if agent_id == terminal_id:
                    short = extract_output(raw, self.task_type)
                
                new_cache.set_node_data(agent_id, ctx, raw, short)
            
            final_raw = new_cache.node_outputs_raw.get(terminal_id, "")
            final_answer = extract_output(final_raw, self.task_type)
            
            return {
                "final": final_answer,
                "raw_trace": new_cache.node_outputs_raw.copy(),
                "short_trace": new_cache.node_outputs_short.copy(),
            }, new_cache
            
        finally:
            if original_template is not None:
                self.agents[start_node].template = original_template

    async def arun_single_node_only(self, agent_id: int, question: str, 
                                     context: str, new_prompt: Optional[str] = None) -> Tuple[str, str]:
        agent = self.agents[agent_id]
        
        original_template = None
        if new_prompt is not None:
            original_template = agent.template
            agent.template = new_prompt
        
        try:
            raw, short = await agent.arun_full(question, context)
            return raw, short
        finally:
            if original_template is not None:
                agent.template = original_template
