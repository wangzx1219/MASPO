
import random
import asyncio
from typing import List, Dict, Optional, Tuple, Set, Any 
from tqdm import tqdm
from dataclasses import dataclass, field

from config import GraphType, AgentType, TaskType
from prompts import (
    get_role_description, PROMPT_OPTIMIZE_TEMPLATE,
    ANSWER_EVALUATE_TEMPLATE, INTERMEDIATE_COMPARE_TEMPLATE,
    FINAL_ANSWER_COMPARE_TEMPLATE
)
from utils import async_call_llm, parse_comparison_result, extract_output
from agents import MAS, InferenceCache
import re

@dataclass
class AgentOptState:
    agent_id: int
    current_beam: List[Dict]  
    best_overall_node: Dict   
    total_layers_explored: int = 0 
    current_patience_count: int = 0  
    misleading_cases: List[Dict] = field(default_factory=list)

    recent_bad_cases: List[Dict] = field(default_factory=list)
    misalignment_rates_per_depth: List[float] = field(default_factory=list)  
    beam_refresh_kendall_scores: List[Dict] = field(default_factory=list)
    def __init__(self, agent_id: int, initial_prompt: str):
        self.agent_id = agent_id
        initial_node = {
            "prompt": initial_prompt,
            "cumulative_score": 0.0,
            "path": [initial_prompt]
        }
        self.current_beam = [initial_node]
        self.best_overall_node = initial_node
        self.total_layers_explored = 0
        self.current_patience_count = 0
        self.recent_bad_cases = []
        self.misleading_cases = []
        self.misalignment_rates_per_depth = []
        self.beam_refresh_kendall_scores = []


class MAPromptOptimizer:

    def __init__(self, mas: MAS, train_questions: List[str], 
                 seed_prompt_map: Dict[int, str], evaluator_client):
        self.mas = mas
        self.train_q = train_questions
        self.seed = seed_prompt_map
        self.best_prompt = seed_prompt_map.copy()
        self.evaluator = evaluator_client
        self.task_type = mas.task_type

    async def _compare(self, requirement: str, ans_a: str, ans_b: str, 
                       agent_id: int, question: str, context: str) -> bool:
        agent = self.mas.agents[agent_id]
        role_desc = get_role_description(agent.type, self.task_type)
    
        template = ANSWER_EVALUATE_TEMPLATE.get(
            self.task_type, ANSWER_EVALUATE_TEMPLATE[TaskType.MATH]
        )
    
        prompt = template.format(
            agent_type=agent.type.value,
            role_description=role_desc,
            question=question.strip(),
            requirement=requirement,
            Answer_A=ans_a,
            Answer_B=ans_b,
        )
        raw = await async_call_llm(self.evaluator, prompt, temperature=0, 
                                   max_tokens=16384, use_ds_api=True)
        return parse_comparison_result(raw)

    async def _compare_intermediate(self, question: str, output_a: str, output_b: str) -> bool:
        template = INTERMEDIATE_COMPARE_TEMPLATE.get(
            self.task_type, INTERMEDIATE_COMPARE_TEMPLATE[TaskType.MATH]
        )
    
        prompt = template.format(
            question=question.strip(),
            output_a=output_a.strip() or "(empty)",
            output_b=output_b.strip() or "(empty)"
        )
        resp = await async_call_llm(self.evaluator, prompt, temperature=0.0, use_ds_api=True)
        return parse_comparison_result(resp)

    async def _compare_final_answer(self, question: str, requirement: str, 
                                    ans_a: str, ans_b: str) -> bool:
        template = FINAL_ANSWER_COMPARE_TEMPLATE.get(
            self.task_type, FINAL_ANSWER_COMPARE_TEMPLATE[TaskType.MATH]
        )
    
        prompt = template.format(
            question=question.strip(),
            requirement=requirement,
            Answer_A=ans_a.strip(),
            Answer_B=ans_b.strip()
        )
        resp = await async_call_llm(self.evaluator, prompt, temperature=0.0, use_ds_api=True)
        return parse_comparison_result(resp)
    
    @staticmethod
    def _compute_kendall_top2_overlap(ranking_before: List[str], ranking_after: List[str]) -> float:
        if len(ranking_before) < 1 or len(ranking_after) < 1:
            return 1.0
        
        return 1.0 if ranking_before[0] == ranking_after[0] else 0.0



    def _sanitize_prompt(self, prompt: str, old_p: str) -> str:

        def normalize_braces(text: str) -> str:
            text = re.sub(r'\{+', '{', text) 
            text = re.sub(r'\}+', '}', text)
            return text

        prompt = normalize_braces(prompt)
        old_p = normalize_braces(old_p)

        all_placeholders = re.findall(r'\{([^}]*)\}', prompt)
        new_patterns = set()
        numeric_placeholders = []

        for ph in all_placeholders:
            if ph == "question" or ph == "context":
                new_patterns.add("{" + ph + "}")
            elif ph.isdigit():
                numeric_placeholders.append(ph)
            elif ph == "": 
                pass
            else:
                return old_p


        escaped_prompt = prompt.replace("{", "{{").replace("}", "}}")
        final_prompt = escaped_prompt.replace("{{question}}", "{question}")
        final_prompt = final_prompt.replace("{{context}}", "{context}")

        question_count = final_prompt.count("{question}")
        if question_count == 0:
            final_prompt += "\nQuestion: {question}"
        elif question_count > 1:
            return old_p

        context_count = final_prompt.count("{context}")
        old_has_context = "{context}" in old_p

        if old_has_context:
            if context_count == 0:
                final_prompt += "\nContext: {context}"
            elif context_count > 1:
                return old_p
        elif context_count > 1:
            return old_p

        return final_prompt

    async def _propose_new_prompt(self, requirement: str, old_p: str, 
                                  qa: Dict[str, Dict[str, str]], agent_id: int, 
                                  question: str,
                                  successor_info: str = "") -> str:
        agent = self.mas.agents[agent_id]
        role_desc = get_role_description(agent.type, self.task_type)

        samples_block = "\n\n".join([
            f"Problem {i+1}:\n{q.strip()}\n\nContext:\n{data['context'].strip() or '(no context)'}\n\nAgent Output:\n{data['output'].strip()}"
            for i, (q, data) in enumerate(qa.items())
        ])
    
        full_requirement = "Ensure the agent's role, responsibilities, and input format remain consistent. " + requirement
        # [新增] 注入下游反馈信息
        audience_instruction = ""
        if successor_info:
            audience_instruction = (
                f"\n\n[DOWNSTREAM CONSTRAINT & FEEDBACK]\n"
                f"The output of this agent serves as INPUT for a downstream agent.\n"
                f"{successor_info}\n"
                f"**Optimization Goal**: Crucially, modify the prompt so the output addresses the issues above and strictly adheres to constraints to help the downstream agent succeed."
            )
        template = PROMPT_OPTIMIZE_TEMPLATE.get(
            self.task_type, PROMPT_OPTIMIZE_TEMPLATE[TaskType.MATH]
        )
        augmented_requirements = full_requirement + audience_instruction
    
        prompt = template.format(
            agent_type=agent.type.value,
            role_description=role_desc,
            requirements=full_requirement,
            prompt=old_p,
            samples=samples_block,
        )
    
        raw = await async_call_llm(self.evaluator, prompt, temperature=0.7, 
                                   max_tokens=16384, use_ds_api=True)
        try:
            raw_prompt = raw.split("<prompt>")[1].split("</prompt>")[0].strip()
            print(f"[DEBUG] Extracted prompt:\n{raw_prompt}\n")
            return self._sanitize_prompt(raw_prompt, old_p)
        except IndexError:
            print("[WARN] Failed to extract <prompt>...</prompt>, falling back.")
            return old_p

    async def _build_baseline_caches(self, questions: List[str], 
                                      prompt_map: Dict[int, str]) -> Dict[str, InferenceCache]:
        mas_temp = MAS(
            self.mas.gtype, self.mas.task_type,
            Ns=self.mas.Ns, Na=self.mas.Na, Nr=self.mas.Nr, Nd=self.mas.Nd
        )
        mas_temp.inject_prompt_map(prompt_map)
    
        async def run_one(q: str) -> Tuple[str, InferenceCache]:
            _, cache = await mas_temp.arun_with_cache(q)
            return q, cache
    
        results = await asyncio.gather(*[run_one(q) for q in questions])
        return {q: cache for q, cache in results}

    async def _evaluate_candidate(self, 
                                cand_prompt: str,
                                agent_id: int,
                                eval_samples: List[str],
                                baseline_caches: Dict[str, InferenceCache],
                                temp_prompt_map: Dict[int, str],
                                predecessors: List[int],
                                terminal_id: int,
                                is_terminal: bool,
                                requirement: str,
                                pure_local_mode: bool = False,
                                use_lookahead_score: bool = False,
                                lookahead_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
                                random_sample_set: Optional[Set[str]] = None) -> Dict[str, Any]:  # [新增参数]
        n_eval = len(eval_samples)
        successors = []
        if use_lookahead_score and not is_terminal:
            successors = self.mas.get_successors(agent_id)
        
        async def process_pipeline(q: str):
            base_cache = baseline_caches[q]
            mas_eval = MAS(
                self.mas.gtype, self.mas.task_type,
                Ns=self.mas.Ns, Na=self.mas.Na,
                Nr=self.mas.Nr, Nd=self.mas.Nd
            )
            mas_eval.inject_prompt_map(temp_prompt_map)
            result, _ = await mas_eval.arun_from_node(agent_id, base_cache, cand_prompt)
            cand_raw_trace = result["raw_trace"]
            
            base_output = base_cache.node_outputs_raw.get(agent_id, "")
            cand_output = cand_raw_trace.get(agent_id, "")
            ctx = base_cache.get_context_for_node(agent_id, predecessors)
            
            sample_info = {
                "question": q,
                "context": ctx,
                "output": cand_output
            }
            
            local_tasks = []
            local_metas = []
            if is_terminal:
                local_tasks.append(
                    self._compare(requirement, cand_output, base_output, agent_id, q, ctx)
                )
                local_metas.append({"type": "terminal", "info": sample_info})
            else:
                if not pure_local_mode:
                    base_final = base_cache.node_outputs_raw.get(terminal_id, "")
                    cand_final = cand_raw_trace.get(terminal_id, "")
                    local_tasks.append(
                        self._compare_final_answer(q, requirement, cand_final, base_final)
                    )
                    local_metas.append({"type": "global", "info": sample_info})
            
                local_tasks.append(
                    self._compare_intermediate(q, cand_output, base_output)
                )
                local_metas.append({"type": "local", "info": sample_info})
                if use_lookahead_score and successors:
                    for succ_id in successors:
                        succ_cand_out = cand_raw_trace.get(succ_id, "")
                        succ_base_out = base_cache.node_outputs_raw.get(succ_id, "")
                        if succ_cand_out and succ_base_out:
                            local_tasks.append(self._compare_intermediate(q, succ_cand_out, succ_base_out))
                            local_metas.append({"type": "next_local", "info": sample_info})
            
            sample_scores = await asyncio.gather(*local_tasks)
            return sample_scores, local_metas
        
        all_pipelines = await asyncio.gather(*[process_pipeline(q) for q in eval_samples])
        
        results = []
        compare_metadata = []
        for scores, metas in all_pipelines:
            results.extend(scores)
            compare_metadata.extend(metas)
        
        wins_global = 0
        wins_local = 0
        wins_next_local = 0
        count_next_local = 0
        bad_sample_candidates = {} 
        case_analysis_map = {}
        
        for res, meta in zip(results, compare_metadata):
            is_win = res
            comp_type = meta["type"]
            sample_info = meta["info"]
            q_key = sample_info["question"]
            
            if q_key not in case_analysis_map:
                case_analysis_map[q_key] = {"info": sample_info}
            
            if comp_type == "local":
                case_analysis_map[q_key]["local"] = is_win
            elif comp_type == "global":
                case_analysis_map[q_key]["global"] = is_win
            elif comp_type == "terminal":
                case_analysis_map[q_key]["global"] = is_win
            elif comp_type == "next_local":
                wins_next_local += 1 if is_win else 0
                count_next_local += 1
                case_analysis_map[q_key]["next_local"] = is_win
            
            if not is_win and (comp_type == "local" or comp_type == "terminal"):
                bad_sample_candidates[q_key] = sample_info
        
        collected_bad_cases = list(bad_sample_candidates.values())[:3]


        misalignment_count = 0
        total_evaluable = 0
        
        collected_misleading = []
        candidates_with_priority = []
        
        for q, data in case_analysis_map.items():
            if "local" not in data:
                continue
            
            is_random_sample = (random_sample_set is None) or (q in random_sample_set)
            
            is_local_win = data.get("local", False)
            
            if not is_local_win:
                if is_random_sample:
                    total_evaluable += 1
                continue
            
            is_global_win = data.get("global", True)
            is_next_win = data.get("next_local", True)
            
            if is_random_sample:
                total_evaluable += 1
                if (not is_next_win) or (not is_global_win):
                    misalignment_count += 1
            
            priority = -1
            if (not is_next_win) and (not is_global_win):
                priority = 0
            elif not is_next_win:
                priority = 1
            elif not is_global_win:
                priority = 2
            
            if priority != -1:
                candidates_with_priority.append({
                    "priority": priority,
                    "info": data["info"]
                })
        
        candidates_with_priority.sort(key=lambda x: x["priority"])
        collected_misleading = [x["info"] for x in candidates_with_priority[:3]]
        
        misalignment_rate = misalignment_count / total_evaluable if total_evaluable > 0 else 0.0
        
        if is_terminal:
            wins_global = sum(results)
            win_rate = wins_global / len(eval_samples) if eval_samples else 0
        else:
            wins_global = sum(r for r, m in zip(results, compare_metadata) if m["type"] == "global")
            wins_local = sum(r for r, m in zip(results, compare_metadata) if m["type"] == "local")
            
            if pure_local_mode:
                win_rate = wins_local / len(eval_samples) if eval_samples else 0
            else:
                rate_local = wins_local / n_eval if n_eval > 0 else 0
                rate_global = wins_global / n_eval if n_eval > 0 else 0
                
                if use_lookahead_score and count_next_local > 0:
                    rate_next = wins_next_local / count_next_local
                    w_local, w_next, w_global = lookahead_weights
                    win_rate = (rate_local * w_local) + (rate_next * w_next) + (rate_global * w_global)
                else:
                    win_rate = (rate_global * 0.3 + rate_local * 0.7)

        return {
            "score": win_rate - 0.5,
            "wins_global": wins_global,
            "wins_local": wins_local,
            "wins_next_local": wins_next_local,
            "bad_cases": collected_bad_cases,
            "misleading_cases": collected_misleading,
            "misalignment_rate": misalignment_rate,
            "misalignment_count": misalignment_count,
            "total_evaluable": total_evaluable,
        }


    async def _optimize_agent(self, agent_id: int, requirement: str) -> str:
        agent = self.mas.agents[agent_id]
        best_p = self.best_prompt[agent_id]
        terminal_id = self.mas.get_terminal_id()
        is_terminal = (agent_id == terminal_id)
        predecessors = self.mas.get_predecessors(agent_id)
    
        current_level = [{
            "prompt": best_p,
            "cumulative_score": 0.0,
            "path": [best_p],
        }]
        max_depth = 10
        beam_width = 2
        best_overall = {"prompt": best_p, "cumulative_score": 0.0}

        desc_prefix = f"Agent-{agent_id}({agent.type.value}){'[TERM]' if is_terminal else ''}"
        current_base_map = self.best_prompt.copy()
        with tqdm(total=max_depth, desc=desc_prefix, leave=False) as tree_bar:
            for depth in range(max_depth):
                eval_samples = random.sample(self.train_q, min(10, len(self.train_q)))
                eval_samples_part1 = eval_samples[:5]
                eval_samples_part2 = eval_samples[5:]
                n_eval = len(eval_samples)

                node_results = await asyncio.gather(*[
                    self.process_single_node(
                        node=node,
                        aid=agent_id,
                        preds=predecessors,
                        term_id=terminal_id,
                        is_term=is_terminal,
                        req=requirement,
                        samples=eval_samples,
                        samples_p1=eval_samples_part1,
                        samples_p2=eval_samples_part2,
                        base_prompt_map=current_base_map,
                        agent_states=None, 
                        use_stochastic_sampling=False,
                        use_feedback=False 
                    )
                    for node in current_level
                ])

                all_next_nodes = []
                for res in node_results:
                    all_next_nodes.extend(res["nodes"])
                    if res["best_cumulative"] > best_overall["cumulative_score"]:
                        best_overall["prompt"] = res["best_prompt"]
                        best_overall["cumulative_score"] = res["best_cumulative"]

                all_next_nodes.sort(key=lambda x: x["cumulative_score"], reverse=True)
                current_level = all_next_nodes[:beam_width]

                tree_bar.set_postfix_str(f"depth {depth+1}, best={best_overall['cumulative_score']:.3f}")
                tree_bar.update(1)

                if not current_level:
                    break

        return best_overall["prompt"]

    async def optimize_all(self, requirement: str = " ") -> Dict[int, str]:
        order = self.mas._topo_order()
    
        with tqdm(total=len(order), desc="[OPT]") as topo_bar:
            for aid in order:
                if self.mas.agents[aid].type == AgentType.AGGREGATOR:
                    topo_bar.update(1)
                    continue
            
                # 优化当前节点
                new_p = await self._optimize_agent(aid, requirement)
                self.best_prompt[aid] = new_p
            
                topo_bar.set_postfix_str(f"Agent-{aid} done")
                topo_bar.update(1)
    
        return self.best_prompt


    async def _optimize_agent_stateful(self,
                                    state: AgentOptState,
                                    requirement: str,
                                    max_depth: int = 10,
                                    beam_width: int = 2,
                                    patience: int = 3) -> Tuple[int, bool]:
        agent_id = state.agent_id
        if self.mas.agents[agent_id].type == AgentType.AGGREGATOR:
            return 0, False
        
        if state.total_layers_explored >= max_depth:
            return 0, False

        terminal_id = self.mas.get_terminal_id()
        is_terminal = (agent_id == terminal_id)
        predecessors = self.mas.get_predecessors(agent_id)
        initial_prompt_map = self.best_prompt.copy()
        current_best_prompt = initial_prompt_map[agent_id]

        if state.total_layers_explored == 0:
            initial_node = {
                "prompt": current_best_prompt,
                "cumulative_score": 0.0,
                "path": [current_best_prompt]
            }
            state.current_beam = [initial_node]
            state.best_overall_node = initial_node.copy()

        depth = state.total_layers_explored
        eval_samples = random.sample(self.train_q, min(10, len(self.train_q)))
        eval_samples_part1 = eval_samples[:5]
        eval_samples_part2 = eval_samples[5:]

        async def process_single_node(node: Dict, aid: int, preds: List[int], term_id: int,
                                    is_term: bool, req: str, samples: List[str],
                                    samples_p1: List[str], samples_p2: List[str]) -> Dict:
            node_prompt = node["prompt"]
            temp_prompt_map = self.best_prompt.copy()
            temp_prompt_map[aid] = node_prompt

            local_baseline_caches = await self._build_baseline_caches(samples, temp_prompt_map)
            qa_for_proposal1 = {}
            for q in samples_p1:
                cache = local_baseline_caches[q]
                ctx = cache.get_context_for_node(aid, preds)
                qa_for_proposal1[q] = {"context": ctx, "output": cache.node_outputs_raw.get(aid, "")}

            qa_for_proposal2 = {}
            for q in samples_p2:
                cache = local_baseline_caches[q]
                ctx = cache.get_context_for_node(aid, preds)
                qa_for_proposal2[q] = {"context": ctx, "output": cache.node_outputs_raw.get(aid, "")}

            sample1 = samples_p1[0] if samples_p1 else samples[0]
            sample2 = samples_p2[0] if samples_p2 else samples[-1]

            cand_prompts = await asyncio.gather(
                self._propose_new_prompt(req, node_prompt, qa_for_proposal1, aid, sample1),
                self._propose_new_prompt(req, node_prompt, qa_for_proposal2, aid, sample2)
            )
            candidates = list(set(cand_prompts))
            candidate_scores = {}

            for cand_p in candidates:
                if cand_p == node_prompt:
                    candidate_scores[cand_p] = {"score": 0.0, "wins_global": len(samples)//2, "wins_local": len(samples)//2}
                    continue
                score_info = await self._evaluate_candidate(
                    cand_prompt=cand_p, agent_id=aid, eval_samples=samples,
                    baseline_caches=local_baseline_caches, temp_prompt_map=temp_prompt_map,
                    predecessors=preds, terminal_id=term_id, is_terminal=is_term, requirement=req
                )
                candidate_scores[cand_p] = score_info

            new_nodes = []
            local_best = {"prompt": node_prompt, "score": 0.0}
            for cand_p in candidates:
                info = candidate_scores[cand_p]
                if info["score"] > 0:
                    new_node = {
                        "prompt": cand_p,
                        "cumulative_score": node["cumulative_score"] + info["score"],
                        "path": node["path"] + [cand_p],
                    }
                    new_nodes.append(new_node)
                    if info["score"] > local_best["score"]:
                        local_best = {"prompt": cand_p, "score": info["score"]}
                else:
                    new_nodes.append({
                        "prompt": node_prompt,
                        "cumulative_score": node["cumulative_score"],
                        "path": node["path"],
                    })

            return {
                "nodes": new_nodes,
                "best_prompt": local_best["prompt"],
                "best_cumulative": node["cumulative_score"] + max(0, local_best["score"]),
            }

        # 处理当前层节点
        if not state.current_beam:
            state.current_beam = [{
                "prompt": state.best_overall_node["prompt"],
                "cumulative_score": state.best_overall_node["cumulative_score"],
                "path": state.best_overall_node["path"]
            }]

        node_results = await asyncio.gather(*[
            process_single_node(
                node=node, aid=agent_id, preds=predecessors, term_id=terminal_id,
                is_term=is_terminal, req=requirement, samples=eval_samples,
                samples_p1=eval_samples_part1, samples_p2=eval_samples_part2
            )
            for node in state.current_beam
        ])

        # 收集结果+更新状态
        all_next_nodes = []
        current_best_node = state.best_overall_node.copy()
        for res in node_results:
            all_next_nodes.extend(res["nodes"])
            if res["best_cumulative"] > current_best_node["cumulative_score"]:
                current_best_node = {
                    "prompt": res["best_prompt"],
                    "cumulative_score": res["best_cumulative"],
                    "path": res.get("path", [])
                }

        # Beam剪枝
        all_next_nodes.sort(key=lambda x: x["cumulative_score"], reverse=True)
        state.current_beam = all_next_nodes[:beam_width]

        # 更新patience计数
        improved = current_best_node["cumulative_score"] > state.best_overall_node["cumulative_score"] + 1e-6
        trigger_patience = False
        base_score = state.best_overall_node["cumulative_score"]
        if improved:
            state.current_patience_count = 0
            state.best_overall_node = current_best_node
        else:
            state.current_patience_count += 1
            if state.current_patience_count >= patience:
                trigger_patience = True

        # 更新全局prompt（有显著改进时）
        score_delta = state.best_overall_node["cumulative_score"] - base_score
        if score_delta > 0:
            old_prompt = self.best_prompt[agent_id]
            new_prompt = state.best_overall_node["prompt"]
            self.best_prompt[agent_id] = new_prompt

        # 推进探索层数
        state.total_layers_explored += 1
        return 1, trigger_patience


    async def optimize_all_round_robin(self, 
                                    requirement: str = " ",
                                    max_total_depth: int = 10,  # 每个Agent必须完成的总次数
                                    beam_width: int = 2,
                                    patience: int = 3) -> Dict[int, str]:
        topo_order = self.mas._topo_order()
        optimizable_agents = [
            aid for aid in topo_order
            if self.mas.agents[aid].type != AgentType.AGGREGATOR
        ]

        if not optimizable_agents:
            return self.best_prompt

        agent_states = {
            aid: AgentOptState(aid, self.best_prompt[aid])
            for aid in optimizable_agents
        }


        round_num = 0
        while True:
            round_num += 1
            all_completed = all(
                state.total_layers_explored >= max_total_depth
                for state in agent_states.values()
            )
            if all_completed:
                break

            for aid in optimizable_agents:
                state = agent_states[aid]
                if state.total_layers_explored >= max_total_depth:
                    continue

                agent = self.mas.agents[aid]
                agent_type_str = f"{agent.type.value}{' [TERM]' if aid == self.mas.get_terminal_id() else ''}"
                state.current_patience_count = 0

                while True:
                    if state.total_layers_explored >= max_total_depth:
                        break

                    layers_explored, trigger_patience = await self._optimize_agent_stateful(
                        state=state,
                        requirement=requirement,
                        max_depth=max_total_depth,
                        beam_width=beam_width,
                        patience=patience
                    )

                    if trigger_patience:
                        break

        return self.best_prompt


    async def _optimize_agent_fixed_rounds(self,
                                        state: AgentOptState,
                                        requirement: str,
                                        agent_states: Dict[int, AgentOptState], 
                                        use_dynamic_switching: bool = False,
                                        use_stochastic_sampling: bool = False,
                                        use_beam_refresh: bool = False,  # [新增参数]
                                        rounds_per_turn: int = 2,
                                        max_total_depth: int = 10,
                                        use_feedback: bool = False,
                                        use_misleading_sampling: bool = False,
                                        use_lookahead_score: bool = False,
                                        lookahead_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> int:
        agent_id = state.agent_id
        if self.mas.agents[agent_id].type == AgentType.AGGREGATOR:
            return 0
        
        terminal_id = self.mas.get_terminal_id()
        is_terminal = (agent_id == terminal_id)
        predecessors = self.mas.get_predecessors(agent_id)
        
        if use_beam_refresh and state.total_layers_explored > 0 and state.current_beam:
            await self._refresh_beam_scores(
                state, requirement, predecessors, terminal_id, is_terminal
            )
        
        completed_rounds = 0
        if state.total_layers_explored == 0:
            initial_node = {
                "prompt": state.best_overall_node["prompt"],
                "cumulative_score": 0.0,
                "path": [state.best_overall_node["prompt"]]
            }
            state.current_beam = [initial_node]
        
        for _ in range(rounds_per_turn):
            if state.total_layers_explored >= max_total_depth:
                break
            
            current_base_map = None
            if use_dynamic_switching and agent_states:
                current_base_map = self._get_dynamic_context(
                    agent_states, 
                    current_agent_id=agent_id, 
                    current_depth=state.total_layers_explored
                )
            
            eval_samples = random.sample(self.train_q, min(10, len(self.train_q)))
            eval_samples_part1 = eval_samples[:5]
            eval_samples_part2 = eval_samples[5:]
            
            if not state.current_beam:
                state.current_beam = [{
                    "prompt": state.best_overall_node["prompt"],
                    "cumulative_score": state.best_overall_node["cumulative_score"],
                    "path": state.best_overall_node["path"]
                }]
            
            node_results = await asyncio.gather(*[
                self.process_single_node(
                    node=node,
                    aid=agent_id,
                    preds=predecessors,
                    term_id=terminal_id,
                    is_term=is_terminal,
                    req=requirement,
                    samples=eval_samples,
                    samples_p1=eval_samples_part1,
                    samples_p2=eval_samples_part2,
                    base_prompt_map=current_base_map,
                    agent_states=agent_states,
                    use_stochastic_sampling=use_stochastic_sampling,
                    use_feedback=use_feedback,
                    use_misleading_sampling=use_misleading_sampling,
                    use_lookahead_score=use_lookahead_score,
                    lookahead_weights=lookahead_weights,
                )
                for node in state.current_beam
            ])
            depth_misalignment_rates = []
            for res in node_results:
                if "avg_misalignment_rate" in res:
                    depth_misalignment_rates.append(res["avg_misalignment_rate"])

            if depth_misalignment_rates:
                avg_depth_misalignment = sum(depth_misalignment_rates) / len(depth_misalignment_rates)
            else:
                avg_depth_misalignment = 0.0
            state.misalignment_rates_per_depth.append(avg_depth_misalignment)

            all_next_nodes = []
            current_best_node = state.best_overall_node.copy()
            current_best_bad_cases = []
            current_best_misleading = []

            for res in node_results:
                all_next_nodes.extend(res["nodes"])
                if res["best_cumulative"] > current_best_node["cumulative_score"]:
                    current_best_node = {
                        "prompt": res["best_prompt"],
                        "cumulative_score": res["best_cumulative"],
                        "path": res.get("path", [])
                    }
                    current_best_bad_cases = res.get("best_bad_cases", [])
                    current_best_misleading = res.get("best_misleading_cases", []) 
            all_next_nodes.sort(key=lambda x: x["cumulative_score"], reverse=True)
            state.current_beam = all_next_nodes[:self.beam_width]
            
            if current_best_node["cumulative_score"] > state.best_overall_node["cumulative_score"] + 1e-6:
                state.best_overall_node = current_best_node
                self.best_prompt[agent_id] = current_best_node["prompt"]
                if current_best_bad_cases:
                    state.recent_bad_cases = current_best_bad_cases
                if current_best_misleading:
                    state.misleading_cases = current_best_misleading

            state.total_layers_explored += 1
            completed_rounds += 1
        return completed_rounds

    async def optimize_all_fixed_rounds(self,
                                    requirement: str = " ",
                                    max_total_depth: int = 10,
                                    rounds_per_turn: int = 2,
                                    beam_width: int = 2,
                                    use_dynamic_switching: bool = False,
                                    use_stochastic_sampling: bool = False,
                                    use_beam_refresh: bool = False,
                                    use_feedback: bool = False,
                                    use_misleading_sampling: bool = False,
                                    use_lookahead_score: bool = False,
                                    lookahead_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> Dict[int, str]: # [新增参数]
        self.beam_width = beam_width
        topo_order = self.mas._topo_order()
        optimizable_agents = [
            aid for aid in topo_order
            if self.mas.agents[aid].type != AgentType.AGGREGATOR
        ]
        if not optimizable_agents:
            return self.best_prompt
        
        agent_states = {
            aid: AgentOptState(aid, self.best_prompt[aid])
            for aid in optimizable_agents
        }
        
        
        round_num = 0
        while True:
            round_num += 1
            all_completed = all(
                state.total_layers_explored >= max_total_depth
                for state in agent_states.values()
            )
            if all_completed:
                break
            
            
            for aid in optimizable_agents:
                state = agent_states[aid]
                if state.total_layers_explored >= max_total_depth:
                    continue
                
                agent = self.mas.agents[aid]
                agent_type_str = f"{agent.type.value}{' [TERM]' if aid == self.mas.get_terminal_id() else ''}"
                remaining = max_total_depth - state.total_layers_explored
                process_rounds = min(rounds_per_turn, remaining)
                
                await self._optimize_agent_fixed_rounds(
                    state=state,
                    requirement=requirement,
                    agent_states=agent_states,
                    use_dynamic_switching=use_dynamic_switching,
                    use_stochastic_sampling=use_stochastic_sampling,
                    use_beam_refresh=use_beam_refresh,
                    rounds_per_turn=process_rounds,
                    max_total_depth=max_total_depth,
                    use_feedback=use_feedback,
                    use_misleading_sampling=use_misleading_sampling,
                    use_lookahead_score=use_lookahead_score,
                    lookahead_weights=lookahead_weights,
                )

        statistics = self._aggregate_statistics(agent_states, max_total_depth)
        
        self._print_statistics_summary(statistics)
        
        return self.best_prompt, statistics

    def _aggregate_statistics(self, agent_states: Dict[int, AgentOptState], 
                            max_depth: int) -> Dict[str, Any]:

        terminal_id = self.mas.get_terminal_id()
        
        misalignment_per_agent = {}
        for aid, state in agent_states.items():
            if aid == terminal_id:
                continue
            misalignment_per_agent[aid] = state.misalignment_rates_per_depth.copy()
        
        averaged_by_depth = []
        for depth_idx in range(max_depth):
            rates_at_depth = []
            for aid, rates in misalignment_per_agent.items():
                if depth_idx < len(rates):
                    rates_at_depth.append(rates[depth_idx])
            if rates_at_depth:
                averaged_by_depth.append(sum(rates_at_depth) / len(rates_at_depth))
            else:
                averaged_by_depth.append(0.0)
        
        kendall_per_agent = {}
        all_kendall_scores = []
        for aid, state in agent_states.items():
            scores = [record["kendall_top2_overlap"] for record in state.beam_refresh_kendall_scores]
            kendall_per_agent[aid] = scores
            all_kendall_scores.extend(scores)
        
        avg_kendall = sum(all_kendall_scores) / len(all_kendall_scores) if all_kendall_scores else 1.0
        
        return {
            "misalignment_rates": {
                "per_agent": misalignment_per_agent,
                "averaged_by_depth": averaged_by_depth,
            },
            "kendall_scores": {
                "per_agent": kendall_per_agent,
                "averaged": avg_kendall,
                "all_scores": all_kendall_scores,
                "detailed_records": {
                    aid: state.beam_refresh_kendall_scores 
                    for aid, state in agent_states.items()
                }
            }
        }
    def _print_statistics_summary(self, statistics: Dict[str, Any]):
        avg_by_depth = statistics["misalignment_rates"]["averaged_by_depth"]
        
        per_agent = statistics["misalignment_rates"]["per_agent"]


    async def process_single_node(self, node: Dict, aid: int, preds: List[int], term_id: int,
                                is_term: bool, req: str, samples: List[str],
                                samples_p1: List[str], samples_p2: List[str],
                                base_prompt_map: Optional[Dict[int, str]] = None,
                                agent_states: Optional[Dict[int, AgentOptState]] = None, 
                                use_stochastic_sampling: bool = False,
                                use_feedback: bool = False,
                                use_misleading_sampling: bool = False,
                                use_lookahead_score: bool = False,
                                lookahead_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> Dict:
        final_samples = []
        injected_questions = set()
        
        if use_misleading_sampling and agent_states:
            for pred_id in preds:
                if pred_id in agent_states:
                    traps = agent_states[pred_id].misleading_cases
                    for t in traps:
                        injected_questions.add(t["question"])
        
        injected_list = list(injected_questions)
        if len(injected_list) > 5:
            injected_list = random.sample(injected_list, 5)
        
        if injected_list:
            final_samples.extend(injected_list)
        
        target_total = 10
        needed = max(0, target_total - len(final_samples))
        
        pool = [q for q in self.train_q if q not in injected_questions]
        
        random_samples = [] 
        if needed > 0:
            if len(pool) >= needed:
                random_samples = random.sample(pool, needed)
            else:
                random_samples = pool.copy()
            final_samples.extend(random_samples)
        
        random_sample_set = set(random_samples)
        
        current_samples = final_samples
        mid = len(current_samples) // 2
        current_samples_p1 = current_samples[:mid]
        current_samples_p2 = current_samples[mid:]
        node_prompt = node["prompt"]
        
        diverse_prompt_maps = []
        base_map_template = (base_prompt_map or self.best_prompt).copy()
        base_map_template[aid] = node_prompt 
        predecessors_set = set(preds)
        
        for i, sample in enumerate(current_samples):
            current_map = base_map_template.copy()
            is_robustness_sample = (i >= len(current_samples) * 0.7)
            if use_stochastic_sampling and agent_states and is_robustness_sample:
                for pred_id in predecessors_set:
                    if pred_id in agent_states and agent_states[pred_id].current_beam:
                        beam_nodes = agent_states[pred_id].current_beam
                        if len(beam_nodes) > 1:
                            alternatives = beam_nodes[1:]
                            chosen = random.choice(alternatives)
                            current_map[pred_id] = chosen["prompt"]
            diverse_prompt_maps.append(current_map)
        
        base_map_for_eval = diverse_prompt_maps[0] 
        local_baseline_caches = await self._build_diverse_baseline_caches(current_samples, diverse_prompt_maps)
        qa_for_proposal1 = {}
        for q in current_samples_p1:
            cache = local_baseline_caches[q]
            ctx = cache.get_context_for_node(aid, preds)
            qa_for_proposal1[q] = {"context": ctx, "output": cache.node_outputs_raw.get(aid, "")}
        
        qa_for_proposal2 = {}
        for q in current_samples_p2:
            cache = local_baseline_caches[q]
            ctx = cache.get_context_for_node(aid, preds)
            qa_for_proposal2[q] = {"context": ctx, "output": cache.node_outputs_raw.get(aid, "")}
        
        sample1 = current_samples_p1[0] if current_samples_p1 else samples[0]
        sample2 = current_samples_p2[0] if current_samples_p2 else samples[-1]
        
        successor_info_text = ""
        if use_feedback and agent_states:
            successors = self.mas.get_successors(aid)
            feedback_messages = []
            
            for succ_id in successors:
                if succ_id in agent_states:
                    bad_cases = agent_states[succ_id].recent_bad_cases
                    if bad_cases:
                        succ_agent_type = self.mas.agents[succ_id].type.value
                        
                        cases_str = "\n".join([
                            f"  - Case {i+1}:\n"
                            f"    [Problem]: \"{case['question']}\"\n"
                            f"    [Context Snippet]: \"{case['context'][:200]}\"\n" 
                            f"    The downstream agent ({succ_agent_type}) failed to produce a correct/better answer."
                            for i, case in enumerate(bad_cases[:2])
                        ])
                        
                        msg = (f"Feedback from downstream Agent-{succ_id} ({succ_agent_type}):\n"
                            f"Your previous outputs led to failures in the downstream task in the following cases:\n{cases_str}")
                        feedback_messages.append(msg)
            
            if feedback_messages:
                successor_info_text = "\n\n".join(feedback_messages)
        cand_prompts = await asyncio.gather(
            self._propose_new_prompt(req, node_prompt, qa_for_proposal1, aid, sample1, successor_info=successor_info_text),
            self._propose_new_prompt(req, node_prompt, qa_for_proposal2, aid, sample2, successor_info=successor_info_text)
        )
        candidates = list(set(cand_prompts))

        
        async def evaluate_wrapper(cand_p: str) -> Tuple[str, Dict]:
            """包装函数：评估单个候选Prompt"""
            if cand_p == node_prompt:
                return cand_p, {
                    "score": 0.0, 
                    "wins_global": len(current_samples)//2, 
                    "wins_local": len(current_samples)//2,
                    "bad_cases": [],
                    "misleading_cases": [],
                    "misalignment_rate": 0.0,
                    "misalignment_count": 0,
                    "total_evaluable": 0,
                }
            
            # [修改] 传入 random_sample_set 参数
            score_info = await self._evaluate_candidate(
                cand_prompt=cand_p, 
                agent_id=aid, 
                eval_samples=current_samples,
                baseline_caches=local_baseline_caches, 
                temp_prompt_map=base_map_for_eval,
                predecessors=preds, 
                terminal_id=term_id, 
                is_terminal=is_term, 
                requirement=req,
                use_lookahead_score=use_lookahead_score,
                lookahead_weights=lookahead_weights,
                random_sample_set=random_sample_set 
            )
            return cand_p, score_info
        eval_results = await asyncio.gather(*[evaluate_wrapper(p) for p in candidates])
        
        candidate_scores = {p: info for p, info in eval_results}
        new_nodes = []
        local_best = {
            "prompt": node_prompt, 
            "score": 0.0,
            "bad_cases": [],
            "misleading_cases": [],
            "misalignment_rate": 0.0,
        }
        all_misalignment_rates = []
        
        for cand_p in candidates:
            info = candidate_scores[cand_p]
            all_misalignment_rates.append(info.get("misalignment_rate", 0.0))
            if info["score"] > local_best["score"]:
                local_best = {
                    "prompt": cand_p, 
                    "score": info["score"],
                    "bad_cases": info["bad_cases"],
                    "misleading_cases": info["misleading_cases"],
                    "misalignment_rate": info.get("misalignment_rate", 0.0),
                }
            if info["score"] > 0:
                new_node = {
                    "prompt": cand_p,
                    "cumulative_score": node["cumulative_score"] + info["score"],
                    "path": node["path"] + [cand_p],
                    "_temp_bad_cases": info["bad_cases"],
                    "_temp_misleading": info["misleading_cases"],
                    "misalignment_rate": info.get("misalignment_rate", 0.0),
                }
                new_nodes.append(new_node)
            else:
                new_nodes.append({
                    "prompt": node_prompt,
                    "cumulative_score": node["cumulative_score"],
                    "path": node["path"],
                    "_temp_bad_cases": [],
                    "_temp_misleading": [],
                    "_temp_misalignment_rate": 0.0,
                })
        avg_misalignment_rate = sum(all_misalignment_rates) / len(all_misalignment_rates) if all_misalignment_rates else 0.0
        return {
            "nodes": new_nodes,
            "best_prompt": local_best["prompt"],
            "best_cumulative": node["cumulative_score"] + max(0, local_best["score"]),
            "best_bad_cases": local_best["bad_cases"],
            "best_misleading_cases": local_best["misleading_cases"],
            "avg_misalignment_rate": avg_misalignment_rate,
        }


    def _get_dynamic_context(self, agent_states: Dict[int, AgentOptState], 
                           current_agent_id: int, 
                           current_depth: int) -> Dict[int, str]:

        context_map = self.best_prompt.copy()

        try:
            direct_predecessors = set(self.mas.graph.predecessors(current_agent_id))
        except AttributeError:
            direct_predecessors = set(self.mas.get_predecessors(current_agent_id))
        should_switch = (current_depth > 0 and current_depth % 2 != 0)
        
        for aid, state in agent_states.items():
            if aid == current_agent_id:
                continue
            
            if not state.current_beam:
                continue
            selected_prompt = state.best_overall_node["prompt"]
            if should_switch and aid in direct_predecessors:
                sorted_nodes = sorted(state.current_beam, key=lambda x: x["cumulative_score"], reverse=True)
                if len(sorted_nodes) >= 2:
                    selected_prompt = sorted_nodes[1]["prompt"]
            
            context_map[aid] = selected_prompt
            
        return context_map

    async def _build_diverse_baseline_caches(self, questions: List[str], 
                                           prompt_maps: List[Dict[int, str]]) -> Dict[str, InferenceCache]:

        assert len(questions) == len(prompt_maps), "Questions and prompt_maps must align"
        async def run_one(q: str, p_map: Dict[int, str]) -> Tuple[str, InferenceCache]:
            mas_temp = MAS(
                self.mas.gtype, self.mas.task_type,
                Ns=self.mas.Ns, Na=self.mas.Na, Nr=self.mas.Nr, Nd=self.mas.Nd
            )
            mas_temp.inject_prompt_map(p_map)
            _, cache = await mas_temp.arun_with_cache(q)
            return q, cache
    
        results = await asyncio.gather(*[
            run_one(q, p_map) for q, p_map in zip(questions, prompt_maps)
        ])
        return {q: cache for q, cache in results}


    async def _refresh_beam_scores(self, 
                                 state: AgentOptState, 
                                 requirement: str,
                                 predecessors: List[int],
                                 terminal_id: int,
                                 is_terminal: bool) -> None:

        if not state.current_beam:
            return

        print(f"    ↻ [Beam Refresh] Re-evaluating {len(state.current_beam)} nodes in beam...")
        ranking_before = [node["prompt"] for node in sorted(
            state.current_beam, 
            key=lambda x: x["cumulative_score"], 
            reverse=True
        )]
        scores_before = {node["prompt"]: node["cumulative_score"] for node in state.current_beam}
        eval_samples = random.sample(self.train_q, min(10, len(self.train_q)))
        
        current_global_map = self.best_prompt.copy()
        prompt_maps = [current_global_map] * len(eval_samples)
        
        local_baseline_caches = await self._build_diverse_baseline_caches(eval_samples, prompt_maps)
        
        async def re_evaluate_node(node):
            prompt = node["prompt"]
            
            if prompt == self.best_prompt[state.agent_id]:
                node["cumulative_score"] = 0.0
                return node
            
            temp_map = current_global_map.copy()
            temp_map[state.agent_id] = prompt
            
            score_info = await self._evaluate_candidate(
                cand_prompt=prompt,
                agent_id=state.agent_id,
                eval_samples=eval_samples,
                baseline_caches=local_baseline_caches,
                temp_prompt_map=temp_map,
                predecessors=predecessors,
                terminal_id=terminal_id,
                is_terminal=is_terminal,
                requirement=requirement
            )
            
            wins_global = score_info["wins_global"]
            wins_local = score_info["wins_local"]
            
            if is_terminal:
                win_rate = wins_global / len(eval_samples) if eval_samples else 0
            else:
                total = len(eval_samples)
                if total > 0:
                    win_rate = (wins_global/total) * 0.3 + (wins_local/total) * 0.7
                else:
                    win_rate = 0
            node["cumulative_score"] = win_rate - 0.5
            return node

        new_beam = await asyncio.gather(*[re_evaluate_node(n) for n in state.current_beam])
        
        state.current_beam = sorted(new_beam, key=lambda x: x["cumulative_score"], reverse=True)

        ranking_after = [node["prompt"] for node in state.current_beam]
        scores_after = {node["prompt"]: node["cumulative_score"] for node in state.current_beam}
        
        kendall_score = self._compute_kendall_top2_overlap(ranking_before, ranking_after)
        
        state.beam_refresh_kendall_scores.append({
            "depth": state.total_layers_explored,
            "kendall_top2_overlap": kendall_score,
            "ranking_before": ranking_before[:3],
            "ranking_after": ranking_after[:3],
            "scores_before": {p[:50]: s for p, s in list(scores_before.items())[:3]},
            "scores_after": {p[:50]: s for p, s in list(scores_after.items())[:3]},
        })
        
        top_node = state.current_beam[0]
        state.best_overall_node = top_node.copy()
        
        if top_node["prompt"] != self.best_prompt[state.agent_id]:
             print(f"    ⚡ [Beam Refresh] Anchor Shifted! New best score: {top_node['cumulative_score']:.3f}")
             self.best_prompt[state.agent_id] = top_node["prompt"]
        else:
             print(f"    ✓ [Beam Refresh] Anchor retained. Score updated to: {top_node['cumulative_score']:.3f}")

    async def _optimize_agent_simple_serial(self, agent_id: int, requirement: str, 
                                            rounds: int = 9, sample_size: int = 3) -> str:
        agent = self.mas.agents[agent_id]
        
        terminal_id = self.mas.get_terminal_id()
        is_terminal = (agent_id == terminal_id)
        predecessors = self.mas.get_predecessors(agent_id)
        
        current_prompt = self.best_prompt[agent_id]
        desc_prefix = f"SPO-Agent-{agent_id}({agent.type.value})"
        
        with tqdm(total=rounds, desc=desc_prefix, leave=False) as pbar:
            for r in range(rounds):
                
                eval_samples = random.sample(self.train_q, min(sample_size, len(self.train_q)))
                current_map = self.best_prompt.copy()
                baseline_caches = await self._build_baseline_caches(eval_samples, current_map)
                
                qa_for_proposal = {}
                for q in eval_samples:
                    cache = baseline_caches[q]
                    ctx = cache.get_context_for_node(agent_id, predecessors)
                    output = cache.node_outputs_raw.get(agent_id, "")
                    qa_for_proposal[q] = {"context": ctx, "output": output}
                
                sample_q = eval_samples[0]
                new_prompt = await self._propose_new_prompt(
                    requirement, current_prompt, qa_for_proposal, agent_id, sample_q
                )
                
                if new_prompt == current_prompt:
                    pbar.set_postfix_str(f"Skip (Same)")
                    pbar.update(1)
                    continue

                cand_map = self.best_prompt.copy()
                cand_map[agent_id] = new_prompt
                
                score_info = await self._evaluate_candidate(
                    cand_prompt=new_prompt,
                    agent_id=agent_id,
                    eval_samples=eval_samples,
                    baseline_caches=baseline_caches,
                    temp_prompt_map=cand_map,
                    predecessors=predecessors,
                    terminal_id=terminal_id,
                    is_terminal=is_terminal,
                    requirement=requirement,
                    pure_local_mode=True 
                )
                
                if score_info["score"] > 0:
                    self.best_prompt[agent_id] = new_prompt
                    current_prompt = new_prompt
                    pbar.set_postfix_str(f"Update! Score: {score_info['score'] + 0.5:.2f}")
                else:
                    pbar.set_postfix_str(f"Keep. Score: {score_info['score'] + 0.5:.2f}")
                
                pbar.update(1)

        return current_prompt

    async def optimize_all_simple_sequential(self, requirement: str = " ", 
                                             rounds: int = 9) -> Dict[int, str]:
        order = self.mas._topo_order()
        
        with tqdm(total=len(order), desc="[SPO]") as topo_bar:
            for aid in order:
                if self.mas.agents[aid].type == AgentType.AGGREGATOR:
                    topo_bar.update(1)
                    continue
                await self._optimize_agent_simple_serial(
                    agent_id=aid, 
                    requirement=requirement, 
                    rounds=rounds, 
                    sample_size=3
                )
                
                topo_bar.set_postfix_str(f"Agent-{aid} Done")
                topo_bar.update(1)
        
        return self.best_prompt
