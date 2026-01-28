import json
import time
import random
import asyncio
import argparse
from typing import List, Dict, Any, Optional

from tqdm.asyncio import tqdm_asyncio

from config import (
    GraphType, AgentType, TaskType, DATASET_CONFIG,
    aclient, bclient, create_evaluator_client
)
from prompts import OPTIMIZATION_REQUIREMENTS
from utils import normalize_answer, extract_output, extract_function_name_from_tests
from data_loaders import load_test_data, load_train_for_opt, get_task_type, get_default_use_judge
from agents import MAS
from optimizers import MAPromptOptimizer


async def aprocess_single(item: Dict[str, Any], graph_type: GraphType, 
                          task_type: TaskType, prompt_map: Optional[Dict[int, str]] = None,
                          use_judge: bool = False, nr: int = 1) -> Dict[str, Any]:
    problem = item['problem']
    unique_id = item['unique_id']

    if task_type == TaskType.CODE and 'test_list' in item:
        func_name = extract_function_name_from_tests(item['test_list'])
        if func_name:
            problem = f"{problem}\n\nNote: The function should be named '{func_name}'."
    try:
        mas = MAS(graph_type, task_type, use_judge=use_judge, judge_client=bclient, Nr=nr)
        if prompt_map:
            mas.inject_prompt_map(prompt_map)
        
        start = time.time()
        result = await mas.arun(problem)
        end = time.time()
        final_output = result['final']
        terminal_id = mas.get_terminal_id()
        final_raw = result["raw_trace"][terminal_id]
        
        if task_type == TaskType.CODE:
            if use_judge and mas.judge and item.get('test_list'):
                is_correct = await mas.judge.ajudge(final_output, item['test_list'])
            else:
                is_correct = "def " in final_output
        else:
            if use_judge and mas.judge:
                is_correct = await mas.judge.ajudge(problem, item['answer'], final_raw)
            else:
                correct_answer = normalize_answer(item['answer'])
                model_answer = normalize_answer(final_output)
                is_correct = model_answer == correct_answer
                if task_type in [TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE]:
                    is_correct = is_correct or correct_answer in model_answer
        return {
            "unique_id": unique_id,
            "graph_type": graph_type.value,
            "output": final_output,
            "final_raw": final_raw,
            "correct": is_correct,
            "response_time": end - start,
            "full_output": result,
            "error": None
        }
    except Exception as e:
        return {
            "unique_id": unique_id,
            "graph_type": graph_type.value,
            "output": None,
            "final_raw": None,
            "correct": False,
            "response_time": 0,
            "full_output": None,
            "error": str(e)
        }
async def aprocess_task(item: Dict[str, Any], graph_types: List[GraphType],
                        task_type: TaskType, prompt_map: Optional[Dict[int, str]] = None,
                        use_judge: bool = False, nr: int = 1) -> List[Dict[str, Any]]:

    return [await aprocess_single(item, gt, task_type, prompt_map, use_judge) 
            for gt in graph_types]
async def limited_aprocess_task(item: Dict[str, Any], graph_types: List[GraphType],
                                 task_type: TaskType, sem: asyncio.Semaphore,
                                 prompt_map: Optional[Dict[int, str]] = None,
                                 use_judge: bool = False, nr: int = 1):
    async with sem:
        return await aprocess_task(item, graph_types, task_type, prompt_map, use_judge, nr=nr)
# ------------------ 测试套件 ------------------
async def arun_test_suite(data: List[Dict[str, Any]],
                          task_type: TaskType,
                          graph_types: List[GraphType] = None,
                          sample_size: int = None,
                          output_file: str = "test_results.json",
                          max_concurrent: int = 20,
                          prompt_map: Optional[Dict[int, str]] = None,
                          use_judge: bool = False,
                          nr: int = 1):
    if graph_types is None:
        graph_types = list(GraphType)
    
    original_len = len(data)
    if sample_size and sample_size < original_len:
        data = random.sample(data, sample_size)
        print(f"{original_len} samples {sample_size}")
    else:
        print(f"use {original_len} samples")
    task_sem = asyncio.Semaphore(max_concurrent)
    results = {
        "total": len(data),
        "task_type": task_type.value,
        "graph_types": {gt.value: {"correct": 0, "total": 0, "accuracy": 0.0, "avg_response_time": 0.0}
                       for gt in graph_types},
        "detailed": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "use_judge": use_judge,
    }
    start_total = time.time()
    tasks = [
        limited_aprocess_task(item, graph_types, task_type, task_sem, prompt_map, use_judge)
        for item in data
    ]
    results_list = await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="任务进度")
    flat_results = [r for sub in results_list for r in sub]
    # 整理结果
    detailed_map = {}
    for r in flat_results:
        uid, gt_val = r["unique_id"], r["graph_type"]
        if uid not in detailed_map:
            orig = next(item for item in data if item["unique_id"] == uid)
            detailed_map[uid] = {
                "unique_id": uid,
                "problem": orig["problem"],
                "models": {},
            }
            if task_type == TaskType.CODE:
                detailed_map[uid]["test_list"] = orig.get("test_list", [])
            else:
                detailed_map[uid]["correct_answer"] = normalize_answer(orig["answer"])
        
        detailed_map[uid]["models"][gt_val] = {
            "output": r["output"],
            "raw_output": r["final_raw"],
            "correct": r["correct"],
            "response_time": r["response_time"],
            "error": r["error"],
        }
        
        st = results["graph_types"][gt_val]
        st["total"] += 1
        if r["correct"]:
            st["correct"] += 1
        st["avg_response_time"] += r["response_time"]
    for gt in graph_types:
        st = results["graph_types"][gt.value]
        if st["total"]:
            st["accuracy"] = st["correct"] / st["total"]
            st["avg_response_time"] /= st["total"]
    results["detailed"] = list(detailed_map.values())
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["duration"] = time.time() - start_total
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run MAS evaluation with configurable dataset and prompt mode.")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=list(DATASET_CONFIG.keys()),
                        help=f"Dataset to use: {', '.join(DATASET_CONFIG.keys())}")
    parser.add_argument("--graph", type=str, default="reflect",
                        choices=[gt.value for gt in GraphType],
                        help="Graph type to use")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable prompt optimization. If not set, use original prompts.")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use LLM judge for non-code tasks (code tasks always use code judge)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of samples to use (default: all)")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Maximum concurrent tasks")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to pre-optimized prompt file (JSON)")
    
    parser.add_argument("--round-robin", action="store_true",
                        help="Use round-robin optimization instead of sequential optimization")
    parser.add_argument("--depth", type=int, default=10,
                        help="Maximum optimization rounds per agent (for round-robin mode)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience: pause agent after N rounds without improvement")
    parser.add_argument("--fixed-rounds", action="store_true",
                    help="Use fixed rounds per turn optimization (2 rounds per agent each turn)")
    parser.add_argument("--dynamic-switching", action="store_true",
                    help="Enable Dynamic Anchor Switching for joint optimization (zero-cost)")
    parser.add_argument("--stochastic-sampling", action="store_true",
                    help="Enable Stochastic Beam Context Sampling for robust joint optimization")
    parser.add_argument("--beam-refresh", action="store_true",
                    help="Enable Beam Refresh strategy to re-score nodes based on updated partners")
    parser.add_argument("--nr", type=int, default=1,
                    help="Number of reflect rounds (for reflect topology)")
    parser.add_argument("--feedback", action="store_true",
                    help="Enable multi-agent collaborative feedback (pass bad cases from downstream to upstream).")
    parser.add_argument("--misleading-sampling", action="store_true",
                    help="Enable sampling injection of upstream 'misleading cases' (Local Win / Global Lose).")
    parser.add_argument("--lookahead-score", action="store_true",
                    help="Enable Lookahead Scoring: 0.5*Local + 0.3*Next_Local + 0.2*Global")
    parser.add_argument("--lookahead-weights", type=str, default="4:4:2",
                        help="Weights for lookahead scoring (Local:Next:Global), e.g., '4:4:2'. Default is 4:4:2 (0.4, 0.4, 0.2)")

    
    args = parser.parse_args()
    dataset = args.dataset
    graph_type = GraphType(args.graph)
    task_type = get_task_type(dataset)
    
    default_use_judge = get_default_use_judge(dataset)
    if task_type == TaskType.CODE:
        use_judge = True
    else:
        use_judge = args.use_llm_judge
    
    try:
        w_parts = [float(x) for x in args.lookahead_weights.split(':')]
        if len(w_parts) != 3:
            raise ValueError("Must provide exactly 3 numbers separated by colon.")
        total_w = sum(w_parts)
        if total_w == 0:
            raise ValueError("Sum of weights cannot be 0.")
        lookahead_weights = tuple(w / total_w for w in w_parts)
    except Exception as e:
        print(f"[Warning] Invalid lookahead-weights format ({e}), using default 0.4:0.4:0.2")
        lookahead_weights = (0.4, 0.4, 0.2)
    print(f"Lookahead Weights: {args.lookahead_weights} -> Local:{lookahead_weights[0]:.2f}, Next:{lookahead_weights[1]:.2f}, Global:{lookahead_weights[2]:.2f}")

    print(f"Dataset: {dataset}")
    print(f"Task Type: {task_type.value}")
    print(f"Graph Type: {graph_type.value}")
    print(f"Optimize: {args.optimize}")
    print(f"Round-Robin Mode: {args.round_robin}")
    if args.round_robin:
        print(f"  - Depth per Agent: {args.depth}")
        print(f"  - Patience: {args.patience}")
    print(f"Use Judge: {use_judge}" + (" (code judge)" if task_type == TaskType.CODE else " (LLM judge)" if use_judge else " (string match)"))
    prompt_map = None
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_map = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded prompts from {args.prompt_file}")
    if args.optimize and not args.prompt_file:
        
        train_problems = load_train_for_opt(dataset, k=50)
        mas = MAS(graph_type, task_type, Nr=args.nr)
        
        seed_map = {
            i: mas.agents[i].template 
            for i in range(len(mas.agents))
            if mas.agents[i].type != AgentType.AGGREGATOR and mas.agents[i].template
        }
        
        requirement = OPTIMIZATION_REQUIREMENTS.get(task_type, "")
        evaluator_client = create_evaluator_client()
        
        optimizer = MAPromptOptimizer(mas, train_problems, seed_map, evaluator_client)
        statistics = None
        
        if args.round_robin:
            prompt_map = asyncio.run(optimizer.optimize_all_round_robin(
                requirement=requirement,
                max_total_depth=args.depth,
                patience=args.patience,
                beam_width=2
            ))
        elif args.fixed_rounds:
            prompt_map, statistics = asyncio.run(optimizer.optimize_all_fixed_rounds(
                requirement=requirement,
                max_total_depth=9,
                rounds_per_turn=3,
                beam_width=2,
                use_dynamic_switching=args.dynamic_switching,
                use_stochastic_sampling=args.stochastic_sampling,
                use_beam_refresh=args.beam_refresh,
                use_feedback=args.feedback,
                use_misleading_sampling=args.misleading_sampling,
                use_lookahead_score=args.lookahead_score,
                lookahead_weights=lookahead_weights
            ))
        else:
            prompt_map = asyncio.run(optimizer.optimize_all(requirement=requirement))
        
        if args.round_robin:
            mode_suffix = "rr"
        elif args.lookahead_score and args.misleading_sampling:
            mode_suffix = "ms_ls"
        elif args.misleading_sampling:
            mode_suffix = "ms"
        elif args.lookahead_score:
            mode_suffix = "ls"
        else:
            mode_suffix = "topo"
        prompt_output_file = f"prompt/tbdspo_{dataset}_{graph_type.value}_{mode_suffix}.json"
        with open(prompt_output_file, "w", encoding="utf-8") as f:
            json.dump(prompt_map, f, indent=2, ensure_ascii=False)
        print(f"Optimized prompts saved to {prompt_output_file}")

        if statistics:
            stats_output_file = f"stats/tbdspo_{dataset}_{graph_type.value}_{mode_suffix}_stats.json"

            import os
            os.makedirs("stats", exist_ok=True)
            
            with open(stats_output_file, "w", encoding="utf-8") as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            print(f"Optimization statistics saved to {stats_output_file}")
            
            
    
    test_data = load_test_data(dataset)
    
    suffix = "tbdspo" if (args.optimize or args.prompt_file) else "original"
    if args.round_robin and args.optimize:
        suffix = "tbdspo_rr"
    output_file = f"result/{dataset}_{graph_type.value}_{suffix}.json"
    
    asyncio.run(arun_test_suite(
        test_data,
        task_type=task_type,
        graph_types=[graph_type],
        sample_size=args.sample_size,
        output_file=output_file,
        max_concurrent=args.max_concurrent,
        prompt_map=prompt_map,
        use_judge=use_judge,
        nr=args.nr,
    ))

    asyncio.run(arun_test_suite(
        test_data,
        task_type=task_type,
        graph_types=[graph_type],
        sample_size=args.sample_size,
        output_file=output_file,
        max_concurrent=args.max_concurrent,
        prompt_map=prompt_map,
        use_judge=use_judge,
        nr=args.nr
    ))
if __name__ == "__main__":
    main()
