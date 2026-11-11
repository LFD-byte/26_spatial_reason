"""阶段 1 第 2 步:用 vLLM 跑教师模型(在单独的 vllm conda 环境里运行,只依赖 vllm 和标准库)。

    ~/software/anaconda3/envs/vllm/bin/python scripts/05_run_teacher_vllm.py \
        --model data/models/Qwen3-32B --tp 2 \
        --requests data/traces/qwen3_32b/requests_train.jsonl \
        --output data/traces/qwen3_32b/teacher_train.jsonl

每条请求采 n 条回答;支持断点续跑(已写出的 id 会跳过)。
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=640)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=2000, help="每批提交的请求数(便于断点续跑)")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 条(试跑用)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true", help="关闭 CUDA graph / torch.compile")
    p.add_argument("--disable-custom-all-reduce", action="store_true",
                   help="两卡无 P2P(nvidia-smi topo 为 SYS)时用 NCCL 而非 vLLM 自定义 all-reduce")
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    reqs = [json.loads(line) for line in open(args.requests)]
    if args.limit:
        reqs = reqs[: args.limit]
    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in open(out_path)}
    todo = [r for r in reqs if r["id"] not in done]
    print(f"[teacher] {len(reqs)} requests, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True, seed=0,
              enforce_eager=args.enforce_eager, disable_custom_all_reduce=args.disable_custom_all_reduce)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                        seed=0)
    tok = llm.get_tokenizer()
    with open(out_path, "a") as f:
        for s in range(0, len(todo), args.chunk):
            batch = todo[s:s + args.chunk]
            prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for r in batch]
            outs = llm.generate(prompts, sp)
            for r, o in zip(batch, outs):
                f.write(json.dumps({"id": r["id"], "outputs": [c.text for c in o.outputs],
                                    "finish_reasons": [c.finish_reason for c in o.outputs]},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"[teacher] {min(s + args.chunk, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
