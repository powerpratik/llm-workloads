"""M3 - KV-cache growth in agentic trajectories (SWE-bench Verified via mini-SWE-agent; tau-bench).

For every LLM call in a trajectory we reconstruct the prompt (all previous messages; both
agents keep the full, unpruned history) and the decoded output. Per trajectory we report:
  steps, persistent prefix (system + task [+ tool schemas]), peak context (= KV footprint
  without eviction), total decoded tokens, share of the context that is environment
  observation vs. model output, and the cumulative prefill work with/without cross-step
  prefix reuse. Outputs: results/agent_trajectories.csv (per trajectory),
  results/agent_summary.csv (per agent/model), results/agent_growth.json (context-vs-step curves).
"""
import ast
import glob
import json

import numpy as np

from common import CHAT_MSG_OVERHEAD, RAW, RESULTS, ntok, summarize, write_csv, write_json

per_traj, curves = [], {}


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return json.dumps(content)


def analyse(workload, traj_id, roles, toks, prefix_extra=0, resolved=None):
    """roles/toks: per message. The model is called before every assistant message."""
    toks = [t + CHAT_MSG_OVERHEAD for t in toks]
    ctx, cum_no_reuse, steps, dec, obs, curve = prefix_extra, 0, 0, 0, 0, []
    first_asst = roles.index("assistant") if "assistant" in roles else len(roles)
    persistent = prefix_extra + sum(toks[:first_asst])
    peak = ctx
    for r, t in zip(roles, toks):
        if r == "assistant":
            steps += 1
            cum_no_reuse += ctx          # prompt re-prefilled from scratch at this call
            curve.append(ctx)
            dec += t
            peak = ctx + t               # KV held at the end of this call (prompt + its output)
            obs_at_peak = obs
        elif steps > 0:
            obs += t
        ctx += t
    # messages after the final LLM call (e.g. the observation logged after submission) never reach the model
    obs = obs_at_peak if steps else obs
    per_traj.append(dict(workload=workload, traj=traj_id, steps=steps, persistent_prefix=persistent, peak_context=peak,
                         decoded_total=dec, observation_total=obs, obs_share=obs / peak if peak else None,
                         decode_share=dec / peak if peak else None, persistent_share=persistent / peak if peak else None,
                         prefill_no_reuse=cum_no_reuse, prefill_with_prefix_reuse=peak - dec,
                         reuse_saving_x=(cum_no_reuse / max(1, peak - dec)), resolved=resolved))
    curves.setdefault(workload, []).append(curve)


def swebench():
    for d in sorted(p for p in (RAW / "swebench_trajs").iterdir() if p.is_dir()):
        sub = d.name
        model = sub.split("_", 2)[-1]
        files = sorted(glob.glob(str(d / "*.traj.json")))
        for f in files:
            s = json.load(open(f))
            msgs = s["messages"]
            roles = [m["role"] for m in msgs]
            toks = ntok([text_of(m.get("content")) for m in msgs])
            analyse(f"swebench-verified:{model}", s.get("instance_id", f), roles, toks)
        print(sub, len(files))


def tool_schema_tokens(env):
    infos = []
    for f in glob.glob(str(RAW / f"repos/taubench/tau_bench/envs/{env}/tools/*.py")):
        tree = ast.parse(open(f).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_info":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return):
                        try:
                            infos.append(ast.literal_eval(sub.value))
                        except Exception:
                            pass
    return ntok(json.dumps(infos))[0], len(infos)


def taubench():
    for f in sorted(glob.glob(str(RAW / "repos/taubench/historical_trajectories/*.json"))):
        name = f.split("/")[-1][:-5]
        env = "airline" if "airline" in name else "retail"
        schema_tok, n_tools = tool_schema_tokens(env)
        for t in json.load(open(f)):
            msgs = t["traj"]
            roles = [m["role"] for m in msgs]
            texts = [text_of(m.get("content")) or json.dumps(m.get("tool_calls")) for m in msgs]
            analyse(f"tau-bench:{name}", f"{t['task_id']}-{t.get('trial', 0)}", roles, ntok(texts),
                    prefix_extra=schema_tok, resolved=t.get("reward"))
        print(name, "tool schemas:", n_tools, "tools /", schema_tok, "tokens")


if __name__ == "__main__":
    swebench()
    taubench()
    write_csv(per_traj, RESULTS / "agent_trajectories.csv")
    summ = []
    for w in dict.fromkeys(r["workload"] for r in per_traj):
        rs = [r for r in per_traj if r["workload"] == w]
        row = dict(workload=w, n=len(rs))
        for k in ["steps", "persistent_prefix", "peak_context", "decoded_total", "obs_share", "decode_share",
                  "persistent_share", "reuse_saving_x"]:
            s = summarize([r[k] for r in rs])
            row.update({f"{k}_p10": s["p10"], f"{k}_p50": s["p50"], f"{k}_p90": s["p90"]})
        row["peak_context_max"] = max(r["peak_context"] for r in rs)
        summ.append(row)
        print(f"{w:55s} steps_p50={row['steps_p50']:.0f} peak_p50={row['peak_context_p50']:.0f} "
              f"peak_p90={row['peak_context_p90']:.0f} dec_share={row['decode_share_p50']:.2f} "
              f"obs_share={row['obs_share_p50']:.2f} persist={row['persistent_prefix_p50']:.0f} reuse_x={row['reuse_saving_x_p50']:.1f}")
    write_csv(summ, RESULTS / "agent_summary.csv")
    # keep growth curves compact: context size at each step for up to 150 trajectories per workload
    write_json({w: c[:150] for w, c in curves.items()}, RESULTS / "agent_growth.json")
