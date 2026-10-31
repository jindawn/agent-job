from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bootstrap import build_agent
from .config import ROOT, Settings


def _ask(agent, message: str, role: str, session_id: str | None, interactive: bool, auto: bool | None):
    res = agent.run(message, session_id, role)
    while res.status == "awaiting_approval":
        for c in res.pending_approvals:
            print(f"  ⚠ 需要确认: {c['name']}({c['arguments']})")
        if auto is None:
            ok = interactive and input("  是否执行? [y/N] ").strip().lower() == "y"
        else:
            ok = auto
        res = agent.resume(res.session_id, {c["id"]: ok for c in res.pending_approvals})
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agenthub")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="one-shot question")
    a.add_argument("question")
    a.add_argument("--role", default="support")
    a.add_argument("--yes", action="store_true", help="auto-approve side-effecting tools")
    a.add_argument("--trace", action="store_true")
    c = sub.add_parser("chat", help="interactive chat")
    c.add_argument("--role", default="support")
    e = sub.add_parser("eval", help="run the evaluation suite")
    e.add_argument("--min-pass", type=float, default=0.0)
    e.add_argument("--out", default=str(ROOT / "reports" / "latest.json"))
    sv = sub.add_parser("serve", help="start the HTTP API (needs the [server] extra)")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sub.add_parser("mcp", help="run the MCP stdio server")
    args = p.parse_args(argv)

    if args.cmd == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()
        return 0
    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("agenthub.server:app", host=args.host, port=args.port)
        return 0

    s = Settings.from_env()
    agent = build_agent(s)
    if args.cmd == "ask":
        res = _ask(agent, args.question, args.role, None, sys.stdin.isatty(), True if args.yes else None)
        print(res.answer)
        if args.trace:
            import json
            print(json.dumps(res.trace, ensure_ascii=False, indent=2))
    elif args.cmd == "chat":
        print(f"AgentHub ({s.provider}/{getattr(agent.llm, 'model', '?')}) - 输入 exit 退出")
        sid = None
        while True:
            try:
                q = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in ("exit", "quit"):
                break
            if q:
                res = _ask(agent, q, args.role, sid, True, None)
                sid = res.session_id
                print(f"助手> {res.answer}\n")
    elif args.cmd == "eval":
        from .evals.runner import format_report, run_all, save_report
        results, summary = run_all(agent)
        model = getattr(agent.llm, "model", "?")
        print(format_report(results, summary, model))
        save_report(results, summary, model, Path(args.out))
        return 0 if summary["pass_rate"] >= args.min_pass else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
