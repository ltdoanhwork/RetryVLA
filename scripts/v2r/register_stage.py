"""Stage registration for V2R (run on the agent box by scripts/v2r/watch.py; manual: python scripts/v2r/register_stage.py R1).
analyze_v2r -> RESULTS auto table + NOTES line + REGISTRY + review auto section -> commit + push (explicit paths)."""
import json, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
V = REPO / "runs/v2r"
HOST = "agent"


def register(stage):
    subprocess.call(["bash", "-lc", f"cd {REPO} && source env/activate.sh >/dev/null 2>&1 && "
                     f"python scripts/v2r/analyze_v2r.py > runs/_analysis/v2r_analyze.log 2>&1"])
    md = REPO / "runs/_analysis/v2r/v2r.md"
    body = md.read_text() if md.exists() else "(analysis failed, see runs/_analysis/v2r_analyze.log)"
    with open(REPO / "experiments/RESULTS.md", "a") as f:
        f.write(f"\n## V2R auto table after {stage} ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; "
                "pre-registration PLAN \"V2R\"; agent reading follows separately)\n"
                "`scripts/v2r/analyze_v2r.py` -> runs/_analysis/v2r/v2r.{md,json}.\n\n" + body + "\n")
    with open(REPO / "experiments/NOTES.md", "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}: v2r chain ({HOST}): stage {stage} complete, "
                "auto table appended to RESULTS (reading pending).\n")
    rv = REPO / "docs/research/REVIEW_2026-11-25_RECOVER.md"
    if rv.exists():   # keep the review's auto-results section current (between the markers)
        t = rv.read_text()
        a, b = "<!-- V2R_AUTO_BEGIN -->", "<!-- V2R_AUTO_END -->"
        if a in t and b in t:
            t = t.split(a)[0] + a + f"\n_Auto-updated by the chain after {stage}, " \
                f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}._\n\n" + body + "\n" + b + t.split(b)[1]
            rv.write_text(t)
    subprocess.call(["bash", "-lc", f"cd {REPO} && python scripts/v2r/register_runs.py >> runs/_analysis/v2r_analyze.log 2>&1; "
                     f"scripts/git_commit.sh 'V2R {stage}: results (auto table) + registry' experiments/RESULTS.md "
                     "experiments/NOTES.md experiments/REGISTRY.md docs/research/REVIEW_2026-11-25_RECOVER.md && (env -u LD_LIBRARY_PATH git push -q origin main || "
                     "(env -u LD_LIBRARY_PATH git pull -q --rebase origin main && env -u LD_LIBRARY_PATH git push -q origin main))"])



if __name__ == "__main__":
    st = sys.argv[1]
    register(st)
    (V / f"register_done_{st}").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
