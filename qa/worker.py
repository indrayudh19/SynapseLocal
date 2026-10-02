"""
qa/worker.py
Subprocess entry point for Q2 (retrieve) and Q3 (rerank).
Usage: python -m qa.worker <stage> <session_id> <run_id>

Runs the specified stage, prints a single JSON status line, exits non-zero on failure.
"""
import json
import sys
import traceback


def main():
    if len(sys.argv) != 4:
        print(json.dumps({"status": "error", "message": "Usage: python -m qa.worker <stage> <session_id> <run_id>"}))
        sys.exit(1)

    stage = sys.argv[1]
    session_id = sys.argv[2]
    run_id = sys.argv[3]

    try:
        if stage == "retrieve":
            from qa.retrieve import run
            result = run(session_id, run_id)
        elif stage == "rerank":
            from qa.rerank import run
            result = run(session_id, run_id)
        else:
            result = {"status": "error", "message": f"Unknown stage: {stage}"}
            print(json.dumps(result))
            sys.exit(1)

        print(json.dumps({"status": result.get("status", "ok")}))
        sys.exit(0)

    except Exception as e:
        error_msg = f"{stage} failed: {str(e)}"
        # Write error.json to run directory
        try:
            from qa import paths as qa_paths
            err_path = qa_paths.error_path(session_id, run_id)
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump({"stage": stage, "error": error_msg, "traceback": traceback.format_exc()}, f)
        except Exception:
            pass

        print(json.dumps({"status": "error", "message": error_msg}))
        sys.exit(1)


if __name__ == "__main__":
    main()
