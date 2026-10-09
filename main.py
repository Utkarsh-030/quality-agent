"""CLI entry point.

  python main.py --mock                       # offline demo, no API key needed
  python main.py --data sample_data/batch_ok.json --mock
  GEMINI_API_KEY=... python main.py           # real LLM
"""
import argparse
import json

from agent import GeminiLLM, MockLLM, QualityAgent


def main() -> None:
    p = argparse.ArgumentParser(description="Manufacturing Quality Inspection Agent")
    p.add_argument("--data", default="sample_data/batch_bad.json")
    p.add_argument("--mock", action="store_true", help="use the offline mock LLM")
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()

    with open(args.data) as f:
        task = json.load(f)
    llm = MockLLM() if args.mock else GeminiLLM()
    out = QualityAgent(llm, max_steps=args.max_steps).run(task)
    print("\n=== FINAL REPORT ===")
    print(json.dumps(out["result"], indent=2))


if __name__ == "__main__":
    main()
