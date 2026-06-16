"""Terminal chat. Usage:
  python cli.py                # agentic mode (default)
  python cli.py --simple       # single-shot hybrid RAG (no agent loop)
"""
import sys

import rag
import agent


def render_sources(sources):
    print("\n\nSources:")
    for s in sources:
        loc = f"{s['title']} — {s['section']}"
        print(f"  [{s['n']}] {loc}  {s['url']}")


def main():
    simple = "--simple" in sys.argv
    mode = "simple hybrid RAG" if simple else "agentic RAG"
    print(f"Game of Thrones assistant ({mode}). Ctrl-C to quit.\n")
    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            print("\nAssistant: ", end="", flush=True)
            if simple:
                stream, sources = rag.answer(q)
            else:
                stream, sources = agent.answer(q, verbose=True)
            for tok in stream:
                print(tok, end="", flush=True)
            render_sources(sources)
            print("\n" + "-" * 60)
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
