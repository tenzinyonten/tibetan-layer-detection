#!/usr/bin/env python3
"""Tool-calling wrapper around title_author_skill.extract, for any LLM that supports function calling.

    from tool_handler import handle_tool_call, claude_tool, openai_tool, gemini_tool
    # 1. give the model one of the tool definitions (claude_tool() / openai_tool() / gemini_tool())
    # 2. when the model calls it:  result_json = handle_tool_call(name, arguments_dict, call_llm=your_function)
    # 3. send result_json back to the model as the tool result.

`call_llm(system_prompt, user_text) -> str` is the function that calls the LLM that does the extraction (it can be a different model
from the one that decides to call the tool). With call_llm=None the rule-based answer is returned (no network).
Self-test, no key and no network:  python3 title_author/skill/tool_handler.py selftest"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import title_author_skill as S

SCHEMA = json.loads((HERE / "tool_schema.json").read_text(encoding="utf-8"))
NAME = SCHEMA["name"]


def claude_tool():
    """Anthropic Messages API tool definition."""
    return {"name": NAME, "description": SCHEMA["description"], "input_schema": SCHEMA["parameters"]}


def openai_tool():
    """OpenAI function/tool definition (chat completions)."""
    return {"type": "function", "function": {"name": NAME, "description": SCHEMA["description"], "parameters": SCHEMA["parameters"]}}


def gemini_tool():
    """Gemini functionDeclarations (types in upper case)."""
    def up(x):
        if isinstance(x, dict): return {k: (v.upper() if k == "type" and isinstance(v, str) else up(v)) for k, v in x.items()}
        return x
    return {"functionDeclarations": [{"name": NAME, "description": SCHEMA["description"], "parameters": up(SCHEMA["parameters"])}]}


def handle_tool_call(name, arguments, call_llm=None):
    """Run the tool the model asked for. `arguments` is a dict (or a JSON string). Returns a JSON string (the tool result)."""
    if name != NAME:
        return json.dumps({"error": f"unknown tool {name!r}"})
    if isinstance(arguments, str):
        try: arguments = json.loads(arguments)
        except Exception: return json.dumps({"error": "arguments are not valid JSON"})
    head = (arguments or {}).get("text_head")
    if not head:
        return json.dumps({"error": "text_head is required"})
    whole = bool(arguments.get("whole_work", False)) or not arguments.get("text_tail")
    res = S.extract(head, arguments.get("text_tail") or "", whole, call_llm=call_llm)
    return json.dumps({k: res.get(k) for k in ("title", "title_region", "author", "author_region", "notes", "needs_review", "review_reasons")}, ensure_ascii=False)


def selftest():
    head = "༄༅། །ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་བཞུགས།། ཚེས་བཅུའི་མཆོད་པའི་ཡི་གེ་དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན་གྱིས་སྦྱར་བའོ།།"
    fake = lambda s, u: json.dumps({"title": "ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་", "title_region": "head", "author": "གྲགས་པ་རྒྱལ་མཚན", "author_region": "head"})
    out = json.loads(handle_tool_call(NAME, {"text_head": head, "whole_work": True}, call_llm=fake))
    assert out["author"] == "དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན", out                      # role word restored by the rule layer
    out2 = json.loads(handle_tool_call(NAME, json.dumps({"text_head": head}), call_llm=None))   # rule-only, arguments as a JSON string
    assert out2["author"], out2
    assert json.loads(handle_tool_call("nope", {}))["error"]
    assert json.loads(handle_tool_call(NAME, {}))["error"]
    assert claude_tool()["input_schema"]["required"] == ["text_head"] and openai_tool()["function"]["name"] == NAME
    assert gemini_tool()["functionDeclarations"][0]["parameters"]["type"] == "OBJECT"
    print("tool_handler selftest OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]: selftest()
    else: sys.exit(__doc__)
