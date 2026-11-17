import base64, json, os, re, time
import httpx

HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11500")
URL = f"http://{HOST}/api/chat"

def chat(model, prompt, think=None, fmt=None, image_b64=None):
    msg = {"role": "user", "content": prompt}
    if image_b64:
        msg["images"] = [image_b64]
    payload = {"model": model, "messages": [msg], "stream": False,
               "options": {"temperature": 0, "seed": 42}}
    if think is not None:
        payload["think"] = think
    if fmt:
        payload["format"] = fmt
    t = time.perf_counter()
    r = httpx.post(URL, json=payload, timeout=900)
    r.raise_for_status()
    d = r.json()
    text = d["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.split("</think>")[-1].strip()
    return text, time.perf_counter() - t, d.get("eval_count", 0)

def parse_json(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

ALLOWED = {"summarizer", "math_checker", "fig_checker", "ref_verifier",
           "novelty", "methodology", "final_reviewer"}

def test_planner():
    prompt = (
        "You are a planner for a paper-review system. Allowed agents: "
        + ", ".join(sorted(ALLOWED)) + ".\n"
        "Paper abstract: We derive closed-form SINR expressions for cell-free massive MIMO "
        "with MMSE channel estimation and prove a monotonicity result in the number of APs.\n"
        "Section headings: Introduction; System Model; Channel Estimation; Derivation of SINR; "
        "Proof of Theorem 1; Numerical Results; Conclusion.\n"
        'Return ONLY JSON: {"tasks":[{"agent":str,"target":str,"priority":int,"est_difficulty":float}]}. '
        "Always include summarizer and final_reviewer."
    )
    text, lat, tok = chat("qwen3:4b-instruct", prompt, fmt="json")
    d = parse_json(text)
    ok = bool(d and isinstance(d.get("tasks"), list) and d["tasks"])
    agents = {t.get("agent") for t in d["tasks"]} if ok else set()
    ok = ok and agents <= ALLOWED and {"math_checker", "final_reviewer"} <= agents
    return ok, lat, tok, f"agents={sorted(agents)}"

MATH = ("Check this derivation step by step. Claim: (a+b)^2 - (a-b)^2 = {res}.\n"
        "Step 1: (a+b)^2 = a^2 + 2ab + b^2.\n"
        "Step 2: (a-b)^2 = a^2 - 2ab + b^2.\n"
        "Step 3: Subtracting, the a^2 and b^2 terms cancel and we get {res}.\n"
        'Finish with one line of JSON: {{"error_found": true/false, "step": int, "explanation": str}}')

def test_math(planted):
    res = "2ab" if planted else "4ab"
    text, lat, tok = chat("deepseek-r1:7b", MATH.format(res=res))
    d = parse_json(text)
    got = d.get("error_found") if d else None
    return got == planted, lat, tok, f"error_found={got} (expected {planted})"

def test_methodology():
    prompt = ("Review this methods paragraph. We train a channel estimator on 1000 channel "
              "realisations and report NMSE on the same 1000 realisations. We compare against no "
              "other method and conclude our estimator is the best known.\n"
              'Return ONLY JSON: {"findings":[{"issue":str,"severity":"low|medium|high"}]}')
    text, lat, tok = chat("qwen3:14b", prompt, think=False, fmt="json")
    d = parse_json(text)
    ok = bool(d and d.get("findings"))
    return ok, lat, tok, text.replace("\n", " ")[:300]

def make_chart():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["A", "B", "C"], [3, 7, 5])
    ax.set_ylabel("Score")
    fig.savefig("/tmp/test_chart.png", dpi=100)
    with open("/tmp/test_chart.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

def test_vision(model, img):
    prompt = 'Which bar is tallest and what is its value? Return ONLY JSON: {"bar": str, "value": number}'
    text, lat, tok = chat(model, prompt, image_b64=img)
    d = parse_json(text)
    ok = bool(d and str(d.get("bar")).strip().upper() == "B" and abs(float(d.get("value", 0)) - 7) < 1)
    return ok, lat, tok, text.replace("\n", " ")[:150]

if __name__ == "__main__":
    tests = [("planner  qwen3:4b-instruct", test_planner),
             ("math PLANTED r1:7b", lambda: test_math(True)),
             ("math CLEAN   r1:7b", lambda: test_math(False)),
             ("methodology qwen3:14b", test_methodology)]
    img = make_chart()
    if img:
        tests += [(f"vision {m}", (lambda m=m: test_vision(m, img)))
                  for m in ("qwen2.5vl:7b", "gemma3:4b")]
    else:
        print("matplotlib not found in the container: skipping vision tests")
    for name, fn in tests:
        try:
            ok, lat, tok, detail = fn()
            print(f"[{'PASS' if ok else 'FAIL'}] {name:28s} {lat:6.1f}s {tok:5d} tok | {detail}")
        except Exception as e:
            print(f"[ERR ] {name:28s} {e}")