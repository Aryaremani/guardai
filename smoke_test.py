import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
from predict import run_prediction

r = run_prediction("You are an absolute idiot, I hate you.")
print("is_safe:", r["is_safe"])
for k, v in r["details"].items():
    print(f"  {k}: {v['probability']}%  flag={v['flag']}")
print("SMOKE TEST PASSED")
