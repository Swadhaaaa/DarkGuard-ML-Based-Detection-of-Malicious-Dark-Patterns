"""
DarkGuard — Inference Verification Script
Loads the saved model and runs 5 test cases.
"""
import json
from dark_pattern_ml import DarkPatternPredictor

predictor = DarkPatternPredictor.load("model_artifacts")

cases = [
    ("Amazon Flash Sale",   dict(Fake_Urgency=1,Scarcity=1,Confusing_Text=0,Hidden_Cost=1,Forced_Action=0,Social_Proof_Fake=1,Misdirection=0,Visual_Trick=1)),
    ("Clean E-commerce",    dict(Fake_Urgency=0,Scarcity=0,Confusing_Text=0,Hidden_Cost=0,Forced_Action=0,Social_Proof_Fake=0,Misdirection=0,Visual_Trick=0)),
    ("Subscription Trap",   dict(Fake_Urgency=1,Scarcity=0,Confusing_Text=1,Hidden_Cost=1,Forced_Action=1,Social_Proof_Fake=0,Misdirection=1,Visual_Trick=0)),
    ("Max Dark Pattern",    dict(Fake_Urgency=1,Scarcity=1,Confusing_Text=1,Hidden_Cost=1,Forced_Action=1,Social_Proof_Fake=1,Misdirection=1,Visual_Trick=1)),
    ("Single Timer Only",   dict(Fake_Urgency=1,Scarcity=0,Confusing_Text=0,Hidden_Cost=0,Forced_Action=0,Social_Proof_Fake=0,Misdirection=0,Visual_Trick=0)),
]

print("=" * 75)
print("  DARKGUARD ML - INFERENCE VERIFICATION")
print("=" * 75)
print(f"  {'Page':<26}  {'Verdict':<22}  {'Confidence':>11}  {'Threat':>10}")
print("  " + "-" * 71)

last_result = None
for name, sig in cases:
    r = predictor.predict(sig)
    verdict = "DARK PATTERN" if r["is_dark_pattern"] else "CLEAN"
    print(f"  {name:<26}  {verdict:<22}  {r['confidence']*100:>10.1f}%  {r['threat_level']:>10}")
    last_result = r

print("=" * 75)
print("\nFull JSON output for last case:")
print(json.dumps(last_result, indent=2))

# Batch test
print("\nBatch prediction test (10 records)...")
records = [sig for _, sig in cases * 2]
results = predictor.predict_batch(records)
print(f"Batch OK: {len(results)} results returned.")
print("INFERENCE VERIFICATION COMPLETE.")
