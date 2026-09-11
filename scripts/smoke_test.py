from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.taxonomy import weak_label
from src.policy import decide

assert weak_label("I forgot my Apple ID password") == "account_access"
assert decide("account_access", 0.99, 0.9, "forgot password")["auto_handle"] is False
print("Smoke test passed.")
