# sdk_probe.py
import os, sys
import google.generativeai as genai

print("google-generativeai version:", getattr(genai, "__version__", "unknown"))

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "AIzaSyBM1upg3vXDjYKJVY740gI4y1kJHHusCUk")
if not api_key:
    print("ERROR: No GEMINI_API_KEY/GOOGLE_API_KEY in env")
    sys.exit(1)

genai.configure(api_key=api_key)

names = []
try:
    for m in genai.list_models():
        # Only keep models that support text generation
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            names.append(m.name.split("/")[-1])
except Exception as e:
    print("ERROR listing models:", repr(e))
    sys.exit(1)

names = sorted(set(names))
print("\nSupported text models for your key:")
for n in names:
    print(" -", n)

# Choose the best candidate we recognize
CANDIDATES = [
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro",
    "gemini-pro",
]
available = [c for c in CANDIDATES if c in names]
print("\nBest available from known candidates:", available[0] if available else "NONE")
