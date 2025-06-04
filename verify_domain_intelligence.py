#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 19:11:25 2025

@author: rameshbk
"""

import os
import json
import importlib.util

# Paths
DOMAIN_PROMPT_FILE = "generated_prompted_qa_from_domain_knowledge.jsonl"
DOMAIN_MODULES = [
    "domain_knowledge.py",
    "domain_knowledge_updated.py",
    "enhanced_prompt.py",
    "enhanced_prompt_updated.py"
]

print("🔍 Checking environment configuration...")

# Check ENV variables
print("📌 OLLAMA_URL:", os.getenv("OLLAMA_URL"))
print("📌 MODEL_NAME:", os.getenv("MODEL_NAME"))
print("📌 CACHE_DIR:", os.getenv("CACHE_DIR"))

# Check domain prompt file
print("\n📂 Checking for prompt JSONL file...")
if os.path.exists(DOMAIN_PROMPT_FILE):
    with open(DOMAIN_PROMPT_FILE, "r") as f:
        examples = [json.loads(line) for line in f.readlines()]
        print(f"✅ Found prompt file: {DOMAIN_PROMPT_FILE} with {len(examples)} examples.")
else:
    print(f"❌ Prompt file missing: {DOMAIN_PROMPT_FILE}")

# Check if domain modules are available and importable
print("\n📦 Verifying domain-specific modules:")
for module_file in DOMAIN_MODULES:
    if os.path.exists(module_file):
        spec = importlib.util.spec_from_file_location("domain_module", module_file)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            print(f"✅ Module loaded: {module_file}")
        except Exception as e:
            print(f"❌ Failed to load {module_file}: {e}")
    else:
        print(f"❌ Missing: {module_file}")

# Ollama endpoint test (optional)
try:
    import requests
    print("\n🌐 Testing LLM connection...")
    response = requests.post(
        os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/tags"
    )
    if response.status_code == 200:
        print("✅ Ollama server reachable.")
    else:
        print("⚠️ Ollama server reachable but returned:", response.status_code)
except Exception as e:
    print("❌ Ollama check failed:", e)
