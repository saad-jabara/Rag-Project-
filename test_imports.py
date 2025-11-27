#!/usr/bin/env python3
"""Test which imports work"""

try:
    from langchain.chains import RetrievalQA
    print("✓ from langchain.chains import RetrievalQA")
except Exception as e:
    print(f"✗ from langchain.chains import RetrievalQA - {e}")

try:
    from langchain_classic.chains import RetrievalQA
    print("✓ from langchain_classic.chains import RetrievalQA")
except Exception as e:
    print(f"✗ from langchain_classic.chains import RetrievalQA - {e}")

try:
    from langchain_community.chains import RetrievalQA
    print("✓ from langchain_community.chains import RetrievalQA")
except Exception as e:
    print(f"✗ from langchain_community.chains import RetrievalQA - {e}")

try:
    import langchain
    print(f"✓ langchain version: {langchain.__version__}")
except Exception as e:
    print(f"✗ langchain - {e}")

try:
    import langchain_classic
    print(f"✓ langchain_classic installed")
except Exception as e:
    print(f"✗ langchain_classic - {e}")

print("\nAvailable in langchain:")
import langchain
print(dir(langchain))