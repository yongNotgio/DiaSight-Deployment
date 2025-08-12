import requests
import json

# Test the API endpoints
base_url = "http://localhost:8000"

print("Testing DiaSight API Endpoints")
print("=" * 40)

# Test 1: Root endpoint
try:
    response = requests.get(f"{base_url}/")
    print("✅ Root endpoint:", response.json())
except Exception as e:
    print("❌ Root endpoint error:", e)

# Test 2: Available models
try:
    response = requests.get(f"{base_url}/models")
    print("✅ Available models:", response.json())
except Exception as e:
    print("❌ Models endpoint error:", e)

# Test 3: Model info
try:
    response = requests.get(f"{base_url}/model-info")
    print("✅ Model info:", response.json())
except Exception as e:
    print("❌ Model info error:", e)

# Test 4: Health check
try:
    response = requests.get(f"{base_url}/health")
    print("✅ Health check:", response.json())
except Exception as e:
    print("❌ Health check error:", e)

# Test 5: Single prediction
try:
    test_data = {
        "age": 65,
        "hb1ac": 8.5,
        "duration": 10,
        "egfr": 60,
        "ldl": 120,
        "hdl": 40,
        "chol": 200,
        "sbp": 140,
        "dbp": 90,
        "hbp": 1,
        "sex": 1,
        "uric": 5.5,
        "bun": 20,
        "urea": 40,
        "trig": 150,
        "ucr": 100,
        "alt": 30,
        "ast": 25
    }
    
    response = requests.post(f"{base_url}/predict", json=test_data)
    result = response.json()
    print("✅ Prediction result:")
    print(f"   Predicted class: {result['prediction']}")
    print(f"   Risk score: {result['risk_score']:.2f}%")
    print(f"   Confidence: {result['confidence']:.3f}")
except Exception as e:
    print("❌ Prediction error:", e)

print("\n" + "=" * 40)
print("API testing complete!")
