#!/usr/bin/env python3
"""
Test script for Gemini LM integration
"""

from gemini_lm import SemiconductorGeminiLM

def test_gemini_integration():
    """Test basic Gemini LM functionality"""
    try:
        # Initialize Gemini LM
        gemini_lm = SemiconductorGeminiLM.from_defaults()
        print(f"✓ Gemini LM initialized successfully")
        print(f"  Model: {gemini_lm.model}")
        print(f"  API Base: {gemini_lm.api_base}")
        
        # Test basic response (only if API key is set)
        if gemini_lm.api_key and gemini_lm.api_key != 'your-gemini-api-key':
            print("\n🧪 Testing Gemini response...")
            response = gemini_lm.get_response("What is 2+2?")
            print(f"✓ Response received: {response[:100]}...")
        else:
            print("\n⚠️  No valid API key found. Set GEMINI_API_KEY environment variable to test API calls.")
            
        print("\n✅ Gemini integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini integration: {e}")
        return False

if __name__ == "__main__":
    test_gemini_integration()