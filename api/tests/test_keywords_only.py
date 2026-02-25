import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from api.scripts.follow_up_message import FollowUpMessage

def test_keyword_matching_only():
    follow_up = FollowUpMessage()
    
    print("\n--- Testing keyword matching ---")
    
    # Test "how much"
    print("\nTEST 1: 'how much'")
    suggestions = follow_up.get_suggestions_by_keywords("how much is the service?")
    print(f"Result count: {len(suggestions)}")
    if suggestions:
        print(f"First suggestion: {suggestions[0]['text']}")
    
    # Test "wedding"
    print("\nTEST 2: 'wedding'")
    suggestions = follow_up.get_suggestions_by_keywords("I want to inquire about wedding packages")
    print(f"Result count: {len(suggestions)}")
    if suggestions:
        print(f"First suggestion: {suggestions[0]['text']}")
    else:
        print("FAILED: No suggestions for 'wedding'")

if __name__ == '__main__':
    test_keyword_matching_only()
