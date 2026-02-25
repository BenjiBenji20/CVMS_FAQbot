import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from api.scripts.follow_up_message import FollowUpMessage

def run_tests():
    follow_up = FollowUpMessage()
    
    print("Testing suggestion loading...")
    suggestions = follow_up.suggest_follow_ups("qa-wedding-pricing-hm")
    print(f"Suggestions for wedding-pricing: {len(suggestions)}")
    assert len(suggestions) > 0
    
    print("\nTesting orchestrator...")
    message, actions, suggestions = follow_up.follow_up_message_orchestrator(qa_id="qa-booking-process")
    print(f"Orchestrated message snippet: {message[:30]}...")
    assert "filling out the form" in message
    
    print("\nTesting keyword matching...")
    # Test "how much"
    suggestions = follow_up.get_suggestions_by_keywords("how much is the service?")
    print(f"Suggestions for 'how much': {len(suggestions)}")
    assert len(suggestions) > 0
    assert "portfolio" in suggestions[0]["text"].lower()

    # Test "wedding"
    print("Testing 'wedding' keyword...")
    suggestions = follow_up.get_suggestions_by_keywords("I want to inquire about wedding packages")
    print(f"Suggestions for 'wedding': {len(suggestions)}")
    if not suggestions:
        print("FAILED: No suggestions returned for 'wedding'")
        print(f"Wedding pricing group in data: {'qa-wedding-pricing-hm' in follow_up.follow_up_questions_data}")
        if 'qa-wedding-pricing-hm' in follow_up.follow_up_questions_data:
            print(f"Tags: {follow_up.follow_up_questions_data['qa-wedding-pricing-hm'].get('tags')}")
    
    assert len(suggestions) > 0
    assert "wedding portfolio" in suggestions[0]["text"].lower()

    print("\nALL TESTS PASSED!")

if __name__ == '__main__':
    try:
        run_tests()
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
