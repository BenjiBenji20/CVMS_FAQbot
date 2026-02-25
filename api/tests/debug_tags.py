import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from api.scripts.follow_up_message import FollowUpMessage

def debug_tags():
    follow_up = FollowUpMessage()
    
    print(f"Total suggestion groups: {len(follow_up.follow_up_questions_data)}")
    
    wedding_id = "qa-wedding-pricing-hm"
    if wedding_id in follow_up.follow_up_questions_data:
        data = follow_up.follow_up_questions_data[wedding_id]
        tags = data.get("tags", [])
        print(f"Tags for {wedding_id}: {tags}")
        
        message = "I want to inquire about wedding packages"
        message_lower = message.lower()
        print(f"Message lower: '{message_lower}'")
        
        results = []
        any_match = False
        for tag in tags:
            match = tag.lower() in message_lower
            results.append(f"'{tag}': {'YES' if match else 'NO'}")
            if match: any_match = True
        
        print(f"Results: {', '.join(results)}")
        print(f"ANY MATCH FOUND: {any_match}")
    else:
        print(f"ERROR: {wedding_id} not found in data!")
        print(f"Available keys: {list(follow_up.follow_up_questions_data.keys())[:5]}")

if __name__ == '__main__':
    debug_tags()
