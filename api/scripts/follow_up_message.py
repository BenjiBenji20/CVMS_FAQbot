import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FollowUpMessage:
    def __init__(self):
        try:
            THIS_FILE_DIR = Path(__file__).parent
            DOCS_DIR = THIS_FILE_DIR.parent / "documents"
            
            follow_up_questions_file = DOCS_DIR / "suggest-follow-up-questions.json"
            if not follow_up_questions_file.exists():
                raise FileExistsError(f"File not found: {follow_up_questions_file.name}")
        
            self.follow_up_questions_data = {}
            
            # load follow_up_questions_file JSON file
            with open(follow_up_questions_file, 'r', encoding='utf-8') as f:
                self.follow_up_questions_data: dict = json.load(f)
                
            qa_file = DOCS_DIR / "cvms-qa-structured-data.jsonl"
            if not qa_file.exists():
                raise FileExistsError(f"File not found: {qa_file.name}")
            
            self.qa_file_data = {}
            
            # load cvms-qa-structured-data.jsonl JSONL file
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line: # skip empty lines
                        continue
                    
                    qa_entry = json.loads(line)
                    if "id" in qa_entry:
                        self.qa_file_data[qa_entry["id"]] = qa_entry
                    
            action_file = DOCS_DIR / "cvms-structured-data.json"
            if not action_file.exists():
                raise FileExistsError(f"File not found: {action_file.name}")
            
            self.action_data = {}
            
            # load cvms-structured-data.json JSON file
            with open(action_file, 'r', encoding='utf-8') as f:
                action_list = json.load(f)
                self.action_data = {action['id']: action for action in action_list}
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON error: {e}")
            
        except KeyError as e:
            logger.warning(f"Missing key: {e}")
        
        except Exception:
            raise    
    

    def suggest_follow_ups(self, qa_id: str) -> list[dict[str, str]]:
        """
        Args:
            qa_id: use to map a key in suggest-follow-up-questions.json file
        """
        data = self.follow_up_questions_data.get(qa_id)
        if data and "suggestions" in data:
            return data["suggestions"][:3]
        return []
    
    
    def from_qa_follow_ups(self, qa_id: str) -> dict:
        """
        Args:
            qa_id: use to map an answer key in cvms-qa-structured-data.jsonl file
        """
        return self.qa_file_data.get(qa_id)


    def get_suggestions_by_keywords(self, message: str) -> list[dict]:
        """
        Score each entry by word overlap and return mixed suggestions
        from multiple matching entries (max 3 total).
        
        Strategy:
        - Find all entries with overlap > 0
        - Sort by overlap score (descending)
        - Take 1 suggestion from each top entry until we have 3 total
        - Deduplicate by action_id/qa_id to avoid showing same button twice
        """
        message_lower = message.lower()
        message_words = set(message_lower.split())
        
        # Collect all matches with their scores
        matches = []
        
        for qa_id, data in self.follow_up_questions_data.items():
            tags = data.get("tags", [])
            
            # Find highest overlap for this entry
            max_overlap = 0
            for tag in tags:
                tag_words = set(tag.lower().split())
                overlap = len(message_words & tag_words)
                if overlap > max_overlap:
                    max_overlap = overlap
            
            # Only include if there's actual overlap
            if max_overlap > 0:
                matches.append({
                    'qa_id': qa_id,
                    'data': data,
                    'overlap': max_overlap
                })
                logger.info(f"Match: {qa_id} (overlap: {max_overlap})")
        
        if not matches:
            return []
        
        # Sort by overlap score (highest first)
        matches.sort(key=lambda x: x['overlap'], reverse=True)
        
        # Mix suggestions from top matches
        mixed_suggestions = []
        seen_ids = set()  # Track to avoid duplicates
        suggestion_index = 0
        max_suggestions = 3
        
        # Round-robin through matches
        while len(mixed_suggestions) < max_suggestions and suggestion_index < 10:  # Safety limit
            added_any = False
            
            for match in matches:
                if len(mixed_suggestions) >= max_suggestions:
                    break
                
                suggestions = match['data'].get('suggestions', [])
                
                # Get next suggestion from this entry
                if suggestion_index < len(suggestions):
                    suggestion = suggestions[suggestion_index]
                    
                    # Create unique ID for deduplication
                    unique_id = suggestion.get('action_id') or suggestion.get('qa_id')
                    
                    if unique_id and unique_id not in seen_ids:
                        mixed_suggestions.append(suggestion)
                        seen_ids.add(unique_id)
                        added_any = True
                        logger.info(f"Added suggestion: {suggestion.get('text')} from {match['qa_id']}")
            
            if not added_any:
                break  # No more suggestions to add
            
            suggestion_index += 1
        
        return mixed_suggestions
    
    
    def follow_up_message_orchestrator(
        self, qa_id: str = None, 
        action_id: str = None
    ) -> tuple[str, list[dict], list[dict]]:
        """
        Orchestrate the response for follow-up message clicks.
        Bypasses LLM and returns deterministic response.
        """
        message = ""
        actions = []
        suggestions = []

        if qa_id:
            qa_entry = self.from_qa_follow_ups(qa_id)
            if qa_entry:
                message = qa_entry.get("answer", "")
                
                # Attach action button if it exists in metadata
                target_action_id = qa_entry.get("action_id")
                if target_action_id and target_action_id in self.action_data:
                    actions.append(self.action_data[target_action_id])
                
                # Fetch its own follow-up suggestions
                suggestions = self.suggest_follow_ups(qa_id)
        
        elif action_id:
            if action_id in self.action_data:
                action = self.action_data[action_id]
                actions.append(action)
                message = f"You can view {action.get('title', 'this page')} here 👇"

        return message, actions, suggestions

follow_up_message = FollowUpMessage()
