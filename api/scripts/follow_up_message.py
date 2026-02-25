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
        Scan message for tags and return corresponding suggestions.
        """
        message_lower = message.lower()
        for qa_id, data in self.follow_up_questions_data.items():
            tags = data.get("tags", [])
            for tag in tags:
                if tag.lower() in message_lower:
                    return data.get("suggestions", [])[:3]
        return []

    
    def follow_up_message_orchestrator(self, qa_id: str = None, action_id: str = None) -> tuple[str, list[dict], list[dict]]:
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
                message = f"You can view {action.get('title', 'this page')} here:"
                # Optional: can add suggestions for action clicks if needed, but requirements say bypass LLM
                # For action_id, we usually just return the button.

        return message, actions, suggestions

follow_up_message = FollowUpMessage()
    