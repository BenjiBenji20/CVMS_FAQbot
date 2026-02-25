import sys
import unittest
from unittest.mock import MagicMock, patch
import asyncio
from pathlib import Path
import json

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from api.scripts.follow_up_message import FollowUpMessage
from api.services.chatbot_service import ChatbotService
from api.schemas.chatbot_schemas import ChatRequest

class TestFollowUpSuggestions(unittest.TestCase):
    def setUp(self):
        self.follow_up = FollowUpMessage()
        self.service = ChatbotService()
        # Mock redis
        self.service.redis_client = MagicMock()
        self.service.redis_client.get.return_value = None

    def test_suggestion_loading(self):
        # Verify suggestions are loaded
        suggestions = self.follow_up.suggest_follow_ups("qa-wedding-pricing-hm")
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        self.assertEqual(suggestions[0]["text"], "See wedding portfolio")

    def test_orchestrator_qa_id(self):
        # Test direct qa_id click
        message, actions, suggestions = self.follow_up.follow_up_message_orchestrator(qa_id="qa-booking-process")
        self.assertIn("You can book by filling out the form", message)
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0]["id"], "booking-page")
        self.assertGreater(len(suggestions), 0)

    @patch('api.services.chatbot_service.chatbot')
    def test_service_deterministic_bypass(self, mock_chatbot):
        # Ensure chatbot (LLM) is NOT called when qa_id is provided
        loop = asyncio.get_event_loop()
        message, actions, suggestions = loop.run_until_complete(
            self.service.get_chat_response("ignore me", qa_id="qa-booking-process")
        )
        
        mock_chatbot.assert_not_called()
        self.assertIn("filling out the form", message)
        self.assertEqual(actions[0]["id"], "booking-page")

    @patch('api.services.chatbot_service.chatbot')
    def test_service_normal_flow_with_suggestions(self, mock_chatbot):
        # Ensure suggestions are attached when LLM returns a qa_id
        mock_chatbot.return_value = ("Test answer", [{"id": "test-action"}], "qa-wedding-pricing-hm")
        
        loop = asyncio.get_event_loop()
        message, actions, suggestions = loop.run_until_complete(
            self.service.get_chat_response("wedding price")
        )
        
        self.assertEqual(message, "Test answer")
        self.assertEqual(len(suggestions), 3) # qa-wedding-pricing-hm has 3 suggestions

if __name__ == '__main__':
    unittest.main()
