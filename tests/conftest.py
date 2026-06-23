"""Pytest configuration for all tests."""
import sys
from unittest.mock import MagicMock

# Pre-register torch mock before any imports
torch_mock = MagicMock()
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()
