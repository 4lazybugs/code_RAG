from .base import QAMode, build_qa_mode, register_qa_mode
from .mcq import MCQ
from .saq import SAQ
from .hotpot import HotpotMode

# 모든 qa_mode를 import하여 registry에 등록
__all__ = ['QAMode', 'build_qa_mode', 'MCQ', 'SAQ', 'HotpotMode']