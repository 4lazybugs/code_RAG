from .base import QAtype, build_qa_type, register_qa_mode
from .mcq import MCQ
from .saq import SAQ
from .hotpot_short import Hotpot_short

# 모든 qa_mode를 import하여 registry에 등록
__all__ = ['QAtype', 'build_qa_type', 'MCQ', 'SAQ', 'Hotpot_short']