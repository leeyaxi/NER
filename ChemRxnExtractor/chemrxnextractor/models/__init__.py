from .model import BertForTagging
from .model import BertCRFForTagging
from .model import BertForRoleLabeling
from .model import BertCRFForRoleLabeling
from .model import BertGlobalPointerForTagging
from .model import BertGlobalPointerForRoleLabeling
__all__ = [
    'BertForTagging',
    'BertCRFForTagging',
    'BertGlobalPointerForTagging',
    'BertForRoleLabeling',
    'BertCRFForRoleLabeling',
    'BertGlobalPointerForRoleLabeling'
]
