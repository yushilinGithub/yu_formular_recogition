from .task import SROIETextRecognitionTask
from .models.vit_models import ViTTRModel, ViT_TR_base
from .utils.scoring import AccEDScorer
from .models.deit import deit_base_distilled_patch16_224, deit_base_distilled_patch16_384
from .okayTask import TrOCRModel
from .utils.bpe import GPT2BPEEnhancedSpace