
from .task import TextRecognitionTask
from .models.vit_models import ViTTRModel, ViT_TR_base
from .models.okayocr_model import OKayOCR
from .utils.scoring import AccEDScorer
from .models.deit import deit_base_distilled_patch16_224, deit_base_distilled_patch16_384