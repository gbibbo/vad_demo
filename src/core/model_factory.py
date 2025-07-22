# src/core/model_factory.py
from typing import Dict, List, Optional, Type, Any
import sys
import os

# Añadir paths del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from src.core.base_vad import BaseVADModel

# Importar configuración
try:
    from config.settings import MODEL_PATHS
except ImportError:
    # Fallback si no existe config
    MODEL_PATHS = {
        "panns": {
            "checkpoint": "Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth",
            "labels": "sed_demo/assets/audioset_labels.csv"
        },
        "epanns": {
            "checkpoint": "models/epanns/E-PANNs/models/checkpoint_closeto_.44.pt"
        }
    }

# Importar modelos específicos
def _import_panns():
    try:
        from src.models.panns_model import PANNsModel
        return PANNsModel
    except ImportError as e:
        print(f"Warning: Could not import PANNsModel: {e}")
        return None

def _import_epanns():
    try:
        from src.models.epanns_model import EPANNsModel
        return EPANNsModel
    except ImportError as e:
        print(f"Warning: Could not import EPANNsModel: {e}")
        return None

def _import_ast():
    try:
        from src.models.ast_model import ASTModel
        return ASTModel
    except ImportError as e:
        print(f"Warning: Could not import ASTModel: {e}")
        return None

def _import_silero():
    try:
        from src.models.silero_model import SileroModel
        return SileroModel
    except ImportError as e:
        print(f"Warning: Could not import SileroModel: {e}")
        return None

def _import_webrtc():
    try:
        from src.models.webrtc_model import WebRTCModel
        return WebRTCModel
    except ImportError as e:
        print(f"Warning: Could not import WebRTCModel: {e}")
        return None

class ModelInfo:
    """Información sobre un modelo disponible"""
    def __init__(self, name: str, class_type: Type[BaseVADModel], 
                 description: str, color: str, complexity: str = "medium",
                 **default_kwargs):
        self.name = name
        self.class_type = class_type
        self.description = description
        self.color = color
        self.complexity = complexity  # "low", "medium", "high"
        self.is_available = class_type is not None
        self.default_kwargs = default_kwargs

class ModelFactory:
    """Factory para crear y gestionar modelos VAD"""
    
    def __init__(self):
        self._available_models: Dict[str, ModelInfo] = {}
        self._register_models()
    
    def _register_models(self):
        """Registra todos los modelos disponibles"""
        
        # PANNs - siempre disponible
        panns_class = _import_panns()
        self._available_models["panns"] = ModelInfo(
            name="PANNs",
            class_type=panns_class,
            description="Pretrained Audio Neural Networks - CNN based",
            color="cyan",
            complexity="medium",
            model_path=MODEL_PATHS["panns"]["checkpoint"],
            labels_path=MODEL_PATHS["panns"]["labels"]
        )
        
        # E-PANNs - importación dinámica
        epanns_class = _import_epanns()
        self._available_models["epanns"] = ModelInfo(
            name="E-PANNs",
            class_type=epanns_class,
            description="Efficient PANNs for Voice Activity Detection",
            color="orange",
            complexity="medium",
            checkpoint=MODEL_PATHS["epanns"]["checkpoint"]
        )
        
        # AST - importación dinámica
        ast_class = _import_ast()
        self._available_models["ast"] = ModelInfo(
            name="AST",
            class_type=ast_class,
            description="Audio Spectrogram Transformer",
            color="yellow",
            complexity="high"
        )
        
        # Silero VAD - importación dinámica
        silero_class = _import_silero()
        self._available_models["silero"] = ModelInfo(
            name="Silero",
            class_type=silero_class,
            description="Silero VAD - lightweight and fast",
            color="green",
            complexity="low"
        )
        
        # WebRTC VAD - importación dinámica
        webrtc_class = _import_webrtc()
        self._available_models["webrtc"] = ModelInfo(
            name="WebRTC",
            class_type=webrtc_class,
            description="WebRTC Voice Activity Detection",
            color="red",
            complexity="low"
        )
    
    def get_available_models(self) -> Dict[str, ModelInfo]:
        """Obtiene todos los modelos disponibles"""
        return {k: v for k, v in self._available_models.items() if v.is_available}
    
    def get_model_names(self) -> List[str]:
        """Obtiene nombres de modelos disponibles"""
        return list(self.get_available_models().keys())
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Obtiene información de un modelo específico"""
        return self._available_models.get(model_name.lower())
    
    def create_model(self, model_name: str, **kwargs) -> Optional[BaseVADModel]:
        """Crea una instancia de un modelo específico"""
        model_info = self.get_model_info(model_name)
        
        if not model_info or not model_info.is_available:
            print(f"❌ Model '{model_name}' not available")
            return None
        
        try:
            # Combinar argumentos por defecto con los proporcionados
            combined_kwargs = {**model_info.default_kwargs, **kwargs}
            
            # Crear instancia del modelo
            model = model_info.class_type(**combined_kwargs)
            print(f"✓ Created {model_info.name} model")
            return model
        except Exception as e:
            print(f"❌ Error creating {model_name} model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_model_pair(self, model1_name: str, model2_name: str, 
                         **kwargs) -> tuple[Optional[BaseVADModel], Optional[BaseVADModel]]:
        """Crea un par de modelos para comparación"""
        model1 = self.create_model(model1_name, **kwargs)
        model2 = self.create_model(model2_name, **kwargs)
        
        return model1, model2
    
    def validate_model_combination(self, model_names: List[str]) -> bool:
        """Valida si una combinación de modelos es viable"""
        if len(model_names) > 2:
            print("❌ Maximum 2 models allowed")
            return False
        
        if len(model_names) < 1:
            print("❌ At least 1 model required")
            return False
        
        # Verificar disponibilidad
        for name in model_names:
            if name not in self.get_available_models():
                print(f"❌ Model '{name}' not available")
                return False
        
        # Verificar complejidad combinada
        total_complexity = sum(
            self._complexity_score(self.get_model_info(name).complexity)
            for name in model_names
        )
        
        if total_complexity > 5:  # Threshold ajustable
            print(f"⚠️ High complexity combination ({total_complexity}). May affect performance.")
        
        return True
    
    def _complexity_score(self, complexity: str) -> int:
        """Convierte complejidad a score numérico"""
        scores = {"low": 1, "medium": 2, "high": 3}
        return scores.get(complexity, 2)
    
    def get_recommended_pairs(self) -> List[tuple[str, str]]:
        """Obtiene pares recomendados de modelos"""
        available = self.get_available_models()
        
        recommendations = []
        
        # Pares balanceados por complejidad
        if "panns" in available and "silero" in available:
            recommendations.append(("panns", "silero"))
        
        if "epanns" in available and "webrtc" in available:
            recommendations.append(("epanns", "webrtc"))
        
        if "panns" in available and "epanns" in available:
            recommendations.append(("panns", "epanns"))
        
        # Pares de alta calidad (si el hardware lo permite)
        if "ast" in available and "panns" in available:
            recommendations.append(("ast", "panns"))
        
        return recommendations
    
    def print_available_models(self):
        """Imprime información de modelos disponibles"""
        print("\n" + "="*60)
        print("AVAILABLE VAD MODELS")
        print("="*60)
        
        available = self.get_available_models()
        
        for key, info in available.items():
            status = "✓" if info.is_available else "✗"
            print(f"{status} {info.name:12} | {info.complexity:6} | {info.description}")
        
        print(f"\nTotal available: {len(available)} models")
        
        recommended = self.get_recommended_pairs()
        if recommended:
            print("Recommended pairs:", recommended)
        
        print("="*60)

# Instancia global del factory
model_factory = ModelFactory()