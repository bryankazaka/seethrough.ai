import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from PIL import Image
from transformers import AutoModel, AutoProcessor
from typing import List, Optional, Tuple
import warnings
import cv2
import pickle

# IMPORTANT: This must match EXACTLY the architecture from your training script
class SmolVLM2Classifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        
        # Load base model - matching training script exactly
        self.base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        
        self.base_model = self.base_model.float()
        
        # Get hidden size - matching training script logic
        if hasattr(self.base_model.config, 'text_config') and hasattr(self.base_model.config.text_config, 'hidden_size'):
            self.hidden_size = self.base_model.config.text_config.hidden_size
        elif hasattr(self.base_model.config, 'hidden_size'):
            self.hidden_size = self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, 'd_model'):
            self.hidden_size = self.base_model.config.d_model
        else:
            self.hidden_size = 768
        
        # EXACT same classifier architecture as training
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._init_classifier()
        
    def _init_classifier(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Match the forward logic from training
        inputs = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Get base model outputs
        outputs = self.base_model(**inputs)
        
        # Extract hidden states - matching training logic
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            if isinstance(outputs, torch.Tensor):
                hidden_states = outputs
            else:
                hidden_states = outputs[0] if hasattr(outputs, '__getitem__') else outputs.logits
        
        # Temporal pooling - mean across sequence dimension
        pooled = hidden_states.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Return probability distribution
        probs = F.softmax(logits, dim=-1)
        return probs

def create_simplified_model(model_path: str, model_name: str, output_path: str):
    """Create a simplified PyTorch model that can be loaded easily"""
    
    print("Loading trained model...")
    # Load trained model with exact same architecture
    pytorch_model = SmolVLM2Classifier(model_name, num_classes=2)
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()
    
    # Set dropout to eval mode
    for module in pytorch_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    
    # Save the simplified model
    torch.save({
        'model_state_dict': pytorch_model.state_dict(),
        'model_config': {
            'model_name': model_name,
            'num_classes': 2,
            'hidden_size': pytorch_model.hidden_size
        }
    }, output_path)
    
    return pytorch_model

def export_classifier_to_coreml(model_path: str, model_name: str, output_path: str):
    """Export just the classifier head to CoreML"""
    
    print("Loading trained model...")
    pytorch_model = SmolVLM2Classifier(model_name, num_classes=2)
    checkpoint = torch.load(model_path, map_location='cpu')
    pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()
    
    # Extract just the classifier
    classifier = pytorch_model.classifier
    classifier.eval()
    
    # Set dropout to 0
    for module in classifier.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    
    # Create example input (hidden size from the base model)
    hidden_size = pytorch_model.hidden_size
    example_input = torch.randn(1, hidden_size)
    
    print("Tracing classifier...")
    # Trace just the classifier (this should work since it's simpler)
    traced_classifier = torch.jit.trace(classifier, example_input)
    
    print("Converting classifier to CoreML...")
    # Convert to CoreML
    coreml_classifier = ct.convert(
        traced_classifier,
        inputs=[ct.TensorType(name="features", shape=(1, hidden_size), dtype=np.float32)],
        outputs=[ct.TensorType(name="probabilities", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Add metadata
    coreml_classifier.author = "SmolVLM2 Classifier Head"
    coreml_classifier.short_description = "Classifier head for SmolVLM2 model"
    coreml_classifier.input_description["features"] = f"Feature vector from base model ({hidden_size} dimensions)"
    coreml_classifier.output_description["probabilities"] = "Probability distribution [real, ai_generated]"
    
    # Save
    coreml_classifier.save(output_path)
    print(f"CoreML classifier saved to: {output_path}")
    
    return coreml_classifier

class HybridInference:
    """Hybrid inference using PyTorch for features + CoreML for classification"""
    
    def __init__(self, pytorch_model_path: str, coreml_classifier_path: str, model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"):
        # Load PyTorch model for feature extraction
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.pytorch_model = SmolVLM2Classifier(model_name, num_classes=2)
            self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.pytorch_model = SmolVLM2Classifier(model_name, num_classes=2)
            self.pytorch_model.load_state_dict(checkpoint)
        
        self.pytorch_model.eval()
        
        # Load CoreML classifier
        self.coreml_classifier = ct.models.MLModel(coreml_classifier_path)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        self.image_size = 128
        self.max_frames = 4
    
    def extract_features(self, inputs):
        """Extract features using PyTorch base model"""
        with torch.no_grad():
            # Get base model outputs
            outputs = self.pytorch_model.base_model(**inputs)
            
            # Extract hidden states - matching training logic
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                if isinstance(outputs, torch.Tensor):
                    hidden_states = outputs
                else:
                    hidden_states = outputs[0] if hasattr(outputs, '__getitem__') else outputs.logits
            
            # Temporal pooling - mean across sequence dimension
            pooled = hidden_states.mean(dim=1)
            
            return pooled.numpy()
    
    def preprocess_image(self, image_path: str):
        """Preprocess single image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        
        # Create video by repeating the image
        images = [img] * self.max_frames
        prompt = "<image>" * self.max_frames + " Is this real or AI-generated?"
        
        # Use processor
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            do_resize=True,
            size={'height': self.image_size, 'width': self.image_size}
        )
        
        return inputs
    
    def preprocess_video(self, video_path: str):
        """Preprocess video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        if total_frames <= self.max_frames:
            indices = list(range(total_frames))
            while len(indices) < self.max_frames:
                indices.append(total_frames - 1)
        else:
            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        
        # Read frames
        pil_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                pil_frame = Image.fromarray(frame)
                pil_frames.append(pil_frame)
        
        cap.release()
        
        prompt = "<image>" * self.max_frames + " Is this real or AI-generated?"
        
        # Use processor
        inputs = self.processor(
            text=prompt,
            images=pil_frames,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            do_resize=True,
            size={'height': self.image_size, 'width': self.image_size}
        )
        
        return inputs
    
    def predict(self, file_path: str) -> dict:
        """Run prediction using hybrid approach"""
        # Preprocess input
        if file_path.endswith(('.mp4', '.avi', '.mov', '.webm')):
            inputs = self.preprocess_video(file_path)
        else:
            inputs = self.preprocess_image(file_path)
        
        # Extract features using PyTorch
        features = self.extract_features(inputs)
        
        # Classify using CoreML
        coreml_input = {'features': features}
        coreml_output = self.coreml_classifier.predict(coreml_input)
        probs = coreml_output['probabilities'][0]  # Remove batch dimension
        
        return {
            'real_probability': float(probs[0]),
            'ai_probability': float(probs[1]),
            'prediction': 'AI-generated' if probs[1] > 0.5 else 'Real',
            'confidence': float(max(probs))
        }

if __name__ == "__main__":
    # Configuration matching your training setup
    config = {
        'save_dir': './results',  # Your training save directory
        'model_name': "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        'image_size': 128,  # Match training config
        'max_frames': 4,    # Match training config
    }
    
    # Paths
    model_path = f"{config['save_dir']}/best_model_fold_0.pth"
    simplified_path = "./smolvlm2_video_classifier_simplified.pth"
    coreml_classifier_path = "./smolvlm2_classifier.mlmodel"
    
    print("Creating simplified PyTorch model...")
    simplified_model = create_simplified_model(
        model_path=model_path,
        model_name=config['model_name'],
        output_path=simplified_path
    )
    print(f"Simplified model saved to: {simplified_path}")
    
    print("\nExporting classifier to CoreML...")
    coreml_classifier = export_classifier_to_coreml(
        model_path=model_path,
        model_name=config['model_name'],
        output_path=coreml_classifier_path
    )
    
    print("\nTesting hybrid inference...")
    inference = HybridInference(simplified_path, coreml_classifier_path, config['model_name'])
    
    print("Export complete!")
    print(f"PyTorch model: {simplified_path}")
    print(f"CoreML classifier: {coreml_classifier_path}")
    print("\nTo use this hybrid model:")
    print("1. Use PyTorch for feature extraction (works on any platform)")
    print("2. Use CoreML classifier for final prediction (optimized for Apple devices)")
    print("3. Call HybridInference.predict() with image/video file path")
    
    # Save deployment code
    deployment_code = '''
# Hybrid deployment code
from export_to_onnx import HybridInference

# Load hybrid model (PyTorch + CoreML)
inference = HybridInference(
    "smolvlm2_video_classifier_simplified.pth",
    "smolvlm2_classifier.mlmodel"
)

# Run prediction
result = inference.predict("path/to/your/file.jpg")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Real probability: {result['real_probability']:.3f}")
print(f"AI probability: {result['ai_probability']:.3f}")
'''
    
    with open("hybrid_deployment_example.py", "w") as f:
        f.write(deployment_code)
    
    print(f"Deployment example saved to: hybrid_deployment_example.py")
