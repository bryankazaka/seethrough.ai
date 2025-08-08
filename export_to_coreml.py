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

# Wrapper for Core ML export with preprocessing
class CoreMLExportWrapper(nn.Module):
    def __init__(self, model: SmolVLM2Classifier, image_size: int = 128, max_frames: int = 4):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.max_frames = max_frames
        
        # Set model to eval and disable dropout for Core ML
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0  # Disable dropout for Core ML
        
        # Image normalization parameters (ImageNet standards)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        # Ensure float32 and normalize from [0, 255] to [0, 1]
        images = images.float() / 255.0
        
        # Apply ImageNet normalization
        if images.dim() == 4:  # Single image
            images = (images - self.mean) / self.std
        else:  # Video frames
            # Reshape for normalization
            b, f, c, h, w = images.shape
            images = images.view(b * f, c, h, w)
            images = (images - self.mean) / self.std
            images = images.view(b, f, c, h, w)
        
        return images
    
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # pixel_values should already be preprocessed by the processor
        # No additional preprocessing needed since we're using processor output
        
        # Get predictions
        probs = self.model(pixel_values, input_ids, attention_mask)
        
        return probs

def prepare_fixed_inputs(processor, prompt: str, image_size: int = 128, max_frames: int = 4):
    """Prepare fixed text inputs for the model"""
    # Create dummy images for shape inference - using correct number of frames
    dummy_images = [Image.new('RGB', (image_size, image_size)) for _ in range(max_frames)]
    
    # Process to get input shapes
    inputs = processor(
        text=prompt,
        images=dummy_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,  # Set a reasonable max length
        do_resize=True,
        size={'height': image_size, 'width': image_size}
    )
    
    return inputs['input_ids'], inputs['attention_mask']

def export_to_coreml(
    model_path: str,
    output_path: str,
    model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    image_size: int = 128,  # Match training config
    max_frames: int = 4,     # Match training config
    quantize: bool = True
):
    """Export trained model to Core ML format"""
    
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Load trained model with exact same architecture
    pytorch_model = SmolVLM2Classifier(model_name, num_classes=2)
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()
    
    # Create wrapper
    wrapped_model = CoreMLExportWrapper(pytorch_model, image_size, max_frames)
    wrapped_model.eval()
    
    # Prepare fixed prompt inputs - use a simpler prompt to reduce complexity
    if max_frames > 1:
        prompt = "<image>" * max_frames + " Is this real or AI-generated?"
    else:
        prompt = "<image> Is this real or AI-generated?"
    
    fixed_input_ids, fixed_attention_mask = prepare_fixed_inputs(processor, prompt, image_size, max_frames)
    
    print("Creating traced model...")
    # Create example inputs for tracing - ensure batch size consistency
    example_video = torch.randint(0, 255, (1, max_frames, 3, image_size, image_size), dtype=torch.float32)
    
    # Test the model first to ensure it works
    with torch.no_grad():
        print("Testing model forward pass...")
        try:
            test_output = wrapped_model(example_video, fixed_input_ids, fixed_attention_mask)
            print(f"Test output shape: {test_output.shape}")
        except Exception as e:
            print(f"Model forward pass failed: {e}")
            # Try with different input shapes
            print("Retrying with adjusted inputs...")
            # Ensure the pixel_values match what the processor expects
            dummy_images = [Image.new('RGB', (image_size, image_size)) for _ in range(max_frames)]
            processor_inputs = processor(
                text=prompt,
                images=dummy_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                do_resize=True,
                size={'height': image_size, 'width': image_size}
            )
            
            # Use the exact pixel_values from processor
            example_video = processor_inputs['pixel_values']
            fixed_input_ids = processor_inputs['input_ids'] 
            fixed_attention_mask = processor_inputs['attention_mask']
            
            test_output = wrapped_model(example_video, fixed_input_ids, fixed_attention_mask)
            print(f"Test output shape after adjustment: {test_output.shape}")
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model,
            (example_video, fixed_input_ids, fixed_attention_mask),
            check_trace=False  # Disable trace checking due to model complexity
        )
    
    print("Converting to Core ML...")
    # Define input types for Core ML based on actual tensor shapes
    inputs = [
        ct.TensorType(
            name="pixel_values",
            shape=example_video.shape,
            dtype=np.float32
        ),
        ct.TensorType(
            name="input_ids",
            shape=fixed_input_ids.shape,
            dtype=np.int32
        ),
        ct.TensorType(
            name="attention_mask", 
            shape=fixed_attention_mask.shape,
            dtype=np.int32
        )
    ]
    
    # Convert to Core ML
    try:
        coreml_model = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=[
                ct.TensorType(name="probabilities", dtype=np.float32)
            ],
            compute_units=ct.ComputeUnit.ALL,  # Use Neural Engine if available
            minimum_deployment_target=ct.target.iOS16
        )
    except Exception as e:
        print(f"Initial conversion failed: {e}")
        print("Trying with CPU-only compute units...")
        coreml_model = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=[
                ct.TensorType(name="probabilities", dtype=np.float32)
            ],
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS16
        )
    
    # Add metadata
    coreml_model.author = "SmolVLM2 Video Classifier"
    coreml_model.short_description = "Classifies videos/images as AI-generated or real"
    coreml_model.input_description["pixel_values"] = f"Preprocessed pixel values from processor"
    coreml_model.output_description["probabilities"] = "Probability distribution [real, ai_generated]"
    
    # Optionally quantize the model
    if quantize:
        print("Quantizing model...")
        from coremltools.models.neural_network import quantization_utils
        coreml_model = quantization_utils.quantize_weights(
            coreml_model, 
            nbits=8,
            quantization_mode="kmeans"
        )
    
    # Save the model
    coreml_model.save(output_path)
    print(f"Core ML model saved to {output_path}")
    
    # Save the fixed inputs for reference
    fixed_inputs = {
        'input_ids': fixed_input_ids.numpy(),
        'attention_mask': fixed_attention_mask.numpy(),
        'prompt': prompt,
        'image_size': image_size,
        'max_frames': max_frames
    }
    
    with open(output_path.replace('.mlmodel', '_inputs.pkl'), 'wb') as f:
        pickle.dump(fixed_inputs, f)
    
    return coreml_model, fixed_inputs

class CoreMLInference:
    """Inference wrapper for Core ML model"""
    
    def __init__(self, model_path: str, inputs_path: str):
        self.model = ct.models.MLModel(model_path)
        
        # Load fixed inputs
        with open(inputs_path, 'rb') as f:
            self.fixed_inputs = pickle.load(f)
        
        self.image_size = self.fixed_inputs['image_size']
        self.max_frames = self.fixed_inputs['max_frames']
        
        # Load processor for preprocessing
        from transformers import AutoProcessor
        model_name = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess single image using the processor"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        
        # Create video by repeating the image
        images = [img] * self.max_frames
        
        # Use processor to get pixel_values
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_resize=True,
            size={'height': self.image_size, 'width': self.image_size}
        )
        
        return inputs['pixel_values'].numpy()
    
    def preprocess_video(self, video_path: str) -> np.ndarray:
        """Preprocess video file using the processor"""
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
        
        # Use processor to get pixel_values
        inputs = self.processor(
            images=pil_frames,
            return_tensors="pt",
            do_resize=True,
            size={'height': self.image_size, 'width': self.image_size}
        )
        
        return inputs['pixel_values'].numpy()
    
    def predict(self, file_path: str) -> dict:
        """Run prediction on image or video file"""
        # Preprocess input
        if file_path.endswith(('.mp4', '.avi', '.mov', '.webm')):
            pixel_values = self.preprocess_video(file_path)
        else:
            pixel_values = self.preprocess_image(file_path)
        
        # Prepare inputs
        inputs = {
            'pixel_values': pixel_values,
            'input_ids': self.fixed_inputs['input_ids'],
            'attention_mask': self.fixed_inputs['attention_mask']
        }
        
        # Run inference
        output = self.model.predict(inputs)
        probs = output['probabilities'][0]  # Remove batch dimension
        
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
    
    # Export the trained model to Core ML
    model_path = f"{config['save_dir']}/best_model_fold_0.pth"
    output_path = "./smolvlm2_video_classifier.mlmodel"
    
    # Export with quantization for smaller model size
    coreml_model, fixed_inputs = export_to_coreml(
        model_path=model_path,
        output_path=output_path,
        model_name=config['model_name'],
        image_size=config['image_size'],
        max_frames=config['max_frames'],
        quantize=True
    )
    
    # Test the Core ML model
    print("\nTesting Core ML model...")
    inference = CoreMLInference(
        output_path,
        output_path.replace('.mlmodel', '_inputs.pkl')
    )
    
    # Test on an image (if you have one)
    # result = inference.predict("test_image.jpg")
    # print(f"Test result: {result}")
    
    print("\nExport complete!")
    print(f"Core ML model: {output_path}")
    print(f"Fixed inputs: {output_path.replace('.mlmodel', '_inputs.pkl')}")
    print("\nTo use in iOS/macOS app:")
    print("1. Add the .mlmodel file to your Xcode project")
    print("2. Load the fixed inputs from the .pkl file or hardcode them")
    print(f"3. Preprocess video frames to (1, {config['max_frames']}, 3, {config['image_size']}, {config['image_size']}) format")
    print("4. Run inference and get probability distribution")