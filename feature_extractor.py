# feature_extractor.py (Revised for YOLOv11)
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect

class YoloFeatureExtractor(torch.nn.Module):
    """
    A wrapper for an Ultralytics YOLO model (v8, v11, etc.) to extract features from the neck.
    
    This class loads a pretrained YOLO model and registers a forward hook 
    on the final C2f module within the model's head, which processes the fused 
    feature maps from the neck right before the final detection layer.
    """
    def __init__(self, model_path='yolov8n.pt'): # You will change this to your yolov11.pt
        super().__init__()
        # Load the pretrained YOLO model
        self.model = YOLO(model_path).model
        self.model.eval() # Set to evaluation mode

        self.features = []
        self._hook_handle = None
        
        # --- ROBUST LAYER IDENTIFICATION ---
        # We want to hook the layer that comes just BEFORE the final prediction layer.
        # In Ultralytics models, the head consists of several layers, ending with 'Detect'.
        # The 'Detect' layer takes a list of feature maps as input. We want to capture that list.
        # We will register the hook on the input to the 'Detect' module.
        
        hook_registered = False
        for i, module in enumerate(self.model.model):
            if isinstance(module, Detect):
                # The input to the Detect module is what we want.
                # So we register a hook on the Detect module itself and capture its *input*.
                self._hook_handle = module.register_forward_pre_hook(self._pre_hook)
                print("✅ YOLO Feature Extractor initialized.")
                print(f"   - Hook registered on the *input* of the 'Detect' layer (module {i}).")
                hook_registered = True
                break
        
        if not hook_registered:
            raise RuntimeError("Could not find the 'Detect' layer in the YOLO model to register a hook.")

    def _pre_hook(self, module, input_tuple):
        """
        A 'pre-forward' hook to capture the *input* to the Detect layer.
        The input is a tuple containing a list of feature maps.
        """
        # The input to the Detect layer is a tuple, where the first element
        # is the list of feature maps from the neck.
        self.features = input_tuple[0]

    def forward(self, x):
        """
        Performs a forward pass to extract features.
        
        Args:
            x (torch.Tensor): The input image tensor.
        
        Returns:
            list[torch.Tensor]: A list of feature maps from the YOLO neck.
        """
        with torch.no_grad():
            self.model(x)
        return self.features

    def __del__(self):
        # Clean up the hook when the object is deleted
        if self._hook_handle:
            self._hook_handle.remove()