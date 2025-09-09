"""
Demo Safety Detector Model
Simple mock implementation for demonstration purposes
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Demo safety detection model."""
    
    def initialize(self, args):
        """Initialize the model."""
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(model_config, "safety_scores")
        output1_config = pb_utils.get_output_config_by_name(model_config, "is_safe")
        output2_config = pb_utils.get_output_config_by_name(model_config, "risk_level")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config['data_type'])
        self.output2_dtype = pb_utils.triton_string_to_numpy(output2_config['data_type'])
        
    def execute(self, requests):
        """Execute inference on the batch of requests."""
        responses = []
        
        for request in requests:
            # Get input tensor
            input_image = pb_utils.get_input_tensor_by_name(request, "input_image")
            
            # Mock safety detection logic
            batch_size = input_image.as_numpy().shape[0]
            
            # Generate mock results (mostly safe for demo)
            safety_scores = np.random.uniform(0.1, 0.3, size=(batch_size, 5)).astype(self.output0_dtype)
            is_safe = np.ones((batch_size, 1), dtype=self.output1_dtype)  # Mostly safe
            
            # Create risk level strings
            risk_levels = np.array([["LOW"] for _ in range(batch_size)], dtype=object)
            
            # Create output tensors
            output0_tensor = pb_utils.Tensor("safety_scores", safety_scores)
            output1_tensor = pb_utils.Tensor("is_safe", is_safe)
            output2_tensor = pb_utils.Tensor("risk_level", risk_levels)
            
            # Create inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output0_tensor, output1_tensor, output2_tensor]
            )
            responses.append(inference_response)
            
        return responses
    
    def finalize(self):
        """Clean up resources."""
        pass
