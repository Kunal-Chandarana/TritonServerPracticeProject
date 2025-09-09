"""
Demo Content Classifier Model
Simple mock implementation for demonstration purposes
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Demo content classification model."""
    
    def initialize(self, args):
        """Initialize the model."""
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(model_config, "classification_scores")
        output1_config = pb_utils.get_output_config_by_name(model_config, "predicted_class")
        output2_config = pb_utils.get_output_config_by_name(model_config, "confidence_score")
        
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
            
            # Mock classification logic
            batch_size = input_image.as_numpy().shape[0]
            
            # Generate mock results
            classification_scores = np.random.rand(batch_size, 10).astype(self.output0_dtype)
            predicted_class = np.random.randint(0, 10, size=(batch_size, 1)).astype(self.output1_dtype)
            confidence_score = np.random.uniform(0.7, 0.95, size=(batch_size, 1)).astype(self.output2_dtype)
            
            # Create output tensors
            output0_tensor = pb_utils.Tensor("classification_scores", classification_scores)
            output1_tensor = pb_utils.Tensor("predicted_class", predicted_class)
            output2_tensor = pb_utils.Tensor("confidence_score", confidence_score)
            
            # Create inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output0_tensor, output1_tensor, output2_tensor]
            )
            responses.append(inference_response)
            
        return responses
    
    def finalize(self):
        """Clean up resources."""
        pass
