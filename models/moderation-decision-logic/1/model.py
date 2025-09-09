"""
Production Content Moderation Decision Logic
This model combines outputs from multiple AI models to make final moderation decisions.
"""

import json
import logging
import numpy as np
import triton_python_backend_utils as pb_utils
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    """
    Production-grade content moderation decision logic.
    
    This model implements enterprise business rules for content moderation
    by combining outputs from:
    - Content classification model
    - Safety detection model  
    - OCR text extraction model
    """
    
    def initialize(self, args):
        """Initialize the model with production configurations."""
        self.model_config = json.loads(args['model_config'])
        
        # Production thresholds (configurable via environment)
        self.safety_threshold = 0.85
        self.content_confidence_threshold = 0.7
        self.text_confidence_threshold = 0.6
        
        # Content category mappings (ImageNet subset)
        self.sensitive_categories = {
            # Weapons and violence
            413, 414, 415,  # assault_rifle, shotgun, revolver
            # Adult content indicators
            939, 940, 941,  # bikini, brassiere, etc.
            # Other sensitive categories
            508, 509, 510   # computer_keyboard, computer_mouse, etc. (placeholder)
        }
        
        # Text filtering patterns (production would use more sophisticated NLP)
        self.blocked_keywords = [
            'violence', 'weapon', 'hate', 'discrimination',
            'illegal', 'drugs', 'explicit', 'inappropriate'
        ]
        
        logger.info("Content moderation decision logic initialized successfully")
    
    def execute(self, requests):
        """Execute moderation decision logic for batch of requests."""
        responses = []
        
        for request in requests:
            try:
                # Extract inputs
                inputs = self._extract_inputs(request)
                
                # Make moderation decision
                decision_result = self._make_moderation_decision(inputs)
                
                # Create response tensors
                response_tensors = self._create_response_tensors(decision_result)
                
                # Create inference response
                response = pb_utils.InferenceResponse(output_tensors=response_tensors)
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Processing error: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    def _extract_inputs(self, request) -> Dict[str, Any]:
        """Extract and validate input tensors."""
        inputs = {}
        
        # Content classification inputs
        inputs['content_scores'] = pb_utils.get_input_tensor_by_name(
            request, 'content_scores'
        ).as_numpy()
        inputs['content_class'] = pb_utils.get_input_tensor_by_name(
            request, 'content_class'
        ).as_numpy()
        inputs['content_confidence'] = pb_utils.get_input_tensor_by_name(
            request, 'content_confidence'
        ).as_numpy()
        
        # Safety detection inputs
        inputs['safety_scores'] = pb_utils.get_input_tensor_by_name(
            request, 'safety_scores'
        ).as_numpy()
        inputs['is_safe'] = pb_utils.get_input_tensor_by_name(
            request, 'is_safe'
        ).as_numpy()
        inputs['risk_level'] = pb_utils.get_input_tensor_by_name(
            request, 'risk_level'
        ).as_numpy()
        
        # OCR inputs
        inputs['ocr_text'] = pb_utils.get_input_tensor_by_name(
            request, 'ocr_text'
        ).as_numpy()
        inputs['text_confidence'] = pb_utils.get_input_tensor_by_name(
            request, 'text_confidence'
        ).as_numpy()
        inputs['detected_language'] = pb_utils.get_input_tensor_by_name(
            request, 'detected_language'
        ).as_numpy()
        
        return inputs
    
    def _make_moderation_decision(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core business logic for moderation decisions.
        
        Decision Matrix:
        - APPROVED: Safe content, appropriate category, no concerning text
        - REJECTED: Unsafe content, sensitive category, or blocked text
        - REVIEW_REQUIRED: Borderline cases requiring human review
        """
        
        # Extract key values
        content_class = int(inputs['content_class'][0])
        content_confidence = float(inputs['content_confidence'][0])
        safety_scores = inputs['safety_scores'][0]  # [safe, nsfw, violence, hate, drugs]
        is_safe = bool(inputs['is_safe'][0])
        risk_level = inputs['risk_level'][0].decode('utf-8')
        
        # Process OCR text
        ocr_texts = [text.decode('utf-8') for text in inputs['ocr_text']]
        combined_text = ' '.join(ocr_texts).lower()
        
        # Decision logic
        decision_factors = {
            'safety_check': self._evaluate_safety(safety_scores, is_safe, risk_level),
            'content_check': self._evaluate_content(content_class, content_confidence),
            'text_check': self._evaluate_text(combined_text, inputs['text_confidence'])
        }
        
        # Combine factors for final decision
        final_decision = self._combine_decision_factors(decision_factors)
        
        # Create comprehensive result
        result = {
            'moderation_decision': final_decision['decision'],
            'confidence_score': final_decision['confidence'],
            'content_category': self._get_content_category_name(content_class),
            'safety_assessment': f"Risk: {risk_level}, Safe: {is_safe}",
            'extracted_text': ocr_texts,
            'processing_metadata': json.dumps({
                'decision_factors': decision_factors,
                'model_version': 'v2.1.0',
                'processing_time_ms': 0,  # Would track in production
                'content_class_id': content_class,
                'safety_scores': safety_scores.tolist(),
                'risk_level': risk_level
            })
        }
        
        return result
    
    def _evaluate_safety(self, safety_scores: np.ndarray, is_safe: bool, risk_level: str) -> Dict[str, Any]:
        """Evaluate safety-related factors."""
        safe_score = float(safety_scores[0])  # Safe score
        nsfw_score = float(safety_scores[1])  # NSFW score
        
        if not is_safe or risk_level == 'HIGH':
            return {'result': 'REJECT', 'confidence': 0.95, 'reason': 'Safety violation'}
        elif risk_level == 'MEDIUM' or nsfw_score > 0.3:
            return {'result': 'REVIEW', 'confidence': 0.7, 'reason': 'Moderate risk detected'}
        else:
            return {'result': 'APPROVE', 'confidence': safe_score, 'reason': 'Content appears safe'}
    
    def _evaluate_content(self, content_class: int, confidence: float) -> Dict[str, Any]:
        """Evaluate content classification factors."""
        if content_class in self.sensitive_categories:
            return {'result': 'REJECT', 'confidence': confidence, 'reason': 'Sensitive category'}
        elif confidence < self.content_confidence_threshold:
            return {'result': 'REVIEW', 'confidence': confidence, 'reason': 'Low classification confidence'}
        else:
            return {'result': 'APPROVE', 'confidence': confidence, 'reason': 'Appropriate content category'}
    
    def _evaluate_text(self, text: str, text_confidences: np.ndarray) -> Dict[str, Any]:
        """Evaluate OCR text content."""
        if not text.strip():
            return {'result': 'APPROVE', 'confidence': 1.0, 'reason': 'No text detected'}
        
        # Check for blocked keywords
        for keyword in self.blocked_keywords:
            if keyword in text:
                return {'result': 'REJECT', 'confidence': 0.9, 'reason': f'Blocked keyword: {keyword}'}
        
        # Check text confidence
        avg_confidence = np.mean(text_confidences) if len(text_confidences) > 0 else 0.0
        if avg_confidence < self.text_confidence_threshold:
            return {'result': 'REVIEW', 'confidence': avg_confidence, 'reason': 'Low OCR confidence'}
        
        return {'result': 'APPROVE', 'confidence': avg_confidence, 'reason': 'Text content acceptable'}
    
    def _combine_decision_factors(self, factors: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine individual factor decisions into final decision."""
        
        # Priority: Safety > Content > Text
        # If any factor says REJECT, final decision is REJECT
        # If any factor says REVIEW (and none REJECT), final decision is REVIEW
        # Otherwise, APPROVE
        
        results = [factor['result'] for factor in factors.values()]
        confidences = [factor['confidence'] for factor in factors.values()]
        
        if 'REJECT' in results:
            decision = 'REJECTED'
            confidence = max([f['confidence'] for f in factors.values() if f['result'] == 'REJECT'])
        elif 'REVIEW' in results:
            decision = 'REVIEW_REQUIRED'
            confidence = np.mean(confidences)
        else:
            decision = 'APPROVED'
            confidence = np.mean(confidences)
        
        return {
            'decision': decision,
            'confidence': float(confidence)
        }
    
    def _get_content_category_name(self, class_id: int) -> str:
        """Map class ID to human-readable category name."""
        # In production, this would use actual ImageNet class names
        category_map = {
            281: 'tabby_cat',
            285: 'egyptian_cat',
            413: 'assault_rifle',
            939: 'bikini',
            # ... more mappings
        }
        return category_map.get(class_id, f'class_{class_id}')
    
    def _create_response_tensors(self, result: Dict[str, Any]) -> List[pb_utils.Tensor]:
        """Create response tensors from decision result."""
        
        output_tensors = []
        
        # Moderation decision
        decision_tensor = pb_utils.Tensor(
            'moderation_decision',
            np.array([result['moderation_decision']], dtype=object)
        )
        output_tensors.append(decision_tensor)
        
        # Confidence score
        confidence_tensor = pb_utils.Tensor(
            'confidence_score',
            np.array([result['confidence_score']], dtype=np.float32)
        )
        output_tensors.append(confidence_tensor)
        
        # Content category
        category_tensor = pb_utils.Tensor(
            'content_category',
            np.array([result['content_category']], dtype=object)
        )
        output_tensors.append(category_tensor)
        
        # Safety assessment
        safety_tensor = pb_utils.Tensor(
            'safety_assessment',
            np.array([result['safety_assessment']], dtype=object)
        )
        output_tensors.append(safety_tensor)
        
        # Extracted text
        text_tensor = pb_utils.Tensor(
            'extracted_text',
            np.array(result['extracted_text'], dtype=object)
        )
        output_tensors.append(text_tensor)
        
        # Processing metadata
        metadata_tensor = pb_utils.Tensor(
            'processing_metadata',
            np.array([result['processing_metadata']], dtype=object)
        )
        output_tensors.append(metadata_tensor)
        
        return output_tensors
    
    def finalize(self):
        """Clean up resources."""
        logger.info("Content moderation decision logic finalized")
