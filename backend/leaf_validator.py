"""Leaf image validation module - Simple validation to reject obvious non-leaf images."""

import numpy as np
from PIL import Image
from typing import Tuple

class LeafValidator:
    """Simple validator to reject obvious non-leaf images."""
    
    def __init__(self):
        # Very lenient thresholds - only reject obvious non-leaf images
        self.min_size = 500      # Minimum area in pixels
        self.max_aspect_ratio = 10.0  # Very lenient aspect ratio
        self.min_green_ratio = 0.05   # Very low threshold for green
        
    def validate_leaf_image(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Simple validation - only reject obvious non-leaf images.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Convert to numpy array
            img_array = np.array(image.convert('RGB'))
            
            # Check 1: Basic size check
            if not self._check_minimum_size(img_array):
                return False, "Gambar terlalu kecil untuk dianalisis."
            
            # Check 2: Very lenient aspect ratio
            if not self._check_aspect_ratio(img_array):
                return False, "Gambar memiliki rasio yang terlalu ekstrem."
            
            # Check 3: Basic green presence (very lenient)
            if not self._check_basic_green(img_array):
                return False, "Gambar tidak menunjukkan karakteristik daun tanaman."
            
            # Check 4: Reject obvious artificial patterns
            if self._is_obviously_artificial(img_array):
                return False, "Gambar tidak menunjukkan daun tanaman yang alami."
            
            # If all basic checks pass, accept the image
            return True, "Gambar daun valid"
            
        except Exception as e:
            return False, f"Error dalam validasi gambar: {str(e)}"
    
    def _check_minimum_size(self, img_array: np.ndarray) -> bool:
        """Check if image is large enough."""
        height, width = img_array.shape[:2]
        area = height * width
        return area >= self.min_size
    
    def _check_aspect_ratio(self, img_array: np.ndarray) -> bool:
        """Very lenient aspect ratio check."""
        height, width = img_array.shape[:2]
        aspect_ratio = max(width, height) / min(width, height)
        return aspect_ratio <= self.max_aspect_ratio
    
    def _check_basic_green(self, img_array: np.ndarray) -> bool:
        """Very basic green check - only reject if absolutely no green."""
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Simple check: green channel should be higher than red and blue in some pixels
        green_mask = (g > r) & (g > b) & (g > 30)
        green_ratio = np.sum(green_mask) / (img_array.shape[0] * img_array.shape[1])
        
        return green_ratio >= self.min_green_ratio
    
    def _is_obviously_artificial(self, img_array: np.ndarray) -> bool:
        """Check for obvious artificial patterns that are clearly not leaves."""
        try:
            # Check for very uniform colors (solid colors)
            r_std = np.std(img_array[:, :, 0])
            g_std = np.std(img_array[:, :, 1])
            b_std = np.std(img_array[:, :, 2])
            
            # If all channels have very low standard deviation, it's probably a solid color
            if r_std < 5 and g_std < 5 and b_std < 5:
                return True
            
            # Check for obvious geometric patterns (simple shapes)
            # This is a very basic check - if the image is too uniform, it's probably not a leaf
            total_variance = np.var(img_array)
            if total_variance < 10:  # Very low variance = very uniform = probably not natural
                return True
            
            # Check for obvious text patterns (high contrast edges in regular patterns)
            gray = np.mean(img_array, axis=2)
            edges = np.abs(np.diff(gray, axis=1)) + np.abs(np.diff(gray, axis=0))
            edge_density = np.sum(edges > 50) / edges.size
            
            # If there are too many sharp edges in regular patterns, it might be text
            if edge_density > 0.1:  # Very high edge density might indicate text or artificial patterns
                return True
            
            return False
            
        except Exception:
            return False

def validate_leaf_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Convenience function to validate a leaf image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = LeafValidator()
    return validator.validate_leaf_image(image)
