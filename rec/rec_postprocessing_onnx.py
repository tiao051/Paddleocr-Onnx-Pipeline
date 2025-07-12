"""
PP-OCRv5 Recognition Postprocessing for ONNX
Based on PaddleOCR CTCLabelDecode

Converts CTC model output to text strings
"""

import numpy as np
from typing import List, Tuple
import os

class CTCLabelDecodeONNX:
    """
    CTC Label Decode for ONNX PP-OCRv5 Recognition Model
    
    Converts CTC probability output to text strings using:
    1. Argmax to get predicted indices
    2. CTC decoding to remove blanks and duplicates  
    3. Character mapping to get final text
    """
    
    def __init__(self, 
                 character_dict_path: str = None,
                 use_space_char: bool = True):
        """
        Args:
            character_dict_path: Path to character dictionary file
            use_space_char: Whether to include space character
        """
        self.use_space_char = use_space_char
        
        # Load character set
        if character_dict_path and os.path.exists(character_dict_path):
            self.character_str = self._load_charset_from_file(character_dict_path)
        else:
            # Default PP-OCRv5 character set
            self.character_str = self._get_default_charset()
            
        # Build character list
        self.character = ['blank']  # CTC blank token at index 0
        
        if self.use_space_char:
            self.character.append(' ')
            
        for char in self.character_str:
            self.character.append(char)
            
        self.dict = {char: i for i, char in enumerate(self.character)}
        
    def _load_charset_from_file(self, path: str) -> str:
        """Load character set from file"""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        character_str = ''.join([line.strip() for line in lines])
        return character_str
        
    def _get_default_charset(self) -> str:
        """Default PP-OCRv5 character set"""
        # This is the standard PaddleOCR character set
        character_str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        character_str += "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        return character_str
    
    def decode_ctc(self, preds_idx: np.ndarray) -> str:
        """
        CTC decoding: remove duplicates and blanks
        
        Args:
            preds_idx: Predicted indices array (seq_len,)
            
        Returns:
            Decoded text string
        """
        result = []
        prev_idx = -1
        
        for idx in preds_idx:
            # Skip blank token (index 0) and consecutive duplicates
            if idx != 0 and idx != prev_idx:
                if idx < len(self.character):
                    result.append(self.character[idx])
            prev_idx = idx
            
        return ''.join(result)
    
    def __call__(self, preds: np.ndarray, 
                 label: np.ndarray = None) -> List[Tuple[str, float]]:
        """
        Main postprocessing function
        
        Args:
            preds: Model output (batch_size, seq_len, num_classes)
            label: Ground truth labels (optional, for training)
            
        Returns:
            List of (text, confidence) tuples
        """
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]  # Take last output if multiple
            
        # Handle different input shapes
        if preds.ndim == 4:
            preds = preds.squeeze(axis=1)  # Remove height dimension if present
        elif preds.ndim == 2:
            preds = preds[np.newaxis, :]   # Add batch dimension
            
        batch_size = preds.shape[0]
        
        if len(preds.shape) == 3:
            exp_preds = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
            probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
        else:
            probs = preds
        # Get predicted indices using argmax
        preds_idx = np.argmax(probs, axis=2)  # (batch_size, seq_len)
        
        # Calculate confidence scores
        max_probs = np.max(probs, axis=2)    # (batch_size, seq_len)

        results = []
        for i in range(batch_size):
            # Decode CTC for this sequence
            text = self.decode_ctc(preds_idx[i])
            sequence_probs = []
            prev_idx = -1
            for j, idx in enumerate(preds_idx[i]):
            # Only consider non-blank and non-duplicate characters
                if idx != 0 and idx != prev_idx:  # 0 is blank token
                    sequence_probs.append(max_probs[i][j])
                prev_idx = idx
            if sequence_probs:
                confidence = float(np.mean(sequence_probs))
            else:
                confidence = 0.0
            
        results.append((text, confidence))
            
        return results


def test_ctc_decode():
    """Test CTC decoding with sample data"""
    print("=== Testing PP-OCRv5 CTC Decoding ===")
    
    # Initialize decoder
    decoder = CTCLabelDecodeONNX(use_space_char=True)
    
    print(f"Character set size: {len(decoder.character)}")
    print(f"Sample characters: {decoder.character[:10]}...")
    
    # Create fake prediction (simulating "HELLO" output)
    batch_size, seq_len, num_classes = 1, 20, len(decoder.character)
    fake_preds = np.random.rand(batch_size, seq_len, num_classes) * 0.1
    
    # Manually set high probabilities for "HELLO"
    hello_indices = []
    for char in "HELLO":
        if char in decoder.dict:
            hello_indices.append(decoder.dict[char])
    
    # Set predictions to spell "HELLO" with blanks
    if len(hello_indices) == 5:
        fake_preds[0, 1, hello_indices[0]] = 0.9  # H
        fake_preds[0, 3, hello_indices[1]] = 0.9  # E  
        fake_preds[0, 5, hello_indices[2]] = 0.9  # L
        fake_preds[0, 7, hello_indices[3]] = 0.9  # L
        fake_preds[0, 9, hello_indices[4]] = 0.9  # O
    
    # Decode
    results = decoder(fake_preds)
    
    print(f"Decoded results: {results}")
    print("✅ CTC decoding test completed!")
    
    return decoder


if __name__ == "__main__":
    import os
    
    # Test the decoder
    test_ctc_decode()
    
    print("\n🎯 Next steps:")
    print("1. Test with real ONNX model output")
    print("2. Integrate with recognition pipeline")  
    print("3. Add batch processing support")
    print("4. Optimize for performance")
