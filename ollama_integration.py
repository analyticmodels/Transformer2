"""
Ollama integration for the Transformer2 project.
This module provides functionality to use local Ollama LLMs for training and inference.
"""

import requests
import json
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union
import numpy as np

class OllamaTokenizer:
    """Tokenizer wrapper for Ollama models."""
    
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.api_base = "http://localhost:11434/api"
        # Cache for vocab
        self._vocab_size = None
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using Ollama's tokenizer."""
        try:
            response = requests.post(
                f"{self.api_base}/tokenize",
                json={"model": self.model_name, "text": text}
            )
            response.raise_for_status()
            return response.json()["tokens"]
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize text with Ollama: {str(e)}")
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        try:
            response = requests.post(
                f"{self.api_base}/generate",
                json={
                    "model": self.model_name,
                    "prompt": "",
                    "raw": True,
                    "tokens": token_ids
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise RuntimeError(f"Failed to decode tokens with Ollama: {str(e)}")
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the model."""
        if self._vocab_size is None:
            try:
                # Use a sample text to determine vocab size from model info
                response = requests.post(
                    f"{self.api_base}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": " ",
                        "raw": True,
                        "options": {"num_predict": 1}
                    }
                )
                response.raise_for_status()
                model_info = response.json()
                self._vocab_size = model_info.get("vocab_size", 32000)  # Default to 32k if not provided
            except Exception as e:
                raise RuntimeError(f"Failed to get vocab size from Ollama: {str(e)}")
        return self._vocab_size

class OllamaModel(nn.Module):
    """PyTorch wrapper for Ollama models."""
    
    def __init__(self, model_name: str = "llama2", max_length: int = 512):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.api_base = "http://localhost:11434/api"
        self.tokenizer = OllamaTokenizer(model_name)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using Ollama's API.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional target labels for training [batch_size, seq_len]
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter (0.0 to 1.0)
        
        Returns:
            Dict containing logits and loss if labels provided
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Process each sequence in the batch
        all_logits = []
        total_loss = 0.0
        
        for i in range(batch_size):
            # Convert token IDs to text
            curr_ids = input_ids[i].tolist()
            input_text = self.tokenizer.decode(curr_ids)
            
            try:
                # Get next token probabilities from Ollama
                response = requests.post(
                    f"{self.api_base}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": input_text,
                        "raw": True,
                        "options": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "num_predict": 1
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                # Convert logits to PyTorch tensor
                logits = torch.tensor(
                    result.get("logits", [0] * self.tokenizer.get_vocab_size()),
                    device=device
                ).unsqueeze(0)  # Add batch dimension
                
                # Repeat logits for sequence length
                logits = logits.expand(seq_len, -1)  # [seq_len, vocab_size]
                all_logits.append(logits)
                
                # Calculate loss if labels provided
                if labels is not None:
                    curr_labels = labels[i]
                    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                    total_loss += loss_fn(logits, curr_labels)
            
            except Exception as e:
                raise RuntimeError(f"Failed to get predictions from Ollama: {str(e)}")
        
        # Stack all logits and reshape
        logits = torch.stack(all_logits, dim=0)  # [batch_size, seq_len, vocab_size]
        
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = total_loss / batch_size
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate text using Ollama's API.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter (0.0 to 1.0)
        
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Each row grows by max_new_tokens, so the result cannot be written back into
        # a clone of input_ids -- collect the extended rows and stack them instead.
        sequences = []

        for i in range(batch_size):
            curr_ids = input_ids[i].tolist()
            input_text = self.tokenizer.decode(curr_ids)

            try:
                response = requests.post(
                    f"{self.api_base}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": input_text,
                        "raw": True,
                        "options": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "num_predict": max_new_tokens
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                # Get generated tokens
                new_tokens = result.get("tokens", [])[:max_new_tokens]
                
                # Pad sequence if needed
                if len(new_tokens) < max_new_tokens:
                    new_tokens.extend([0] * (max_new_tokens - len(new_tokens)))
                
                # Append the new tokens to this row of the batch
                sequences.append(torch.cat([
                    input_ids[i],
                    torch.tensor(new_tokens, device=device, dtype=input_ids.dtype)
                ]))

            except Exception as e:
                raise RuntimeError(f"Failed to generate text with Ollama: {str(e)}")

        return torch.stack(sequences, dim=0)
