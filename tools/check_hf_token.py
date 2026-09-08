"""
Smoke test for Hugging Face authentication.

The token is read from the environment (HUGGINGFACE_TOKEN or HF_TOKEN); see
.env.example. Never hardcode a token here.

    python testHFToken.py [--model private/model]
"""

import argparse

from transformers import AutoModel

from transformer2.config import huggingface_token


def main():
    parser = argparse.ArgumentParser(description='Check that a Hugging Face token works')
    parser.add_argument('--model', default='private/model',
                        help='Model repository to load')
    args = parser.parse_args()

    token = huggingface_token(required=True)
    model = AutoModel.from_pretrained(args.model, token=token)
    print(f"Loaded {args.model} successfully ({model.config.model_type}).")


if __name__ == "__main__":
    main()
