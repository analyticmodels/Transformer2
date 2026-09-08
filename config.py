"""
Credential and configuration loading.

Secrets are read from the environment, optionally populated from a local `.env`
file that is never committed. Nothing in this repository should contain a literal
API key, token or secret.

Copy `.env.example` to `.env` and fill in the values:

    cp .env.example .env

Environment variables always win over `.env`, so CI and shell exports keep working
without a file present.
"""

import os

from dotenv import load_dotenv

# Populate os.environ from .env without clobbering variables already exported.
# Safe to call repeatedly; a missing .env is not an error.
load_dotenv(override=False)


class MissingCredentialError(RuntimeError):
    """Raised when a required secret is not configured."""


def get_secret(name, default=None, required=False):
    """
    Read a secret from the environment.

    Args:
        name (str): Environment variable name
        default: Value to use when the variable is unset
        required (bool): Raise instead of returning the default when unset

    Returns:
        str or None: The value, stripped of surrounding whitespace

    Raises:
        MissingCredentialError: If required and unset
    """
    value = os.environ.get(name)
    if value is not None:
        value = value.strip()
    if not value:
        if required:
            raise MissingCredentialError(
                f"{name} is not set. Add it to .env (see .env.example) or export it:\n"
                f"    export {name}=..."
            )
        return default
    return value


def reddit_credentials(client_id=None, client_secret=None, user_agent=None, required=False):
    """
    Resolve Reddit API credentials, preferring explicit arguments over the environment.

    Passing secrets as command-line arguments leaks them into shell history and the
    process list, so the environment is the intended source; the arguments exist only
    for callers that already hold the values.

    Args:
        client_id (str, optional): Overrides REDDIT_CLIENT_ID
        client_secret (str, optional): Overrides REDDIT_CLIENT_SECRET
        user_agent (str, optional): Overrides REDDIT_USER_AGENT
        required (bool): Raise if the id or secret cannot be resolved

    Returns:
        tuple: (client_id, client_secret, user_agent)

    Raises:
        MissingCredentialError: If required and the id or secret is missing
    """
    client_id = client_id or get_secret('REDDIT_CLIENT_ID', required=required)
    client_secret = client_secret or get_secret('REDDIT_CLIENT_SECRET', required=required)
    user_agent = (
        user_agent
        or get_secret('REDDIT_USER_AGENT')
        or 'python:llm-training-data:v1.0'
    )
    return client_id, client_secret, user_agent


def huggingface_token(required=False):
    """
    Resolve a Hugging Face access token.

    Accepts either HUGGINGFACE_TOKEN or the HF_TOKEN name the huggingface_hub
    tooling uses.

    Args:
        required (bool): Raise if no token is configured

    Returns:
        str or None: The token
    """
    token = get_secret('HUGGINGFACE_TOKEN') or get_secret('HF_TOKEN')
    if not token and required:
        raise MissingCredentialError(
            "No Hugging Face token found. Set HUGGINGFACE_TOKEN in .env "
            "(see .env.example) or export it."
        )
    return token
