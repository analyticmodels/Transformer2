"""
Reddit OAuth URL helper (archived).

Credentials are read from the environment / .env -- see .env.example.
"""

import os
import sys

import praw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import reddit_credentials  # noqa: E402

client_id, client_secret, user_agent = reddit_credentials(required=True)

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=os.environ.get("REDDIT_REDIRECT_URI", "http://localhost:8080"),
    user_agent=user_agent,
)
print(reddit.auth.url(scopes=["identity"], state="...", duration="permanent"))
