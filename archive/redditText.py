import praw

reddit = praw.Reddit(
    client_id="***REMOVED***",
    client_secret="***REMOVED***",
    redirect_uri="http://localhost:8080",
    user_agent="testscript by u/fakebot3",
)
print(reddit.auth.url(scopes=["identity"], state="...", duration="permanent"))