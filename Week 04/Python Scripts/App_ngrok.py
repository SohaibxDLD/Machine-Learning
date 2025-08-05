from pyngrok import ngrok

!ngrok authtoken 30dLxqJ7zHQFeiIpbFMsX6Ot6wm_6UQQTqwrCv3kjoe3yeYUZ

!streamlit run spotify_genre_app.py --server.port 8501 --server.address=0.0.0.0 &>/dev/null &

try:
    public_url = ngrok.connect(addr='8501', proto='http', bind_tls=True)
    print("Your app is live at:", public_url)
except Exception as e:
    print("Error creating tunnel:", e)
    print("\nAlternative: Use Colab's built-in preview:")
    print("1. Wait for Streamlit to start")
    print("2. Click the '>' icon next to the output")
    print("3. Select 'Preview on port 8501'")
