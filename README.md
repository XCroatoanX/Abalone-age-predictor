docker build -f .dockerfile -t abalone-app .
docker run -p 8501:8501 abalone-app
