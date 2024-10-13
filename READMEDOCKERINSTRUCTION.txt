{PATH on CMD : cd "C:\Users\shlok\OneDrive\Desktop\sample set\rag_qa_model"}
{Build the Docker Image: docker build -t qa-app . }
{Run the Docker Container: docker run -p 8501:8501 qa-app }

This will map port 8501 in the container to your local machine, allowing you to access the Streamlit app via http://localhost:8501