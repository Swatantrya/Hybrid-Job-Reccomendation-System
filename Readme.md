# Hybrid Job Recommendation System

A machine learning based hybrid job recommendation system that suggests personalized jobs using content-based filtering and collaborative filtering techniques.

## Features

- Personalized job recommendations
- TF-IDF vectorization
- Cosine similarity matching
- Hybrid recommendation approach
- Streamlit web interface
- Large dataset handling

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Dataset Files

- users.csv
- jobs.csv
- interactions.csv

## Model Files

- tfidf_vectorizer.pkl
- job_vectors.pkl
- jobs_data.pkl

## Project Workflow

```text
User Input
   ↓
TF-IDF Vectorization
   ↓
Content Similarity
   ↓
Collaborative Filtering
   ↓
Hybrid Scoring
   ↓
Job Recommendations
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
python -m streamlit run app.py
```

## Project Structure

```text
Hybrid-Job-Recommendation-System/
│
├── app.py
├── hybrid job recommendation system.ipynb
├── users.csv
├── jobs.csv
├── interactions.csv
├── tfidf_vectorizer.pkl
├── job_vectors.pkl
├── jobs_data.pkl
└── README.md
```

## Future Improvements

- Authentication system
- Real-time recommendations
- Job scraping pipeline
- Deep learning recommendation model
- Deployment on cloud platforms

## Author

Swatantrya Ganguli