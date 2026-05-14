import streamlit as st
import pandas as pd
import pickle

from sklearn.metrics.pairwise import cosine_similarity

# Load Saved Files


with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


with open("job_vectors.pkl", "rb") as file:
    job_vectors = pickle.load(file)


with open("jobs_data.pkl", "rb") as file:
    jobs_df = pickle.load(file)


# Streamlit UI

st.set_page_config(
    page_title="Hybrid Job Recommendation",
    layout="wide"
)


st.title("Hybrid Job Recommendation System")

st.write(
    "Find jobs based on your skills and preferences"
)

# User Input


skills = st.text_input(
    "Enter your skills"
)

role = st.text_input(
    "Preferred role"
)

location = st.text_input(
    "Preferred location"
)

experience = st.slider(
    "Experience in years",
    0,
    10,
    1
)


# Recommendation Button


if st.button("Recommend Jobs"):

    user_profile = (
        skills + " " +
        role + " " +
        location + " " +
        str(experience)
    )

    user_vector = vectorizer.transform(
        [user_profile]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        job_vectors
    )

    similarity_scores = similarity_scores.flatten()

    jobs_df["score"] = similarity_scores

    recommended_jobs = jobs_df.sort_values(
        by="score",
        ascending=False
    ).head(10)


    st.subheader("Top Recommended Jobs")


    for _, row in recommended_jobs.iterrows():

        st.markdown("---")

        st.markdown(
            f"### {row['title']}"
        )

        st.write(
            f"Company: {row['company']}"
        )

        st.write(
            f"Location: {row['location']}"
        )

        st.write(
            f"Skills: {row['skills']}"
        )

        st.write(
            f"Match Score: {round(row['score'], 2)}"
        )