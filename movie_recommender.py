import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
st.title("Movie Recommendation system")
movies=pd.read_csv(r"D:\data\archive3\movie.csv")
ratings=pd.read_csv(r"D:\data\archive3\rating.csv")
user_encoder=LabelEncoder()
movie_encoder=LabelEncoder()
ratings["user"]=user_encoder.fit_transform(ratings["userId"])
ratings["movie"]=movie_encoder.fit_transform(ratings["movieId"])
model=load_model("movie_recommender.keras")
user_id=st.number_input("Enter user ID ",min_value=1)
if st.button("Recommend"):
    encoder_user=user_encoder.transform([user_id])[0]
    watched=ratings[ratings["userId"]==user_id]["movieId"].tolist()
    all_movies=ratings["movieId"].unique()
    unseen=np.setdiff1d(all_movies,watched)
    encoded_movies=movie_encoder.transform(unseen)
    user_array=np.full(len(unseen),encoder_user)
    predictions=model.predict([user_array,encoded_movies],verbose=0)
    predictions=predictions.flatten()
    top_idx=np.argsort(predictions)[-10:][::-1]
    recommended_ids=unseen[top_idx]
    recommeded_movies=movies[movies["movieId"].isin(recommended_ids)]
    st.write(recommeded_movies[["title","genres"]])
