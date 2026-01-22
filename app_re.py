import streamlit as st
from recommender import MovieRecommender


@st.cache_resource
def load_recommender():
    return MovieRecommender()

recommender = load_recommender()


st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.write("Content-Based + Item-Based Collaborative Filtering")

# Autocomplete dropdown (filters as user types)
movie_list = sorted(recommender.title_to_id.index)

selected_movie = st.selectbox(
    "Start typing a movie name:",
    movie_list
)

if st.button("Recommend"):
    st.subheader("🎯 Content-Based Recommendations")
    for movie in recommender.recommend_content(selected_movie):
        st.write("👉", movie)

    st.markdown("---")

    st.subheader("👥 Item-Based Collaborative Recommendations")
    st.caption("Recommended based on other users’ ratings")

    for movie in recommender.recommend_item(selected_movie):
        st.write("👉", movie)
