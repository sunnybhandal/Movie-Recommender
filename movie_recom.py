from pandas.core.frame import DataFrame
import streamlit as st
#import streamlit.component.v1 as stc
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

def load_data(data):
    # Load Movies Metadata
    df = pd.read_csv(data)

    return df
    
def vectorize_text(df):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])

    # get cosine
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    return cosine_sim

@st.cache
def get_recommendations(title, cosine_sim, df, num_of_rec = 5):
    df = df.reset_index()
    indices = pd.Series(df.index, index=df['title'])

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:num_of_rec+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]

def main():
    
    st.title("Movie Recommendation App")
    
    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # load data
    df = load_data("mini_data.csv")

    if choice == "Home":
        st.subheader("Home")
        st.text("Sample Data")
        st.dataframe(df.head(10))
        
    elif choice == "Recommend":
        st.subheader("Get a Movie Recommendation")
        cosine_sim = vectorize_text(df)
        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number", 3, 10, 5)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    result = get_recommendations(search_term, cosine_sim, df, num_of_rec)
                    with st.expander("Results as JSON"):
                        results_json = result.to_dict()
                        st.write(results_json)
                except Exception as e:
                    result = "Not Found"

                st.write("Movies:", result)
    
    else:
        st.subheader("About")
        st.text("Built with Stremlit")
    
if __name__ == '__main__':
    main()