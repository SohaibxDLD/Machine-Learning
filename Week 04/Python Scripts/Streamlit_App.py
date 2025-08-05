%%writefile spotify_genre_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Genre mapping
GENRE_MAPPING = {
    0: 'rock', 1: 'pop', 2: 'metal', 3: 'electronic',
    4: 'hiphop', 5: 'jazz', 6: 'classical', 7: 'folk',
    8: 'reggae', 9: 'latin', 10: 'world', 11: 'other'
}

# Load all models
@st.cache_resource
def load_models():
    return {
        'Random Forest': joblib.load('rf_model.pkl'),
        'Logistic Regression': joblib.load('lr_model.pkl'),
        'XGBoost': joblib.load('xgb_model.pkl'),
        'Decision Tree': joblib.load('dt_model.pkl')
    }

# Load sample data
@st.cache_data
def load_data():
    return pd.read_csv('Spotify Dataset.csv', nrows=20000)

def show_data_tab(df):
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.write("First 10 rows:")
        st.dataframe(df.head(10))
    with col2:
        st.write("Dataset info:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    st.write("Descriptive statistics:")
    st.dataframe(df.describe())

def show_visualizations_tab(df):
    st.header("Data Visualizations")

    plot_type = st.selectbox("Choose visualization type:",
                           ["Histogram", "Box Plot", "Correlation Heatmap"])

    if plot_type == "Histogram":
        feature = st.selectbox("Select feature:", df.select_dtypes(include=np.number).columns)
        plt.figure(figsize=(10, 5))
        sns.histplot(df[feature], kde=True)
        st.pyplot(plt)

    elif plot_type == "Box Plot":
        feature = st.selectbox("Select feature:", df.select_dtypes(include=np.number).columns)
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[feature])
        st.pyplot(plt)

    elif plot_type == "Correlation Heatmap":
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

def show_prediction_tab(models):
    st.header("Genre Prediction")

    # Model selection
    model_name = st.selectbox("Choose model:", list(models.keys()))
    model = models[model_name]

    # User input
    with st.expander("Input Features", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            energy_tempo = st.slider('Energy × Tempo', 0.0, 100.0, 50.0)
            dance_valence = st.slider('Danceability × Valence', 0.0, 1.0, 0.5)
            speech_acoustic = st.slider('Speechiness / Acousticness', 0.0, 100.0, 1.0)
            loudness_instrumental = st.slider('Loudness × (1 - Instrumentalness)', -60.0, 0.0, -10.0)

        with col2:
            liveness_tempo = st.slider('Liveness × Tempo', 0.0, 1.0, 0.5)
            key = st.slider('Key (0=C, 1=C#...)', 0, 11, 5)
            mode = st.selectbox('Mode', [0, 1])
            duration_ms = st.slider('Duration (ms)', 0, 600000, 180000)
            popularity = st.slider('Popularity', 0, 100, 50)

    # Prepare input DataFrame
    input_data = {
        'energy_tempo': energy_tempo,
        'dance_valence': dance_valence,
        'speech_acoustic': speech_acoustic,
        'loudness_instrumental': loudness_instrumental,
        'liveness_tempo': liveness_tempo,
        'key': key,
        'mode': mode,
        'duration_ms': duration_ms,
        'popularity': popularity
    }
    input_df = pd.DataFrame([input_data])

    if st.button('Predict Genre'):
        # Preprocess input
        pt = PowerTransformer()
        X = pt.fit_transform(input_df)

        # Predict
        prediction = model.predict(X)[0]
        genre = GENRE_MAPPING.get(prediction, 'unknown')

        # Display results
        st.success(f"Predicted Genre: **{genre.upper()}**")

        # Show probabilities if available
        if hasattr(model, 'predict_proba'):
            st.subheader("Prediction Probabilities")
            probs = model.predict_proba(X)[0]

            prob_df = pd.DataFrame({
                'Genre': [GENRE_MAPPING[i] for i in range(len(probs))],
                'Probability': probs
            }).sort_values('Probability', ascending=False)

            st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))

            # Plot probabilities
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Probability', y='Genre', data=prob_df, palette='viridis')
            plt.title('Genre Prediction Probabilities')
            st.pyplot(plt)

def main():
    st.set_page_config(layout="wide", page_title="Spotify Genre Classifier")
    st.title("🎵 Spotify Music Genre Analysis")

    # Load data and models
    df = load_data()
    models = load_models()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Dataset", "Visualizations", "Predictions"])

    with tab1:
        show_data_tab(df)

    with tab2:
        show_visualizations_tab(df)

    with tab3:
        show_prediction_tab(models)

if __name__ == "__main__":
    import io
    main()
