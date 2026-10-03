import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os
import matplotlib.pyplot as plt


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)


# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #aaaaaa;
}

.genre {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# LOAD MODEL
# ==============================

try:
    model = joblib.load("model.pkl")
except:
    st.error("model.pkl not found. Please train the model first.")
    st.stop()


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(file_path):

    y, sr = librosa.load(file_path, duration=30)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_features = np.mean(mfcc.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    chroma_features = np.mean(chroma.T, axis=0)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    centroid_features = np.mean(
        centroid.T,
        axis=0
    )

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    rolloff_features = np.mean(
        rolloff.T,
        axis=0
    )

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    zcr_features = np.mean(
        zcr.T,
        axis=0
    )

    # Combine
    features = np.hstack([
        mfcc_features,
        chroma_features,
        centroid_features,
        rolloff_features,
        zcr_features
    ])

    return features


# ==============================
# TITLE
# ==============================

st.markdown(
    '<div class="title">🎵 Music Genre Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Classify your music into Classical, Rock or Metal'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ==============================
# FILE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "🎧 Upload an audio file",
    type=[
        "au",
        "wav",
        "mp3",
        "flac",
        "ogg",
        "m4a",
        "aac"
    ]
)


# ==============================
# PREDICTION
# ==============================

if uploaded_file is not None:

    st.success(
        f"File uploaded: {uploaded_file.name}"
    )

    # Audio player
    st.audio(
        uploaded_file
    )

    st.write("")

    if st.button(
        "🔍 Analyze Song",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing the music..."
        ):

            # Create temporary file
            file_extension = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    uploaded_file.read()
                )

                temp_path = temp_file.name


            try:

                # Extract features
                features = extract_features(
                    temp_path
                )

                features = features.reshape(
                    1, -1
                )

                # Prediction
                prediction = model.predict(
                    features
                )

                probabilities = model.predict_proba(
                    features
                )[0]

                predicted_genre = prediction[0]

                # Delete temporary file
                os.remove(temp_path)


                # ==============================
                # RESULT
                # ==============================

                st.markdown("---")

                st.subheader(
                    "🎯 Prediction Result"
                )

                st.markdown(
                    f'<div class="genre">'
                    f'{predicted_genre.upper()}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.write("")


                # ==============================
                # CONFIDENCE
                # ==============================

                st.subheader(
                    "📊 Genre Confidence"
                )

                for genre, probability in zip(
                    model.classes_,
                    probabilities
                ):

                    percentage = (
                        probability * 100
                    )

                    st.write(
                        f"**{genre.capitalize()}** "
                        f"{percentage:.2f}%"
                    )

                    st.progress(
                        float(probability)
                    )


                # ==============================
                # BAR CHART
                # ==============================

                st.subheader(
                    "📈 Probability Distribution"
                )

                fig, ax = plt.subplots()

                ax.bar(
                    model.classes_,
                    probabilities * 100
                )

                ax.set_ylabel(
                    "Confidence (%)"
                )

                ax.set_xlabel(
                    "Genre"
                )

                ax.set_ylim(
                    0,
                    100
                )

                st.pyplot(fig)


                # ==============================
                # CONFIDENCE MESSAGE
                # ==============================

                highest = max(
                    probabilities
                )

                if highest >= 0.70:

                    st.success(
                        "✅ The model is reasonably "
                        "confident about this prediction."
                    )

                elif highest >= 0.50:

                    st.warning(
                        "⚠️ The model has moderate "
                        "confidence in this prediction."
                    )

                else:

                    st.error(
                        "⚠️ The model is uncertain "
                        "about this prediction."
                    )


            except Exception as e:

                st.error(
                    f"Error processing audio: {e}"
                )

                if os.path.exists(
                    temp_path
                ):
                    os.remove(temp_path)


# ==============================
# INFORMATION
# ==============================

st.markdown("---")

st.subheader(
    "ℹ️ About this Project"
)

st.write("""
This Music Genre Classification system uses
Machine Learning to classify audio into three
genres:

🎻 Classical

🎸 Rock

🤘 Metal

Audio features such as MFCC, Chroma,
Spectral Centroid, Spectral Rolloff and
Zero Crossing Rate are extracted from the
input audio and provided to a Random Forest
classifier.
""")