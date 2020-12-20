#!/usr/bin/env python
# coding: utf-8

# FACE RECOGNITION WITH SIAMESE NETWORK

## Import necessary libraries
import numpy as np
# import scann
import streamlit as st
import pickle
import tensorflow as tf
from skimage import io, transform, color
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

# Headings
st.title("Face Recognition with Siamese Network")
st.sidebar.title("Face Recognition with Siamese Network")

st.markdown("This app is a dashboard for Face Recognition via Siamese Network")
st.sidebar.markdown("This app is a dashboard for Face Recognition via Siamese Network️")

st.image("siam.png", width = 700)

def triplet_loss(alpha, emb_size):
    def loss(y_true, y_pred):  # Euclidean distance
        anchor, positive, negative = y_pred[:, :emb_size], y_pred[:, emb_size: 2 * emb_size], y_pred[:, 2 * emb_size:]
        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)

        return tf.maximum(positive_dist - negative_dist + alpha, 0.)

    return loss

## Obtain the files
with open("embeddings.pickle", "rb") as f:
    embedding = pickle.load(f)

mod = tf.keras.models.load_model(
    "embedding_model.h5",
    custom_objects = {
        'loss': triplet_loss(0.2, 16)
    },
    compile = False
)
mod.compile(
    loss = triplet_loss(0.2, 16),
    optimizer="adam"
)

## Classifying as recognized and not-recognized by clustering
def GetNeighbors(db_embeddings, x_test):
    x_train = np.array(list(db_embeddings.values()))
    x_train = np.squeeze(x_train)
    x_train = x_train / np.sqrt(np.sum(x_train ** 2, axis=1, keepdims=True))
    y_train = np.array(list(db_embeddings.keys()))

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(x_train, y_train)
    neighbor = classifier.predict(x_test)[0]
    neighbor_embedding = db_embeddings[neighbor]
    distance = cosine_similarity(neighbor_embedding, x_test)

    return neighbor, distance

img = st.sidebar.file_uploader("Choose a .jpg image", type="jpg")
if img is not None:
    n = 64

    original_img = io.imread(img)
    img = transform.resize(original_img, (n, n))
    img = color.rgb2gray(img)
    st.sidebar.image(original_img, width=180, caption="Test Face")

    x_test = mod.predict(img.reshape(-1, n, n, 1))

    neighbor, distance = GetNeighbors(embedding, x_test)

    # st.image(embedding[neighbor], width=180, caption="Nearest Match")

    if np.squeeze(distance) < 0.95:
        st.sidebar.subheader("Face not recognized!")

    else:
        st.sidebar.subheader("Face of person: " + str(np.squeeze(neighbor).tolist()))
