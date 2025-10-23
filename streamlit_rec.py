import streamlit as st
from pathlib import Path
from PIL import Image
import json
import numpy as np
from image_caption1 import build_index, build_recommendation, load_index, IMAGES_DIR, META_PATH

if "keep_alive" not in st.session_state:
    st.session_state.keep_alive = True

st.title("Image Recommendation algorithm (BLIP + CLIP + FAISS)")

st.header("Build/Rebuild Index")
if st.button("Build FAISS Index from Image folder"):
    try:
        build_index()
        st.success("Index build successfully!")
    except Exception as e:
        st.error(f"Error while building index: {e}")

meta_exist = META_PATH.exists()

if meta_exist:
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        images_list = [f["filename"] for f in meta["files"]]

st.header("Choose Images you like")
if not meta_exist:
    st.info("Please build the index first")

else:
    images_liked = st.multiselect("Select one or more images", images_list)

    cols = st.columns(3)
    for i, fn in enumerate(images_liked):
        img_path = IMAGES_DIR / fn
        if img_path.exists():
            with cols[i % 3]:
                st.image(Image.open(img_path), caption=fn)

if st.button("Get recommendation") and images_liked:
    try:
        recs = build_recommendation(images_liked, top_k=3)
        st.subheader("Recommend Images")

        for r in recs:
            st.image(Image.open(r["path"]), caption=f"{r['filename']} (Score: {r['score']:.4f})\nCaption: {r['caption']}")
    except Exception as e:
        st.error(f"Error while generating recommendation>> {e}")

