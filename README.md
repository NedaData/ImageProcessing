# ImageProcessing
Image Recommendation Engine (BLIP + CLIP + FAISS)

This project builds a content-based image recommendation system using the following components:

Component	Purpose
BLIP (Salesforce/blip-image-captioning-large)	Automatically generates captions for images.
CLIP (openai/clip-vit-base-patch32)	Embeds both images and captions into a shared semantic space.
FAISS (Facebook AI Similarity Search)	Enables fast nearest-neighbor similarity search.

The system learns user preferences from liked images and recommends visually and semantically similar alternatives.

Features

Automatic caption generation for all images

Combined image-text embeddings for richer semantic similarity

Cosine similarity search via FAISS

Preference aggregation from multiple liked images

Installation
pip install torch torchvision pillow numpy faiss-cpu faiss-gpu tqdm transformers


For CUDA-compatible installations:

pip install faiss-gpu

Directory Structure
.
├── images/               # Input images
├── faiss_index.bin       # Stored FAISS vector index (generated)
├── meta.json             # Captions and metadata
└── recommender.py        # Main code file

Usage in Console

Step 1: Create Environments
python -m venv venv

Step 2: Activate Environments
source venv/bin/activate

Step 3: Create **images** folder inside project folder with desired datasets.

Step 4: install all libraries
pip install -r requirements1.txt

Step 5: Build the Index
python image_caption1.py --build

Step 6: Reccomend images
python image_caption1.py --recommend img1.jpg img2.jpg img3.jpg


This command will:

Load each image from the images/ directory

Generate captions using BLIP

Compute embeddings using CLIP

Store vectors inside the FAISS index

Generate Recommendations
python recommender.py --recommend img1.jpg img2.jpg --topk 5

Using in Streamlit

Step 1: Create Environments
python -m venv venv

Step 2: Activate Environments
source venv/bin/activate

Step 3: Create **images** folder inside environmenets with desired datasets

Step 4: install all libraries
pip install -r requirements1.txt

Step 5: Run streamlit
streamlit run streamlit_rec.py

Step 6: Inside streamlit build index

Step 7: Recommend images

Example output:

Recommendations:
dog_beach.jpg (score=0.8123) caption: a dog running on the shore
sunset_hike.png (score=0.7941) caption: a group walking on a hill at sunset

System Logic
Stage	Operation
1	Load image and generate text caption (BLIP)
2	Compute image and text embeddings (CLIP)
3	Combine embeddings (default: 70% image, 30% text)
4	Normalize and store in FAISS index
5	For recommendations, average vectors from liked items and retrieve nearest neighbors



