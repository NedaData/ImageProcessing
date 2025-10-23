import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
import faiss

IMAGES_DIR = Path('images')
INDEX_PATH = Path('faiss_index.bin')
META_PATH = Path('meta.json')

EMBED_DIM = 512
COMBINE_IMAGE_WEIGHT = 0.7
COMBINE_TEXT_WEIGHT = 0.3
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading modells ....')
BLIP_MODEL = "Salesforce/blip-image-captioning-large"
CLIP_MODEL = "openai/clip-vit-base-patch32"

blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(DEVICE)
blip_model.eval()


clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)

def load_image(path: Path, resize: Tuple[int, int]=None):
    img = Image.open(path).convert('RGB')
    if resize:
        img = img.resize(resize)
    return img


def caption_image(img: Image.Image):
    inputs = blip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = blip_model.generate(**inputs, max_length=30, num_beams=3)
    caption = blip_processor.decode(out_ids[0], skip_special_tokens=True)
    return caption


def image_embedding(img: Image.Image):
    inputs = clip_processor(images=img, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs)  # (1, d)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().reshape(-1)

def text_embedding(text: str):
    inputs = clip_processor(text=[text], return_tensors='pt', padding=True).to(DEVICE)
    with torch.no_grad():
        feat = clip_model.get_text_features(**inputs)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().reshape(-1)


def combine_embeddings(img_emb: np.ndarray, txt_emb: np.ndarray, w_img=COMBINE_IMAGE_WEIGHT, w_text=COMBINE_TEXT_WEIGHT):
    vec = w_img * img_emb + w_text * txt_emb
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec.astype(np.float32)

def build_index(images_dir: Path = IMAGES_DIR,
                index_path: Path = INDEX_PATH,
                meta_path: Path = META_PATH):
    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    if not image_files:
        raise RuntimeError(f"No images found in {images_dir.resolve()}")
    
    meta = {"files": [], "dim": EMBED_DIM, "combine_weights": [COMBINE_IMAGE_WEIGHT, COMBINE_TEXT_WEIGHT]}
    vectors = []


    print(f"Ingesting {len(image_files)} images from {images_dir} ...")
    for p in tqdm(image_files):
        try:
            img = load_image(p)
            caption = caption_image(img)
            img_emb = image_embedding(img)
            txt_emb = text_embedding(caption)
            vec = combine_embeddings(img_emb, txt_emb)
            vectors.append(vec)
            meta["files"].append({"filename": p.name, "path": str(p.resolve()), "caption": caption})
        except Exception as e:
            print(f"FAILED {p}: {e}")

    vectors = np.vstack(vectors).astype(np.float32)  # (N, d)
    # Ensure dimension consistency
    d = vectors.shape[1]
    print(f"Built embedding matrix: {vectors.shape}")

    # Use IndexFlatIP for inner-product (works as cosine similarity if vectors are normalized)
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    # Save index & metadata
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved FAISS index -> {index_path}, meta -> {meta_path}")


def load_index(index_path: Path=INDEX_PATH, meta_path: Path=META_PATH):
    if not index_path.exists() or not meta_path.exists():
        raise RuntimeError("Index or meta file not found. Run with --build first.")
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def build_recommendation(user_likes: List[str], top_k: int=3, index_path: Path=INDEX_PATH, meta_path: Path=META_PATH):
    index, meta = load_index(index_path, meta_path)
    files = meta["files"]
    filename_to_idx = {f["filename"]: idx for idx, f in enumerate(files)}
    like_idxs = []
    for fn in user_likes:
        if fn not in filename_to_idx:
            raise ValueError(f"Liked file not found in index: {fn}")
        like_idxs.append(filename_to_idx[fn])
    
    d = index.d
    liked_vecs = []
    for ii in like_idxs:
        vec = np.zeros(d, dtype='float32')
        index.reconstruct(ii, vec)
        liked_vecs.append(vec.reshape(1, -1))
    
    pref = np.vstack(liked_vecs).mean(axis=0, keepdims=True)
    pref = pref / (np.linalg.norm(pref, axis=1, keepdims=True) + 1e-10)

    q = pref.astype(np.float32)
    k_search = top_k + len(like_idxs) + 5
    D, I = index.search(q, k_search)  # D: similarities, I: indices

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx in like_idxs:
            continue
        if idx < 0:
            continue
        f = files[idx]
        results.append({"filename": f["filename"], "path": f["path"], "caption": f["caption"], "score": float(score)})
        if len(results) >= top_k:
            break

    return results


def main():
    global INDEX_PATH, META_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build Faiss from image folder")
    parser.add_argument("--recommend", nargs="+", help="Provide liked filenames: --recommend img1.jpg, img2.jpg, img3.jpg")
    parser.add_argument("--topk", type=int, default=3, help="Top k recommendations")
    parser.add_argument("--index", type=str, default=str(INDEX_PATH), help="FAISS index path")
    parser.add_argument("--meta", type=str, default=str(META_PATH), help="META JSON Path")
    args = parser.parse_args()

    
    INDEX_PATH = Path(args.index)
    META_PATH = Path(args.meta)

    if args.build:
        build_index()
        return 
    
    if args.recommend:
        recs = build_recommendation(args.recommend, top_k=args.topk)
        print("Recommendations: ")
        
        for r in recs:
            print(f"{r['filename']}  (score={r['score']:.4f})  caption: {r['caption']}")
        return

    parser.print_help()

if __name__ == "__main__":
    main()





