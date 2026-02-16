import os
import torch 
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from dataset import IDRiDDataset


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

def generate_embeddings(image_dir,csv_path,output_path):
    dataset = IDRiDDataset(image_dir=image_dir, csv_path=csv_path)
    
    model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(DEVICE)
    model.eval()


    image_embeddings = []
    text_embeddings = []
    grades = []
    image_names = []

    with torch.no_grad():
        for sample in tqdm(dataset):

            image = sample["image"]
            caption = sample["caption"]
            grade = sample["grade"]
            name = sample["image_name"]

            inputs = processor(
                text=[caption],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)

            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds

            # Normalize (IMPORTANT for similarity)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

            image_embeddings.append(img_emb.cpu().numpy())
            text_embeddings.append(txt_emb.cpu().numpy())
            grades.append(grade)
            image_names.append(name)

    image_embeddings = np.vstack(image_embeddings)
    text_embeddings = np.vstack(text_embeddings)

    os.makedirs(output_path, exist_ok=True)

    np.save(os.path.join(output_path, "image_embeddings.npy"), image_embeddings)
    np.save(os.path.join(output_path, "text_embeddings.npy"), text_embeddings)
    np.save(os.path.join(output_path, "grades.npy"), np.array(grades))

    np.save(os.path.join(output_path, "image_names.npy"), np.array(image_names))

    print("Embeddings saved successfully.")


if __name__ == "__main__":
    generate_embeddings(
        image_dir="data/Train/images",
        csv_path="data/annotations_train.csv",
        output_path="embeddings"
    )