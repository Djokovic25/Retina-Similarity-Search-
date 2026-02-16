import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class IDRiDDataset(Dataset):
    def __init__(self,image_dir,csv_path,transform=None):
        self.image_dir=image_dir
        self.df=pd.read_csv(csv_path)
        self.transform=transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, row["Image name"])
        image = Image.open(image_path).convert("RGB")
        caption=row["caption"]
        grade=int(row["Retinopathy grade"])

        if self.transform:
            image=self.transform(image)

        return {
    "image": image,
    "caption": caption,
    "grade": grade,
    "image_name": row["Image name"]
}
