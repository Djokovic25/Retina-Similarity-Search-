from dataset import IDRiDDataset

train_dataset=IDRiDDataset(image_dir="data/Train",csv_path="data/annotations_train.csv")
print("train size",len(train_dataset))
# image, caption, grade, name = train_dataset[0]
# print("Image:", name)
# print("Grade:", grade)
# print("Caption:", caption[:100])
sample = train_dataset[0]

print(sample["image_name"])
print(sample["grade"])
print(sample["caption"][:100])
