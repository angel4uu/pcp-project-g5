#!/usr/bin/env bash
set -e

echo "🔧 Setting up WIDER FACE dataset for YOLO training..."

# Remove previous dataset folders if needed
rm -rf images labels

# Step 1: Create image and label directories
mkdir -p images/train images/val
mkdir -p labels/train labels/val

# Step 2: Download dataset zips
echo "📥 Downloading dataset..."
wget -nc https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_train.zip
wget -nc https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip
wget -nc https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip

# Step 3: Unzip datasets
echo "📂 Unzipping..."
unzip -q WIDER_train.zip
unzip -q WIDER_val.zip
unzip -q wider_face_split.zip

# Step 4: Organize image files (flatten)
find WIDER_train/images -type f -name "*.jpg" -exec mv {} images/train/ \;
find WIDER_val/images -type f -name "*.jpg" -exec mv {} images/val/ \;

# Step 5: Move annotations
mv wider_face_split ./  # contains *.txt for annotation

echo "✅ Setup complete! Ready for annotation conversion."
