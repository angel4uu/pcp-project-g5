#!/usr/bin/env bash

echo "🚀 Running full WIDER FACE dataset preparation..."

# Get absolute path of the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../datasets/widerface"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit 1

# Step 1: Download and organize
echo "🔧 Step 1: Downloading and organizing..."
bash "$SCRIPT_DIR/setup_widerface.sh"

# Step 2: Convert annotations
echo "🔄 Step 2: Converting annotations..."
python3 "$SCRIPT_DIR/convert_labels.py"

# Step 3: Cleanup
echo "🧹 Cleaning up annotation files and temp folders..."
rm -rf wider_face_split
rm -f WIDER_train.zip WIDER_val.zip wider_face_split.zip
rm -rf WIDER_train WIDER_val

echo "🎉 All done! Ready for YOLOv8 training."

