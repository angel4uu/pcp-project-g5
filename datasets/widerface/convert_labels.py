import os
import cv2
import shutil

def convert_annotations(txt_file, img_dir, label_dir):
    print(f"\n📂 Reading annotation file: {txt_file}")
    with open(txt_file, "r") as f:
        lines = f.readlines()

    i = 0
    total = 0
    os.makedirs(label_dir, exist_ok=True)

    while i < len(lines):
        filename = lines[i].strip()

        # Check if filename line is correct
        if i + 1 >= len(lines):
            print(f"⚠️ Unexpected end of file at line {i}.")
            break

        try:
            num_faces = int(lines[i + 1].strip())
        except ValueError:
            print(f"⚠️ Skipping malformed line at index {i}: {lines[i + 1]}")
            i += 1
            continue

        image_path = os.path.join(img_dir, os.path.basename(filename))
        if not os.path.exists(image_path):
            print(f"🚫 Image not found: {image_path}")
            i += 2 + num_faces
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Couldn't read image: {image_path}")
            i += 2 + num_faces
            continue

        h, w = img.shape[:2]
        label_lines = []

        for j in range(num_faces):
            try:
                x, y, bw, bh, *_ = map(int, lines[i + 2 + j].strip().split())
                if bw <= 0 or bh <= 0:
                    continue
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")
            except Exception as e:
                print(f"⚠️ Error parsing face bbox on line {i + 2 + j}: {e}")
                continue

        # Write label file only if it has data
        if label_lines:
            label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(filename))[0] + ".txt")
            with open(label_path, "w") as out:
                out.write("\n".join(label_lines))
            total += 1
        else:
            print(f"🟡 Skipped empty: {filename}")

        i += 2 + num_faces

    print(f"\n🎯 Total label files generated: {total}")

def convert_all(base_path="."):
    print("🔁 Converting WIDER FACE annotations to YOLO format...")
    sets = {
        "train": {
            "txt": os.path.join(base_path, "wider_face_split/wider_face_train_bbx_gt.txt"),
            "img": os.path.join(base_path, "images/train"),
            "lbl": os.path.join(base_path, "labels/train")
        },
        "val": {
            "txt": os.path.join(base_path, "wider_face_split/wider_face_val_bbx_gt.txt"),
            "img": os.path.join(base_path, "images/val"),
            "lbl": os.path.join(base_path, "labels/val")
        }
    }

    for split, paths in sets.items():
        os.makedirs(paths["lbl"], exist_ok=True)
        print(f"➡️  {split} set")
        convert_annotations(paths["txt"], paths["img"], paths["lbl"])

    print("✅ All annotations converted.")

if __name__ == "__main__":
    convert_all()
