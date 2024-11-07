import os
from pathlib import Path
import sys
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from os.path import dirname, realpath, join
C_DIR = dirname(realpath(__file__))
ML_DIR = dirname(C_DIR)


from facedb import FaceDB


def insert_to_db(root_folder, db):
    for person_folder in os.listdir(root_folder):
        person_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_path):
            imgs = []
            names = []
            for image_file in os.listdir(person_path):
                if image_file.endswith('.png'):
                    image_path = os.path.join(person_path, image_file)
                    imgs.append(image_path)
                    names.append(Path(image_file).stem)

            ids, failed_indexes = db.add_many(imgs=imgs, names=names, check_similar=False)

            print(
                f"Failed indexes: {failed_indexes}\n"
                f"IDs: {ids}"
            )
            print(db.all(include=["name"]))

            # Add your assertions here if needed
            # self.assertEqual(len(failed_indexes), 0)

# Example of how to use insert_to_db function
if __name__ == "__main__":
    # Assuming you have already initialized your FaceDB instance
    db = FaceDB(
        path=join(ML_DIR,"facedata"),
        metric="cosine", 
        database_backend="chromadb",
        embedding_dim=512,
        module="arcface"
    )
    
    test_folder = "/home/anhlbt/Downloads/"
    root_folder = "/media/anhlbt/SSD2/workspace/computer_vision/FaceRecognition_2019/faceidsys/datasets/aligned/atvn_emb/112x112/"
    image = "1717060290.6706803.jpg" # "front-cam-1717152847.432085-fzzc2b.jpg" # "1717060290.7084718.jpg"
    img_p = join(test_folder, image)
    # insert_to_db(root_folder, db)
    db.count()
    result = db.recognize(img=img_p, include=["name", "embedding"])
    # print(db.all(include=["embedding"]).df) # [f.embedding for f in db.all(include=["embedding"])]
    # pd.DataFrame([(f.name, f.embedding) for f in db.all(include=["embedding"])])
    print("....")