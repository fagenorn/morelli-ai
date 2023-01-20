import os
import requests
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_images", help="Number of images to download", type=int)
parser.add_argument("--workers", help="Number of workers to use", type=int)
parser.add_argument("--human_only", help="Download only human images", action="store_true")
parser.add_argument("--ai_only", help="Download only AI images", action="store_true")
args = parser.parse_args()

human_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "laion-art.parquet")
ai_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "openprompts.csv")
output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "digital-art")
workers = args.workers if args.workers else 6
num_images = args.num_images if args.num_images else 100

def get_last_file(folder):
    files = os.listdir(folder)
    if files:
        last_file = max(files, key=lambda x: int(x.split(".")[0]))
        return int(last_file.split(".")[0])
    else:
        return 0

human_skip = get_last_file(os.path.join(output, "human"))
ai_skip = get_last_file(os.path.join(output, "ai"))

def download_image(index, url, is_ai=False):
    response = requests.get(url)
    output_file = os.path.join(output, "ai" if is_ai else "human", f"{str(index).zfill(8)}.jpg")
    open(output_file, "wb").write(response.content)

    if verify_image(output_file):
        print(f"Downloaded {url}")
    else:
        print(f"Failed to download {url}")

def verify_image(output_file):
    try:
        im = Image.open(output_file)
        im.verify()

        if not im.format in ['JPEG', 'PNG']:
            im.close()
            raise Exception("Not a valid image")
        
        return True
    except:
        os.remove(output_file)
        return False
       
def download_human_images():
    df = pd.read_parquet(human_dataset)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, row in df.iterrows():
            if index >= human_skip + num_images:
                break
            if human_skip > index:
                continue

            executor.submit(download_image, index, row["URL"], False)

def download_ai_images():
    batch_size = 100
    df = pd.read_csv(ai_dataset, chunksize=batch_size, usecols=['raw_data'])

    for i, chunk in enumerate(df):
        chunk = pd.json_normalize(chunk.raw_data.apply(json.loads))
        chunk['image_uri'] = chunk["raw_discord_data.image_uri"]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, row in chunk.iterrows():
                index = batch_size * i + index
                if index >= ai_skip + num_images:
                    break
                if ai_skip > index:
                    continue

                executor.submit(download_image, index, row["image_uri"], True)

if __name__ == "__main__": 
    if args.human_only:
        download_human_images()
    elif args.ai_only:
        download_ai_images()
    else:
        p1 = multiprocessing.Process(target=download_human_images, name="download_human_images")
        p2 = multiprocessing.Process(target=download_ai_images, name="download_ai_images")
        p1.start()
        p2.start()     
        p1.join()
        p2.join()