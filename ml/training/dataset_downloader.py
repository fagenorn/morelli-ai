import os
import requests
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing

human_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "laion-art.parquet")
human_skip = 0
ai_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "prompts.csv")
ai_skip = 0
output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "digital-art")
workers = 16

def download_image(index, url, is_ai=False):
    print(f"Mock download {index}. {url}")
    return

    response = requests.get(url)
    output_file = os.path.join(output, "ai" if is_ai else "human", f"{index}.jpg")
    open(output_file, "wb").write(response.content)

    if verify_image(output_file):
        print(f"Downloaded {index}")
    else:
        print(f"Failed to download {index}")

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
                if ai_skip > index:
                    continue

                executor.submit(download_image, index, row["image_uri"], True)

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=download_human_images, name="download_human_images")
    p2 = multiprocessing.Process(target=download_ai_images, name="download_ai_images")

    p1.start()
    p2.start()

    p1.join()
    p2.join()