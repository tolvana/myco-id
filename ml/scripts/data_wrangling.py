import os
import hashlib
from dataclasses import dataclass
import io
import requests
import concurrent.futures
from pathlib import Path
from threading import Lock

from PIL import Image
import pandas as pd
from tqdm import tqdm

import geopandas as gpd


@dataclass
class ObservationMedia:
    gbifID: int
    identifier: str
    species: str

    @property
    def url(self):
        return self.identifier


PATH = "data/"
IMG_PATH = "/run/media/hdd/arttu/inaturalist_images"


def download_file(url, save_path, lock, progress_bar):
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors

    data = requests.get(url).content
    img = Image.open(io.BytesIO(data))
    img.thumbnail((512, 512))
    img.save(save_path, "JPEG")
    with lock:
        progress_bar.update(1)


def download_files_in_parallel(observation_medias, max_workers=5):
    base_dir = Path(IMG_PATH)
    base_dir.mkdir(parents=True, exist_ok=True)

    lock = Lock()
    progress_bar = tqdm(total=len(observation_medias), desc="Downloading", unit="file")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for obs_media in observation_medias:
            save_dir = base_dir / obs_media.species
            save_dir.mkdir(parents=True, exist_ok=True)

            # hexdigest of the url to avoid collision
            url_hash = hashlib.md5(obs_media.url.encode()).hexdigest()[:8]

            save_path = save_dir / f"{obs_media.gbifID}_{url_hash}.jpg"
            futures.append(
                executor.submit(
                    download_file, obs_media.url, save_path, lock, progress_bar
                )
            )

        succ = []

        for obs_media, future in zip(
            observation_medias, concurrent.futures.as_completed(futures)
        ):
            try:
                future.result()  # We can handle exceptions here if needed
                # save the url to a file to keep track of the downloaded files
                succ.append(obs_media.url)

            except Exception as e:
                print(f"Error occurred: {e}")

        return succ


def choose_european(df):
    num_unique_species = df["species"].nunique()
    print(f"starting out with {len(df)} observations of {num_unique_species} species")

    # rough bounding box for Europe
    west_lon = -31.0
    east_lon = 60.0
    north_lat = 71.0
    south_lat = 34.0

    df = df[
        (df["decimalLongitude"] >= west_lon)
        & (df["decimalLongitude"] <= east_lon)
        & (df["decimalLatitude"] >= south_lat)
        & (df["decimalLatitude"] <= north_lat)
    ]

    # Load the world map
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    europe = world[world["continent"] == "Europe"]

    # Create a GeoDataFrame from the observations
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude)
    )

    # Perform a spatial join
    gdf = gpd.sjoin(gdf, europe, how="inner", op="intersects")

    # Drop the geometry column
    gdf.drop(columns=["geometry"], inplace=True)

    # Drop the index column
    gdf.drop(columns=["index_right"], inplace=True)
    num_unique_species = gdf["species"].nunique()

    print(
        f"After filtering, left with {len(gdf)} observations of {num_unique_species} species"
    )

    return gdf


if __name__ == "__main__":
    # Load the data
    columns_of_interest = [
        "gbifID",
        "species",
        "eventDate",
        "decimalLatitude",
        "decimalLongitude",
    ]

    observations = pd.read_csv(
        PATH + "occurrence.txt", sep="\t", usecols=columns_of_interest
    )

    print(f"loaded {len(observations)} observations")
    multimedia = pd.read_csv(
        PATH + "multimedia.txt",
        sep="\t",
        usecols=["gbifID", "type", "format", "identifier"],
    )
    print(f"loaded {len(multimedia)} multimedia records")

    multimedia = multimedia[multimedia["type"] == "StillImage"]
    multimedia = multimedia[
        (multimedia["format"] == "image/jpeg") | (multimedia["format"] == "image/png")
    ]

    # add multimedia column to observations

    observations = observations.merge(multimedia, on="gbifID", how="left")
    print(f"left with {len(observations)} media records to download")
    observations.dropna(subset=["identifier", "species"], inplace=True)

    observations = choose_european(observations)

    breakpoint()

    # choose a random sample of 1000 rows. Download the images from the URLs in the "identifier" column,
    # and add the bytes to a column called "image_bytes"

    subsample = observations.sample(50).reset_index(drop=True)

    # chunk the subsample into smaller parts
    chunk_size = 10

    chunks = [subsample.iloc[i : i + chunk_size] for i in range(0, len(subsample), chunk_size)]

    for i, chunk in enumerate(chunks):
        print(f"Downloading chunk {i+1} / {len(chunks)}")
        succ = download_files_in_parallel(
            chunk[["gbifID", "species", "identifier"]].apply(
                lambda x: ObservationMedia(**x.to_dict()), axis=1
            ),
            max_workers=10,
        )
        print(f"Successfully downloaded {len(succ)} / {len(chunk)}")
        # save successfully downloaded urls to a file
        with open("downloaded_urls.txt", "w") as f:
            for url in succ:
                f.write(url + "\n")

    breakpoint()
