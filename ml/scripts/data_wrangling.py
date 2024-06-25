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

import matplotlib.pyplot as plt
import geopandas as gpd

import aiohttp
import aiofiles
import asyncio
import os
from tqdm.asyncio import tqdm


@dataclass
class ObservationMedia:
    gbifID: int
    identifier: str
    species: str

    @property
    def url(self):
        return self.identifier

    @property
    def path(self):
        url_hash = hashlib.md5(self.url.encode()).hexdigest()[:8]
        return f"{self.species}/{self.gbifID}_{url_hash}.jpg"


PATH = "data/"
IMG_PATH = "/run/media/hdd/arttu/inaturalist_images"


async def fetch(session, media, progress, base_dir, semaphore):
    async with semaphore:
        try:
            async with session.get(media.url) as response:
                if response.status == 200:

                    save_path = base_dir / media.path
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    data = await response.read()
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    img.thumbnail((512, 512))
                    img.save(save_path, "JPEG")

                progress.update(1)
        except Exception as e:
            print(f"Failed to download {media.url}: {e}")


async def download_all(observation_medias):

    base_dir = Path(IMG_PATH)
    base_dir.mkdir(parents=True, exist_ok=True)

    media_dclasses = observation_medias[["gbifID", "identifier", "species"]].apply(
        lambda x: ObservationMedia(**x.to_dict()), axis=1
    )

    print(f"{len(media_dclasses)} files to download")

    existing_files = set([str(f) for f in Path(IMG_PATH).rglob("*")])

    not_downloaded = [
        media for media in media_dclasses if not str(base_dir / media.path) in existing_files
    ]

    print(f"Downloading {len(not_downloaded)} files")

    semaphore = asyncio.Semaphore(12)

    async with aiohttp.ClientSession() as session:
        progress = tqdm(total=len(not_downloaded), desc="Downloading")
        tasks = [fetch(session, media, progress, base_dir, semaphore) for media in not_downloaded]
        await asyncio.gather(*tasks)
        progress.close()


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
    #europe = world[world["continent"] == "Europe"]

    europe = world[world["name"].isin(["Finland"])]

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
    print(f"{len(observations)} media records available in total")
    observations.dropna(subset=["identifier", "species"], inplace=True)

    european_species = choose_european(observations)["species"].unique()

    observations = observations[observations["species"].isin(european_species)]

    print(f"left with {len(observations)} media records to download")

    # number of observations per species
    #species_counts = observations["species"].value_counts()

    #thingy = species_counts.sort_values(ascending=False)

    # visualize
    #plt.bar(x=range(len(thingy.index)), height=thingy.values, log=True)
    #plt.show()

    # choose a random sample of 1000 rows. Download the images from the URLs in the "identifier" column,
    # and add the bytes to a column called "image_bytes"

    observations = observations.sample(100000, random_state=43)

    asyncio.run(download_all(observations))

