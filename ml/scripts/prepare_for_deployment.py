import argparse
import asyncio
from dataclasses import dataclass
from typing import Optional
import json
import joblib

import aiohttp
import torch


@dataclass
class SpeciesInformation:
    scientific_name: str
    gbif_id: int
    common_names: dict[str, list[str]]

    @classmethod
    def from_requests(
        cls,
        scientific_name,
        key,
        species_response,
        synonyms_response,
        common_names_response,
    ) -> Optional["SpeciesInformation"]:
        if not key:
            return None

        key = int(key)

        english_common_name = species_response.get("vernacularName")

        synonyms = [
            x.get("vernacularName") for x in synonyms_response.get("results", [])
        ]

        common_names = [
            (x.get("language"), x.get("vernacularName"))
            for x in common_names_response.get("results", [])
        ]

        all_common_names = {}
        all_common_names["eng"] = (
            {english_common_name} if english_common_name else set()
        )
        all_common_names["eng"].update({x for x in synonyms if x})

        for language, name in common_names:
            if not language or not name:
                continue
            all_common_names[language] = all_common_names.get(language, set())
            all_common_names[language].add(name)

        print(scientific_name, key, all_common_names)

        return cls(
            scientific_name=scientific_name,
            gbif_id=key,
            common_names={k: list(v) for k, v in all_common_names.items()},
        )

    @property
    def json(self):
        return {
            "scientific_name": self.scientific_name,
            "gbif_id": self.gbif_id,
            "common_names": self.common_names,
        }


async def get_species_infos(names: list[str]) -> dict[str, SpeciesInformation]:
    async with aiohttp.ClientSession() as session:
        orig_names = [name.replace(" ", "_").lower() for name in names]
        names = [name.replace("_", " ").capitalize() for name in names]
        key_response_tasks = [
            session.get(f"http://api.gbif.org/v1/species/match?name={name}")
            for name in names
        ]
        key_responses = await asyncio.gather(*key_response_tasks)
        keys = [(await response.json()).get("usageKey") for response in key_responses]

        species_response_tasks = [
            session.get(f"http://api.gbif.org/v1/species/{key}") for key in keys
        ]
        synonyms_response_tasks = [
            session.get(f"http://api.gbif.org/v1/species/{key}/synonyms")
            for key in keys
        ]
        common_names_response_tasks = [
            session.get(f"http://api.gbif.org/v1/species/{key}/vernacularNames")
            for key in keys
        ]

        species_responses = await asyncio.gather(*species_response_tasks)
        synonyms_responses = await asyncio.gather(*synonyms_response_tasks)
        common_names_responses = await asyncio.gather(*common_names_response_tasks)

        # await response.jsons

        species_jsons = [await response.json() for response in species_responses]
        synonyms_jsons = [await response.json() for response in synonyms_responses]
        common_names_jsons = [
            await response.json() for response in common_names_responses
        ]

        species_infos = [
            SpeciesInformation.from_requests(
                name,
                key,
                species_json,
                synonyms_json,
                common_names_json,
            )
            for name, key, species_json, synonyms_json, common_names_json in zip(
                names, keys, species_jsons, synonyms_jsons, common_names_jsons
            )
        ]

        return {name: species_info for name, species_info in zip(orig_names, species_infos)}


def main():

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--path", type=str, required=True)

    args = argparser.parse_args()

    path = args.path

    crate = joblib.load(f"{path}/model.joblib")

    model = crate["model"]
    labels = crate["labels"]
    transform = crate["transform"]

    size = transform.resize_size[0]

    torch.onnx.export(
        model,
        torch.randn(1, 3, size, size),
        f"{path}/model.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"]
    )

    species_infos = asyncio.run(get_species_infos(labels))

    with open(f"{path}/metadata.json", "w") as f:
        json.dump({
            "labels": labels,
            "infos": {k: v.json for k,v in species_infos.items()},
            "size": size,
            "mean": transform.mean,
            "std": transform.std
            },
            f,
            ensure_ascii=False
        )




if __name__ == "__main__":
    main()
