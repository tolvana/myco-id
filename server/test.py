from dataclasses import dataclass
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import json
import requests
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
import asyncio
import aiohttp
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

labels = json.loads(open("labels.json").read())

model = models.efficientnet_v2_s()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(labels))

transform = EfficientNet_V2_S_Weights.DEFAULT.transforms()
model.load_state_dict(
    torch.load("model_05_26.pth", map_location=torch.device("cpu"))["model"]
)
model.eval()

@dataclass
class SpeciesInformation:
    scientific_name: str
    gbif_id: int
    common_names: Dict[str, List[str]]

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

async def get_species_infos(names: List[str]) -> Dict[str, SpeciesInformation]:
    async with aiohttp.ClientSession() as session:
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

        return {name: species_info for name, species_info in zip(names, species_infos)}

# Singleton pattern to manage species_infos
class SpeciesInfoManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.species_infos = None
        return cls._instance

    async def initialize_species_infos(self, labels: List[str]):
        self.species_infos = await get_species_infos(labels)

    def get_species_info(self, name: str):
        return self.species_infos.get(name) if self.species_infos else None

species_info_manager = SpeciesInfoManager()

# Dependency to ensure species_infos is initialized and available
async def get_species_info_manager():
    if not species_info_manager.species_infos:
        await species_info_manager.initialize_species_infos(labels)
    return species_info_manager

class ImageURL(BaseModel):
    url: str

@app.post("/classify")
async def classify_image(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    species_info_manager: SpeciesInfoManager = Depends(get_species_info_manager)
    ):
    image = None

    if file:
        print(f"Received file: {file.filename}")
        image = Image.open(io.BytesIO(await file.read()))
    elif url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            print(f"Received image from URL: {url}")
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="No file or URL provided")


    if image:
        image = transform(image).unsqueeze(0)
        out = model(image)
        probs = torch.nn.functional.softmax(out, dim=1).flatten().tolist()
        processed_labels = [label.replace("_", " ").capitalize() for label in labels]
        results_list = sorted(
            zip(processed_labels, probs), key=lambda x: x[1], reverse=True
        )[:5]

        results_list = [(k, v) for k, v in results_list if v > 0.001]

        species_infos = species_info_manager.species_infos

        results = {k: {"probability": v, "info": species_infos.get(k).json if species_infos.get(k) else None} for k, v in results_list}

        return JSONResponse(content=results)

    raise HTTPException(status_code=400, detail="Invalid file")

if __name__ == "__main__":
    import uvicorn
    # reload
    uvicorn.run("test:app", host="0.0.0.0", port=5000, reload=True)

