"""
This script provides the necessary methods to make requests to the USDA API in order to get
nutritional facts of food items.
"""
import os
import requests

from typing import Dict, Final
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()

class Food(BaseModel):
    fdcId: int
    carb: float
    protein: float
    fat: float
    calories: float

class USDAClient:
    """Thin wrapper around USDA FoodData Central v1 API."""
    BASE_SEARCH: Final[str] = str(os.getenv("USDA_API_BASESEARCH", "https://api.nal.usda.gov/fdc/v1/foods/search"))
    BASE_FOOD:   Final[str] = str(os.getenv("USDA_API_BASEFOOD", "https://api.nal.usda.gov/fdc/v1/food"))

    WANTED  = {
        "203": "protein",
        "204": "fat",     
        "205": "carb",    
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: Dict[int, Dict[str, float]] = {}  # fdcId -> macros dict
        self.session = requests.Session()

    def _best_fdc_id(self, query: str) -> int | None:
        """Return top Foundation/SR record fdcId or None."""
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": 5,
            "requireAllWords": False,
            "dataType": ["Foundation"],
        }
        r = self.session.get(self.BASE_SEARCH, params=params, timeout=5)
        r.raise_for_status()
        foods = r.json()["foods"]
        if not foods:
            return None

        # pick first that contains 'raw' or 'whole' or shortes description
        for f in foods:
            desc = f["description"].lower()
            if "raw" in desc or "whole" in desc:
                best = f
                break
        else:
            best = min(foods, key=lambda f: len(f["description"]))
        return best["fdcId"]

    def _macros_by_id(self, fdc_id: int) -> Dict[str, float]:
        """Hit /food/{id}?nutrients=...  (with simple cache)."""
        if fdc_id in self._cache:
            return self._cache[fdc_id]
        
        def fetch(filtered: bool):
            params = {"api_key": self.api_key}
            if filtered:
                params["nutrients"] = ",".join(self.WANTED.keys())
            r = self.session.get(f"{self.BASE_FOOD}/{fdc_id}", params=params, timeout=30)
            if r.status_code == 404 and filtered:
                return fetch(filtered=False)  
            if r.status_code == 404:          
                return None      
                
            r.raise_for_status()
            return r.json()
        
        data = fetch(filtered=True)
        
        wanted = {k: 0.0 for k in self.WANTED.values()}   
        for n in data.get("foodNutrients", []):
            num   = n.get("nutrientNumber") or n.get("nutrient", {}).get("number")
            value = n.get("amount", n.get("value"))
            if num in self.WANTED and value is not None:
                wanted[self.WANTED[num]] = float(value)

        protein = wanted.get("protein", 0.0)
        fat     = wanted.get("fat", 0.0)
        carb    = wanted.get("carb", 0.0)
        kcal    = 4 * protein + 9 * fat + 4 * carb
        wanted["kcal"] = round(kcal, 2)

        self._cache[fdc_id] = wanted
        return wanted

    def get_nutritional_facts(self, label: str) -> Dict[str, float]:
        """
        label → kcal, protein, fat, carb  (per 100 g edible portion).
        Returns zeroes if nothing found.
        """
        fdc_id = self._best_fdc_id(label)
        if fdc_id is None:
            return {v: 0.0 for v in self.WANTED.values()}
        return self._macros_by_id(fdc_id)

if __name__ == "__main__":
    api_key = os.getenv("USDA_API_KEY")
    client = USDAClient(api_key)
    print(client.get_nutritional_facts("peanut"))

