import cdsapi
import zipfile
from pathlib import Path
from typing import List


class CDSRequestBuilder:
    def __init__(self, dataset_name: str):
        self.dataset = dataset_name
        self._params = {
            "format": "grib",
            "product_type": "monthly_mean",
        }

    def set_variable(self, variables: List[str]):
        self._params["variable"] = variables
        return self

    def set_pressure_level(self, level: str):
        self._params["pressure_level"] = level
        return self

    def set_model_level(self, model_level: str):
        self._params["model_level"] = model_level
        return self

    def set_step(self, step: str):
        self._params["step"] = step
        return self

    def set_data_format(self, data_format: str):
        self._params["data_format"] = data_format
        return self

    def set_area(self, north: float, west: float, south: float, east: float):
        self._params["area"] = [north, west, south, east]
        return self

    def set_date_range(self, start: str, end: str):
        self._params["date"] = [f"{start}/{end}"]
        return self

    def set_exact_dates(self, years: List[str], months: List[str] = None, days: List[str] = None):
        self._params["year"] = years
        if months:
            self._params["month"] = months
        if days:
            self._params["day"] = days
        return self

    def set_time(self, time: str = "00:00"):
        self._params["time"] = time
        return self

    def build(self):
        return {
            "dataset": self.dataset,
            "params": self._params
        }


class CDSClient:
    def __init__(self):
        self.client = cdsapi.Client()

    def request_and_download(self, request: dict, output_zip_path: str, extract_to: str = None):
        dataset = request["dataset"]
        params = request["params"]

        print(f"Sending request to CDS API for dataset: {dataset}")
        self.client.retrieve(dataset, params, output_zip_path)

        print(f"Download completed: {output_zip_path}")

        if extract_to:
            with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                print(f"Extracted to: {extract_to}")
            return Path(extract_to).glob("*.grib")

        return output_zip_path
