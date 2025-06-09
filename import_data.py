from cds import *

DATA_SOURCE = "cams-global-ghg-reanalysis-egg4-monthly"
DATA_FILE_ZIP = "bucuresti_co2_temp_2022_2023.zip"
DATA_FILE = "bucuresti_co2_temp_2022_2023.grib"

if __name__ == "__main__":
    builder = CDSRequestBuilder(DATA_SOURCE)
    request = (
        builder
            .set_variable(["2m_temperature", "carbon_dioxide", "methane"])
            .set_pressure_level("1000")
            .set_area(44.6, 25.9, 44.3, 26.3)
            .set_date_range("2003-12-31", "2020-12-31")
            .set_data_format("grib")
            .set_step("0")
            .set_time("12:00")
            .build()
    )

    client = CDSClient()
    client.request_and_download(
        request=request,
        output_zip_path=DATA_FILE_ZIP,
        extract_to=DATA_FILE
    )
