from __future__ import annotations

import functools
from importlib.metadata import files
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Any

import pyearthtools.data
from pyearthtools.data.archive import register_archive
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.transforms.variables import Drop
from pyearthtools.data.transforms.values import SetMissingToNaN


# This dictionary tells pyearthtools which variables have missing values and what those values are.
varname_val_map = {
    "total_cloud_cover": -999.0,
    "low_cloud_cover": -999.0,
    "mid_cloud_cover": -999.0,
    "high_cloud_cover": -999.0,
}


@functools.lru_cache()
def cached_iterdir(path: Path) -> list[Path]:
    """Run iterdir but cached"""
    return list(path.iterdir())


@functools.lru_cache()
def cached_exists(path: Path) -> bool:
    """Run exits but cached"""
    return path.exists()


# TODO:
# - In the future it would be good to add the possibility to have this preprocessing step as part of a pipeline of other preprocessing steps.
# - Other similarly process heavy steps could be added to the pipeline, such as calculation of climatologies, or other derived variables.


# Helper function to preprocess and save NetCDF files as Zarr stores
# @delayed Experimenting with delayed to see if it helps with performance
def preprocess_and_save(file_path, date_range, zarr_output_dir):  # TODO Needs to be implemented correctly
    """
    Open a NetCDF file, preprocess it, and save as a Zarr store.

    Steps performed:
        - Opens the NetCDF file as an xarray Dataset.
        - Drops the 'input_station_id' variable if present (to avoid object dtype issues).
        - Assigns a 'station_id' coordinate from the dataset attributes or filename.
        - Reindexes the time dimension to a common hourly range.
        - Saves the processed Dataset to a Zarr store in the specified output directory.

    Args:
        file_path (str or Path): Path to the NetCDF file.
        date_range (tuple of str): (start, end) date strings for reindexing the time dimension.
        zarr_output_dir (str or Path): Directory where the Zarr store will be saved.

    Returns:
        str: Path to the saved Zarr store.
    """
    try:
        print(f"Preprocessing {file_path} -> {zarr_output_dir}")
        with xr.open_dataset(file_path) as ds:
            if "input_station_id" in ds:
                ds = ds.drop_vars("input_station_id")

            station_id = ds.attrs.get("station_id", file_path.stem)
            ds = ds.assign_coords(station_id=station_id)

            target_time = pd.date_range(date_range[0], date_range[1], freq="h")
            ds = ds.reindex(time=target_time)

            out_path = Path(zarr_output_dir) / f"{file_path.stem}.zarr"
            print(f"Saving to Zarr: {out_path}")
            ds.to_zarr(str(out_path), mode="w")
            print(f"Saved Zarr: {out_path}")
            return str(out_path)
    except Exception as e:
        print(f"Failed to preprocess {file_path}: {e}")
        raise


@register_archive("hadisd", sample_kwargs=dict(station="010010-99999"))
class HadISDIndex(ArchiveIndex):
    # def load(self, *args, **kwargs):
    #     print("RUBBISH LOAD METHOD CALLED! If you see this, your override works.")
    #     return None
    # """HadISD Dataset Index"""

    @property
    def _desc_(self):
        return {
            "singleline": "HadISD Dataset",
            "range": "1931-2024",
            "Documentation": "https://www.metoffice.gov.uk/hadobs/hadisd/",
        }

    def __init__(
        self,
        station: str | list[str] | None = None,  # Allow single station, multiple stations, or None
        variables: list[str] | str | None = None,
        *,
        transforms: Transform | TransformCollection | None = None,  # Ensure this is keyword-only
    ):
        """
        Setup HadISD Indexer

        Args:
            station (str): Station ID to retrieve data for.
            transforms (optional): Base transforms to apply.
        """
        self.station = [station] if isinstance(station, str) else station
        self.variables = [variables] if isinstance(variables, str) else variables

        # Define the base transforms
        base_transform = TransformCollection()
        base_transform += Drop("reporting_stats")

        # Add a transform to select variables (if variables are provided)
        if variables:
            base_transform += pyearthtools.data.transforms.variables.Select(self.variables)
            print(f"Variables selected: {self.variables}")

        # Possibly remove this transform if not needed
        base_transform += SetMissingToNaN(varname_val_map)

        if transforms is None:
            super().__init__(
                transforms=base_transform + TransformCollection(),
            )
        else:
            super().__init__(
                transforms=base_transform + transforms,
            )

        self.record_initialisation()

    def get_all_station_ids(self, zarr_store: Path | str = None) -> list[str]:
        """
        Retrieve all station IDs from the Zarr store's 'station' dimension.

        Args:
            zarr_store (Path | str, optional): Path to the Zarr store. Defaults to HADISD_HOME/zarr.zarr.

        Returns:
            list[str]: A list of all station IDs.
        """
        HADISD_HOME = self.ROOT_DIRECTORIES["hadisd"]
        if zarr_store is None:
            zarr_store = Path(HADISD_HOME) / "zarr.zarr"
        else:
            zarr_store = Path(zarr_store)

        if not cached_exists(zarr_store):
            raise DataNotFoundError(f"Zarr store does not exist: {zarr_store}")

        # Open the Zarr store lazily and get the station dimension
        ds = xr.open_zarr(zarr_store, chunks={}, consolidated=True)
        return [str(s) for s in ds.station.values]

    def filesystem(self, *args, **kwargs) -> dict[str, Path]:
        """
        Return the path to the single Zarr store for all stations.

        Returns:
            dict[str, Path]: Dictionary mapping 'all' to the Zarr store path.
        """
        HADISD_HOME = self.ROOT_DIRECTORIES["hadisd"]
        zarr_store = Path(HADISD_HOME) / "zarr.zarr"  # Update with your actual filename if needed
        if not cached_exists(zarr_store):
            raise DataNotFoundError(f"Zarr store does not exist: {zarr_store}")
        return zarr_store
    
    # TODO: Station selection should be handled as a transform, similar to variable selection.
    # This will allow for flexible, pipeline-based selection and lazy loading.

    def load(
        self,
        zarr_store: Path | str = None,
        station: str | list[str] | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Load data from the single Zarr store, supporting 'all' as a station argument.

        Args:
            zarr_store (Path | str, optional): Path to the Zarr store. Defaults to HADISD_HOME/zarr.zarr.
            station (str | list[str] | None, optional): Station(s) to select. If 'all', loads all stations.
            **kwargs: Additional arguments passed to xarray.open_zarr.

        Returns:
            xr.Dataset: The loaded dataset (lazily loaded, dask-backed).
        """
        HADISD_HOME = self.ROOT_DIRECTORIES["hadisd"]
        if zarr_store is None:
            zarr_store = Path(HADISD_HOME) / "zarr.zarr"
        else:
            zarr_store = Path(zarr_store)
        if not cached_exists(zarr_store):
            raise DataNotFoundError(f"Zarr store does not exist: {zarr_store}")
        # Handle 'all' station selection
        if station == "all" or (isinstance(station, list) and "all" in station):
            station = self.get_all_station_ids(zarr_store)
        ds = xr.open_zarr(zarr_store, consolidated=True, **kwargs)
        if station is not None:
            ds = ds.sel(station=station)
        return ds

    @property
    def _import(self):
        """module to import for to load this step in a Pipeline"""
        return "pyearthtools.tutorial"
