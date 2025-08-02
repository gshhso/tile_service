import rioxarray as rxr
import xarray as xr
import numpy as np
import rasterio
import geopandas as gpd
import mercantile
import pickle

from geopandas.sindex import SpatialIndex as SIndex
from shapely.geometry import box
from pathlib import Path
from typing import cast
from rioxarray import merge
from functools import lru_cache

Pathlike = str | Path
BBOX = tuple[float, float, float, float]
XYZ = tuple[int, int, int]


def get_slice(bbox) -> dict:
    return {"x": slice(bbox[0], bbox[2]), "y": slice(bbox[3], bbox[1])}


def build_rtree_index(
    image_list: list[Pathlike],
    save_root: Pathlike,
    luotu_name: str = "luotu.geojson",
    index_name: str = "index.pkl",
):
    save_root = Path(save_root)
    bounds = []
    for image in image_list:
        with rasterio.open(image) as src:
            bounds.append(box(*src.bounds))
    luotu = gpd.GeoDataFrame({"path": image_list, "geometry": bounds}, crs="EPSG:4326")  # type: ignore
    
    # 保存空间索引
    with open(save_root / index_name, "wb") as f:
        pickle.dump(luotu.sindex, f)
    
    # 保存GeoDataFrame，直接使用文件路径
    luotu.to_file(save_root / luotu_name)


def load_rtree_index(
    save_path: Pathlike, luotu_path: Pathlike
) -> tuple[gpd.GeoDataFrame, SIndex]:
    # 加载空间索引
    with open(save_path, "rb") as f:
        sindex = pickle.load(f)
    
    # 加载GeoDataFrame，直接使用文件路径
    gdf = gpd.read_file(luotu_path)
    return gdf, sindex


def filter_intersect_image(
    luotu: gpd.GeoDataFrame, sindex: SIndex, bbox: BBOX
) -> list[Pathlike]:
    query = box(*bbox)
    filtered_luotu = luotu.iloc[luotu.sindex.query(query, predicate="intersects")]
    filtered_luotu = cast(gpd.GeoSeries, filtered_luotu)
    return filtered_luotu["path"].to_list()


@lru_cache(maxsize=None)
def load_dataarray(path: Pathlike) -> xr.DataArray:
    return cast(xr.DataArray, rxr.open_rasterio(path))

def load_dataarrays(root: Pathlike) -> xr.DataArray:
    image_list = list(Path(root).rglob("**/*.tif"))
    dataarrays = [load_dataarray(image) for image in image_list]
    return merge.merge_arrays(dataarrays)

def get_tiles(dataarray: xr.DataArray, bbox: BBOX) -> np.ndarray:
    return dataarray.sel(**get_slice(bbox)).to_numpy()

def convert_xyz_to_bbox(xyz: XYZ) -> BBOX:
    return mercantile.bounds(xyz[0], xyz[1], xyz[2])
