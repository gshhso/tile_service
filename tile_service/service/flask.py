from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
from flask import Flask, Response
from geopandas import GeoDataFrame
from geopandas.sindex import SpatialIndex
from PIL import Image

from ..tile import (
    Pathlike,
    build_rtree_index,
    convert_xyz_to_bbox,
    filter_intersect_image,
    get_tile,
    load_dataarray,
    load_rtree_index,
)


# 配置管理
class TileServiceConfig:
    """瓦片服务配置类"""

    def __init__(
        self,
        data_root: Pathlike,
        index_root: Pathlike | None = None,
        index_file: Pathlike = "index.pkl",
        luotu_file: Pathlike = "luotu.geojson",
        tile_size: int = 256,
        cache_size: int = 128,
        image_format: str = "PNG",
    ) -> None:
        """
        初始化瓦片服务配置

        Args:
                data_root: 数据根目录路径
                index_root: 索引文件根目录路径,若未提供则为data_root
                index_file: 空间索引文件路径,若未提供则尝试在data_root下寻找，若仍未找到则触发build index
                luotu_file: 落图文件路径,若未提供则尝试在data_root下寻找，若仍未找到则触发build index
                tile_size: 瓦片大小，默认256x256
                cache_size: 缓存大小限制
                image_format: 输出图像格式，支持PNG与JPEG
        """
        self.data_root = Path(data_root)
        self.index_root = Path(index_root) if index_root else self.data_root
        self.index_path = self.index_root / index_file
        self.luotu_path = self.index_root / luotu_file
        self.tile_size = tile_size
        self.cache_size = cache_size
        self.image_format = image_format.upper()


# 缓存管理
class TileCache:
    """瓦片缓存管理器"""

    def __init__(self, max_size: int = 128) -> None:
        """
        初始化缓存管理器

        Args:
                max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self.cache: dict[str, bytes] = {}
        self.access_order: list[str] = []

    def get(self, key: str) -> bytes | None:
        """
        从缓存获取瓦片数据

        Args:
                key: 缓存键

        Returns:
                瓦片字节数据或None
        """
        if key in self.cache:
            # 更新访问顺序(LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, data: bytes) -> None:
        """
        将瓦片数据存入缓存

        Args:
                key: 缓存键
                data: 瓦片字节数据
        """
        if key in self.cache:
            # 更新现有条目
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # 移除最久未使用的条目
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = data
        self.access_order.append(key)

    def _generate_key(self, x: int, y: int, z: int) -> str:
        """
        生成缓存键

        Args:
                x, y, z: 瓦片坐标

        Returns:
                缓存键字符串
        """
        return f"tile_{z}_{x}_{y}"


# 瓦片服务核心类
class TileService:
    """瓦片服务核心处理类"""

    def __init__(self, config: TileServiceConfig) -> None:
        """
        初始化瓦片服务

        Args:
                config: 服务配置对象
                        - data_root: 数据根目录路径
                        - index_root: 索引文件根目录路径
                        - index_path: 空间索引文件路径
                        - luotu_path: 落图文件路径
                        - tile_size: 瓦片大小，默认256x256
                        - cache_size: 缓存大小限制
                        - image_format: 输出图像格式
        """
        self.config = config
        self.cache = TileCache(config.cache_size)
        self.luotu, self.sindex = self._load_spatial_index()

    def _load_spatial_index(self) -> tuple[GeoDataFrame, SpatialIndex]:
        """
        加载空间索引和元数据

        Returns:
                GeoDataFrame和空间索引的元组
        """
        index_root, index_path, luotu_path = self.config.index_root, self.config.index_path, self.config.luotu_path
        # 检查索引文件是否存在
        if not (
            index_path.exists()
            and luotu_path.exists()
        ):
            # 自动构建索引
            image_list = list(index_root.rglob("**/*.tif"))
            if not image_list:
                # 没有tif文件时，创建空的GeoDataFrame和索引
                print(
                    f"警告: 在 {index_root} 中未找到任何.tif文件，将使用空索引"
                )
                empty_gdf = gpd.GeoDataFrame(
                    {"path": [], "geometry": []}, crs="EPSG:4326"
                )  # type: ignore
                return empty_gdf, empty_gdf.sindex

            # 转换为字符串路径列表
            image_paths = [str(img) for img in image_list]
            self.build_index(
                image_paths,
                self.config.data_root,
                luotu_path.name,
                index_path.name
            )


        return load_rtree_index(index_path, luotu_path)

    def _xyz_to_tile_data(self, x: int, y: int, z: int) -> np.ndarray:
        """
        根据XYZ坐标获取瓦片数据数组

        Args:
                x, y, z: 瓦片坐标

        Returns:
                瓦片数据的numpy数组
        """

        # 转换XYZ坐标为地理边界框
        bbox = convert_xyz_to_bbox((x, y, z))

        # 查找相交的图像文件
        intersect_images = filter_intersect_image(self.luotu, self.sindex, bbox)

        if not intersect_images:
            # 返回空瓦片
            return np.zeros(
                (self.config.tile_size, self.config.tile_size, 3), dtype=np.uint8
            )

        # 加载数据数组
        if len(intersect_images) == 1:
            # 单个图像
            dataarray = load_dataarray(intersect_images[0])
        else:
            # 多个图像需要合并
            from ..tile import merge

            dataarrays = [load_dataarray(img) for img in intersect_images]
            dataarray = merge.merge_arrays(dataarrays)

        # 提取瓦片数据
        tile_data = get_tile(dataarray, bbox)
        return tile_data

    def _array_to_image(self, data: np.ndarray, format: str = "PNG") -> bytes:
        """
        将numpy数组转换为图像字节流

        Args:
                data: 数据数组
                format: 图像格式

        Returns:
                图像字节数据
        """

        image = Image.fromarray(data, mode="RGB")

        # 转换为字节流
        buffer = BytesIO()
        image.save(buffer, format=format.upper())
        return buffer.getvalue()

    def _resize_tile(self, data: np.ndarray, target_size: int = 256) -> np.ndarray:
        """
        调整瓦片大小到标准尺寸

        Args:
                data: 原始数据数组
                target_size: 目标尺寸

        Returns:
                调整后的数据数组
        """
        from PIL import Image

        if data.shape[0] == target_size and data.shape[1] == target_size:
            return data

        # 转换为PIL图像进行缩放
        if data.ndim == 2:
            image = Image.fromarray(data.astype(np.uint8), mode="L")
        else:
            image = Image.fromarray(data.astype(np.uint8), mode="RGB")

        # 缩放图像
        resized_image = image.resize(
            (target_size, target_size), Image.Resampling.LANCZOS
        )

        # 转换回numpy数组
        resized_data = np.array(resized_image)

        # 确保维度正确
        if data.ndim == 3 and resized_data.ndim == 2:
            resized_data = np.stack([resized_data] * data.shape[2], axis=-1)

        return resized_data

    def get_tile(self, x: int, y: int, z: int) -> bytes:
        """
        获取指定XYZ坐标的瓦片图像

        Args:
                x, y, z: 瓦片坐标

        Returns:
                瓦片图像字节数据
        """
        # 检查缓存
        cache_key = self.cache._generate_key(x, y, z)
        cached_tile = self.cache.get(cache_key)
        if cached_tile:
            return cached_tile

        # 获取瓦片数据
        tile_data = self._xyz_to_tile_data(x, y, z)

        # 调整尺寸
        resized_data = self._resize_tile(tile_data, self.config.tile_size)

        # 转换为图像
        image_bytes = self._array_to_image(resized_data, self.config.image_format)

        # 存入缓存
        self.cache.put(cache_key, image_bytes)

        return image_bytes

    def build_index(
        self,
        image_list: Sequence[Pathlike],
        save_root: Pathlike,
        luotu_name: str = "luotu.geojson",
        index_name: str = "index.pkl",
    ) -> None:
        """
        构建空间索引

        Args:
                image_list: 图像文件路径列表
                save_root: 保存根目录
                luotu_name: 落图文件名
                index_name: 索引文件名
        """
        build_rtree_index(list(image_list), save_root, luotu_name, index_name)


# Flask应用工厂
def create_app(config: TileServiceConfig) -> Flask:
    """
    创建Flask应用实例

    Args:
            config: 瓦片服务配置

    Returns:
            配置好的Flask应用实例
    """
    app = Flask(__name__)

    # 创建瓦片服务实例
    tile_service = TileService(config)

    # 注册路由
    register_routes(app, tile_service)

    return app


def register_routes(app: Flask, tile_service: TileService) -> None:
    """
    注册瓦片服务路由

    Args:
            app: Flask应用实例
            tile_service: 瓦片服务实例
    """

    @app.route("/tiles/<int:z>/<int:x>/<int:y>")
    def get_tile_endpoint(z: int, x: int, y: int) -> Response:
        """
        瓦片获取端点

        Args:
                z: 缩放级别
                x: X坐标
                y: Y坐标

        Returns:
                包含瓦片图像的HTTP响应
        """
        try:
            tile_bytes = tile_service.get_tile(x, y, z)

            # 确定MIME类型
            content_type = f"image/{tile_service.config.image_format.lower()}"
            if tile_service.config.image_format.upper() == "JPEG":
                content_type = "image/jpeg"

            return Response(
                tile_bytes,
                mimetype=content_type,
                headers={
                    "Cache-Control": "public, max-age=86400",  # 缓存1天
                    # "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Access-Control-Allow-Origin": "*",
                },
            )
                    
        except Exception as e:
            return Response(
                f"获取瓦片失败: {str(e)}", status=500, mimetype="text/plain"
            )

    @app.route("/health")
    def health_check() -> dict[str, str]:
        """
        健康检查端点

        Returns:
                服务状态信息
        """
        return {
            "status": "healthy",
            "service": "tile_service",
            "data_root": str(tile_service.config.data_root),
            "cache_size": str(len(tile_service.cache.cache)),
            "tile_format": tile_service.config.image_format,
        }
