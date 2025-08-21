# Tile Service

一个快速的地理瓦片服务，用于将栅格数据作为Web地图瓦片提供服务。

## 功能特性

- 🚀 高性能瓦片生成和缓存
- 🗺️ 支持标准XYZ瓦片格式
- 📁 自动空间索引构建
- 🔄 智能数据合并和处理
- 🌐 兼容ipyleaflet和其他Web地图库
- ⚡ Fast-fail设计，便于调试

## 安装

### 使用pip+git直接安装

```bash
# 安装基础依赖
pip install git+https://github.com/yourusername/tile_service.git

# 安装开发依赖
pip install "git+https://github.com/yourusername/tile_service.git#egg=tile-service[dev]"

# 安装Jupyter支持
pip install "git+https://github.com/yourusername/tile_service.git#egg=tile-service[jupyter]"
```

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/tile_service.git
cd tile_service

# 安装项目及其依赖
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"

# 安装Jupyter支持
pip install -e ".[jupyter]"
```

### 使用uv安装依赖：

```bash
# 安装基础依赖
uv sync

# 安装开发依赖
uv sync --extra dev

# 安装Jupyter支持
uv sync --extra jupyter
```

## 快速开始

### 1. 配置和启动服务

```python
from tile_service.service.flask import create_app, TileServiceConfig

# 配置瓦片服务
config = TileServiceConfig(
    data_root="/path/to/your/tif/files",
    tile_size=256,
    cache_size=128,
    image_format="PNG"
)

# 创建Flask应用
app = create_app(config)

# 启动服务
app.run(host="0.0.0.0", port=8080)
```

### 2. 在ipyleaflet中使用

```python
import ipyleaflet
from ipyleaflet import Map, TileLayer

# 创建瓦片图层
tile_layer = TileLayer(
    url="http://localhost:8080/tiles/{z}/{x}/{y}",
    name="Custom Tiles"
)

# 创建地图
m = Map(center=[39.9, 116.4], zoom=10)
m.add_layer(tile_layer)
m
```

## 配置选项

- `data_root`: 栅格数据目录路径
- `index_file`: 空间索引文件路径（可选，自动构建）
- `luotu_file`: 地理数据文件路径（可选，自动构建）
- `tile_size`: 瓦片大小，默认256x256
- `cache_size`: 缓存大小限制
- `image_format`: 输出格式（PNG/JPEG）

## API端点

- `GET /tiles/{z}/{x}/{y}` - 获取瓦片
- `GET /health` - 健康检查

## 开发

```bash
# 代码格式化
uv run black tile_service/
uv run isort tile_service/

# 类型检查
uv run mypy tile_service/

# 运行测试
uv run pytest
```

## 许可证

MIT License