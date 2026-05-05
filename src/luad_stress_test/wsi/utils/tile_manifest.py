from typing import TypedDict


class _TileManifest_Slide_Grid(TypedDict):
    rows: int
    columns: int


class _TileManifest_Slide(TypedDict):
    path: str
    name: str
    grid: _TileManifest_Slide_Grid


class _TileManifest_Patch(TypedDict):
    name: str
    size: int
    offset: int
    mpp: float


class TileManifest(TypedDict):
    version: str
    slide: _TileManifest_Slide
    patch: _TileManifest_Patch
