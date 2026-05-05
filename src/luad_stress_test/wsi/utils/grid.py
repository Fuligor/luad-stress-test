from dataclasses import dataclass


@dataclass
class Grid:
    columns: int
    rows: int

    def to_pil_size(self):
        return (self.columns, self.rows)

    def to_numpy_shape(self):
        return (self.rows, self.columns)

    def __repr__(self):
        return f"{self.__class__.__name__}(columns={self.columns}, rows={self.rows})"


@dataclass
class GridLocation:
    column: int
    row: int

    def __repr__(self):
        return f"{self.__class__.__name__}(column={self.column}, row={self.row})"
