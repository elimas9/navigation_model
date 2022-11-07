from enum import IntEnum
import numpy as np
import matplotlib
from scipy.spatial import KDTree


class TileSystem:

    @staticmethod
    def char_to_mazetile(ch):
        raise NotImplementedError()

    @staticmethod
    def mazetile_tochar(mt):
        raise NotImplementedError()

    @staticmethod
    def is_visitable(t):
        raise NotImplementedError()

    @staticmethod
    def colormap():
        raise NotImplementedError()


class StandardTiles(TileSystem):

    class TileType(IntEnum):
        WALL = 1
        CORNER = 2
        CENTRE = 3
        VOID = 0

    @staticmethod
    def char_to_mazetile(ch):
        if ch == 'W':
            return StandardTiles.TileType.WALL
        if ch == 'C':
            return StandardTiles.TileType.CORNER
        if ch == 'M':
            return StandardTiles.TileType.CENTRE
        return StandardTiles.TileType.VOID

    @staticmethod
    def mazetile_tochar(mt):
        if mt == StandardTiles.TileType.WALL:
            return 'W'
        if mt == StandardTiles.TileType.CORNER:
            return 'C'
        if mt == StandardTiles.TileType.CENTRE:
            return 'M'
        if mt == StandardTiles.TileType.VOID:
            return 'V'

    @staticmethod
    def is_visitable(t):
        return t != StandardTiles.TileType.VOID

    @staticmethod
    def colormap():
        return matplotlib.colors.ListedColormap(['w', 'r', 'g', 'b'])


class Maze:

    def __init__(self, layout, subareas, size_tile, tile_system=StandardTiles):
        self._tile_system = tile_system
        self._size_tile = size_tile
        self._layout = self._parse_layout(layout)
        self._sizeX = len(self._layout)  # this assumes that the maze is a square
        self._sizeY = len(self._layout[0])
        self._subareas = self._parse_subareas(subareas)
        # create k-d tree to find closest tiles
        centers = []
        for x in range(self.size_x):
            for y in range(self.size_y):
                if self.is_visitable(x, y) != self.MazeTile.VOID:
                    centers.append(self.disc2cont(x, y))
        self._kdtree = KDTree(centers)

    #################
    # Maze creation
    def _parse_layout(self, strlayout):
        """
        Parse a string maze layout and return an matrix of tiles

        :param strlayout: layout as a string
        :return: matrix of tiles
        """
        layout = []
        row = []
        for c in strlayout:
            if c == '\n':
                layout.append(row)
                row = []
            else:
                row.append(self._tile_system.char_to_mazetile(c))
        if len(row) > 0:
            layout.append(row)
        return layout

    @staticmethod
    def _parse_subareas(strsub):
        """
        Parse a string subareas layout and returs a matrix of numbers

        :param strsub: subareas layout as a string
        :return: matrix of integers or None
        """
        if strsub is None or strsub == "":
            return None
        subareas = []
        row = []
        for c in strsub:
            if c == '\n':
                subareas.append(row)
                row = []
            else:
                row.append(int(c))
        if len(row) > 0:
            subareas.append(row)
        if len(subareas) > 0:
            return subareas
        return None

    @classmethod
    def from_file(cls, filename, size_tile, tile_system=StandardTiles):
        """
        Load a maze from file

        :param filename: path to maze file
        :param size_tile: size of the tiles
        :param tile_system: TileSystem class
        :return: Maze object
        """
        with open(filename) as file:
            filecontent = file.read()

        if len(filecontent) > 0:
            contsplit = filecontent.split("SUB")
            if len(contsplit) > 1:
                subareas = contsplit[1]
                if subareas[0] == '\n':
                    subareas = subareas[1:]
            else:
                subareas = ""
            return cls(contsplit[0], subareas, size_tile=size_tile, tile_system=tile_system)
        else:
            return None

    #################
    # Maze properties
    # noinspection PyPep8Naming
    @property
    def MazeTile(self):
        return self._tile_system.TileType

    @property
    def size_tile(self):
        return self._size_tile

    @property
    def shape(self):
        return self._sizeX, self._sizeY

    @property
    def size_x(self):
        return self._sizeX

    @property
    def size_y(self):
        return self._sizeY

    def description(self):
        """
        Returns a description of the maze.

        :return: dictionary {TileType: num}
        """
        descr = {t: 0 for t in self.MazeTile}
        for x in range(0, self.size_x):
            for y in range(0, self.size_y):
                descr[self.get_tile(x, y)] += 1
        return descr

    @property
    def subareas(self):
        """
        Returns a sequence with all subareas indices

        :return: set of subareas indices
        """
        subs = []
        for x in range(0, self.size_x):
            for y in range(0, self.size_y):
                subs.append(self.get_subarea(x, y))
        return set(subs)


    #################
    # Maze getters
    def assert_subareas(self):
        if self._subareas is None:
            raise RuntimeError("Requesting access to subareas, but no subareas present")

    def get_tile_centre(self, x, y=None):
        """
        Get the center of the tile in continuous coordinates

        No boundary check is performed.

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :return: (cx, cy)
        """
        if y is None:
            y = x[1]
            x = x[0]
        cx = self.size_tile * (x + 1) - self.size_tile / 2
        cy = self.size_tile * (y + 1) - self.size_tile / 2
        return cx, cy

    def get_tile(self, x, y=None):
        """
        Get a tile from the maze

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :return: tile
        """
        if y is None:
            return self._layout[x[0]][x[1]]
        return self._layout[x][y]

    def get_tiles(self, lpos):
        """
        Get a list of tiles from the maze

        :param lpos: list of (x, y) coordinates
        :return: list of tiles
        """
        return [self.get_tile(p) for p in lpos]

    def get_subarea(self, x, y=None):
        """
        Get the index of the subarea of a tile

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :return: index of the subarea of (x, y)
        """
        self.assert_subareas()
        if y is None:
            return self._subareas[x[0]][x[1]]
        return self._subareas[x][y]

    def get_subareas(self, lpos):
        """
        Get the indices of the subareas of a list of tiles

        :param lpos: list of (x, y) coordinates
        :return: indices of the subarea of (x, y)
        """
        return [self.get_subarea(p) for p in lpos]

    def get_coordinates_of_type(self, tile_type):
        """
        Get a list of all coordinates that corresponds to tiles of a certain type

        :param tile_type: type of the tile to search
        :return: list of coordinates
        """
        coords = []
        for x in range(self.size_x):
            for y in range(self.size_y):
                if self.get_tile(x, y) == tile_type:
                    coords.append((x, y))
        return coords

    def get_visitable_coordinates(self):
        """
        Get a list of all coordinates that correspond to visitable tiles

        :return: list of coordinates
        """
        coords = []
        for x in range(self.size_x):
            for y in range(self.size_y):
                if self.is_visitable(x, y):
                    coords.append((x, y))
        return coords

    def get_non_visitable_coordinates(self):
        """
        Get a list of all coordinates that correspond to non-visitable tiles

        :return: list of coordinates
        """
        coords = []
        for x in range(self.size_x):
            for y in range(self.size_y):
                if not self.is_visitable(x, y):
                    coords.append((x, y))
        return coords

    def get_number_visitable_tiles(self):
        """
        Return the number of visitable tiles

        :return: number of visitable tiles
        """
        number_visitable_tiles = 0
        for x in range(self.size_x):
            for y in range(self.size_y):
                if self.is_visitable(x, y):
                    number_visitable_tiles += 1
        return number_visitable_tiles

    def get_closest_visitable(self, x, y=None):
        """
        Get the closest visitable tile

        :param x: x continuous coordinate (or pair of coordinates)
        :param y: y continuous coordinate (or None)
        :param y:
        :return: discrete coordinates of the closest visitable tile
        """
        if y is None:
            y = x[1]
            x = x[0]
        if self.is_visitable(x, y):
            return x, y
        return self.get_closest_visitable_cont(self.disc2cont(x, y))

    #################
    # Continuous coordinates
    def cont2disc(self, x, y=None):
        """
        Converts a continuous position to a discrete one.

        No boundary check is performed.

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :return: tuple with discrete coordinates
        """
        if y is None:
            y = x[1]
            x = x[0]
        dx = int(x // self.size_tile)
        dy = int(y // self.size_tile)
        return dx, dy

    def cont2disc_list(self, lpos):
        """
        Converts a list of continuous positions to a list of discrete ones.

        No boundary check is performed.

        :param lpos: list of continuous positions
        :return: list of tuples with discrete coordinates
        """
        return [self.cont2disc(p) for p in lpos]

    def disc2cont(self, x, y=None):
        """
        Converts a discrete position to a continuous one.

        No boundary check is performed.

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :return: tuple with discrete coordinates
        """
        return self.get_tile_centre(x, y)

    def disc2cont_list(self, lpos):
        """
        Converts a list of discrete positions to a list of continuous ones.

        No boundary check is performed.

        :param lpos: list of discrete positions
        :return: list of tuples with continuous coordinates
        """
        return [self.disc2cont(p) for p in lpos]

    def get_tile_cont(self, x, y=None):
        """
        Get a tile from the maze (continuous coordinates)

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :return: tile
        """
        return self.get_tile(self.cont2disc(x, y))

    def get_tiles_cont(self, lpos):
        """
        Get a list of tiles from the maze (continuous coordinates)

        :param lpos: list of (x, y) coordinates
        :return: list of tiles
        """
        return [self.get_tile_cont(p) for p in lpos]

    def get_subarea_cont(self, x, y=None):
        """
        Get the index of the subarea of a tile (continuous coordinates)

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :return: index of the subarea of (x, y)
        """
        return self.get_subarea(self.cont2disc(x, y))

    def get_subareas_cont(self, lpos):
        """
        Get the indices of the subareas of a list of tiles (continuous coordinates)

        :param lpos: list of (x, y) coordinates
        :return: indices of the subarea of (x, y)
        """
        return [self.get_subarea_cont(p) for p in lpos]

    def get_closest_visitable_cont(self, x, y=None):
        """
        Get the closest visitable tile

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :param y:
        :return: discrete coordinates of the closest visitable tile
        """
        if y is None:
            y = x[1]
            x = x[0]
        if self.is_visitable_cont(x, y):
            return self.cont2disc(x, y)
        _, ii = self._kdtree.query([[x, y]], k=1)
        return self.cont2disc(self._kdtree.data[ii[0]])

    #################
    # Checks
    def is_visitable(self, x, y=None):
        """
        Check if the tile is in range and is visitable

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :return: True if the tile is visitable, false otherwise
        """
        if y is None:
            y = x[1]
            x = x[0]
        if x < 0 or x >= self.size_x or y < 0 or y >= self.size_y:
            return False
        return self._tile_system.is_visitable(self.get_tile(x, y))

    def are_visitable(self, lpos):
        """
        Checks if the positions are visitable

        :param lpos: list of discrete positions
        :return: list of True/False values
        """
        return [self.is_visitable(p) for p in lpos]

    def is_movement_possible(self, x1, y1, x2=None, y2=None):
        """
        Check if the movement between two tiles is possible

        :param x1: x coordinate of the first tile (or pair of coordinates for first tile)
        :param y1: y coordinate of the first tile (or pair of coordinates for second tile)
        :param x2: x coordinate of the second tile (or None)
        :param y2: y coordinate of the second tile (or None)
        :return: True if the movement is possible, false otherwise
        """
        if y2 is None:
            y2 = y1[1]
            x2 = y1[0]
            y1 = x1[1]
            x1 = x1[0]
        return self.is_movement_possible_cont(*self.disc2cont(x1, y1), *self.disc2cont(x2, y2))

    def are_movements_possible(self, x, y, lpos=None):
        """
        Check if multiple movements are possible

        :param x: x coordinate of the first tile (or pair of coordinates for first tile)
        :param y: y coordinate of the first tile (or list of coordinates)
        :param lpos: list of coordinates (or None)
        :return: list of booleans indicating whether the movement is possible or not
        """
        if lpos is None:
            lpos = y
            y = x[1]
            x = x[0]
        return [self.is_movement_possible(x, y, p[0], p[1]) for p in lpos]

    def is_visitable_cont(self, x, y=None):
        """
        Check if the tile is in range and is visitable

        :param x: x coordinate (or pair of coordinates)
        :param y: y coordinate (or None)
        :return: True if the tile is visitable, false otherwise
        """
        return self.is_visitable(self.cont2disc(x, y))

    def are_visitable_cont(self, lpos):
        """
        Checks if the positions are visitable

        :param lpos: list of discrete positions
        :return: list of True/False values
        """
        return [self.is_visitable_cont(p) for p in lpos]

    def is_movement_possible_cont(self, x1, y1, x2=None, y2=None):
        """
        Check if the movement between two tiles is possible

        :param x1: x coordinate of the first tile (or pair of coordinates for first tile)
        :param y1: y coordinate of the first tile (or pair of coordinates for second tile)
        :param x2: x coordinate of the second tile (or None)
        :param y2: y coordinate of the second tile (or None)
        :return: True if the movement is possible, false otherwise
        """
        if y2 is None:
            y2 = y1[1]
            x2 = y1[0]
            y1 = x1[1]
            x1 = x1[0]
        if not self.is_visitable(self.cont2disc(x1, y1)) or not self.is_visitable(self.cont2disc(x2, y2)):
            return False
        # uses a variation of Ray/AABB intersection
        # intersections with vertical lines
        if x1 != x2:
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            m = (y2 - y1) / (x2 - x1)
            minx = int(x1 // self.size_tile)
            maxx = int(x2 // self.size_tile)
            for dx in range(minx, maxx):
                x = (dx + 1) * self.size_tile
                y = m * (x - x1) + y1
                if not self.is_visitable(self.cont2disc(x - self.size_tile, y)):  # left tile
                    return False
                if not self.is_visitable(self.cont2disc(x, y)):  # right tile
                    return False
        # intersections with horizontal lines
        if y1 != y2:
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            m = (x2 - x1) / (y2 - y1)
            miny = int(y1 // self.size_tile)
            maxy = int(y2 // self.size_tile)
            for dy in range(miny, maxy):
                y = (dy + 1) * self.size_tile
                x = m * (y - y1) + x1
                if not self.is_visitable(self.cont2disc(x, y - self.size_tile)):  # lower tile
                    return False
                if not self.is_visitable(self.cont2disc(x, y)):  # upper tile
                    return False
        return True

    def are_movements_possible_cont(self, x, y, lpos=None):
        """
        Check if multiple movements are possible

        :param x: x coordinate of the first tile (or pair of coordinates for first tile)
        :param y: y coordinate of the first tile (or list of coordinates)
        :param lpos: list of coordinates (or None)
        :param step_size: size of the step for the checks
        :return: list of booleans indicating whether the movement is possible or not
        """
        if lpos is None:
            lpos = y
            y = x[1]
            x = x[0]
        return [self.is_movement_possible_cont(x, y, p[0], p[1]) for p in lpos]

    #################
    # Visualization
    def print_layout(self):
        """
        Print the maze layout for visualization
        """
        for row in self._layout:
            for t in row:
                print(self._tile_system.mazetile_tochar(t) + " ", end='')
            print()

    def print_subareas(self):
        """
        Print the maze subareas for visualization
        """
        self.assert_subareas()
        for row in self._subareas:
            for a in row:
                print(str(a) + " ", end='')
            print()

    def print_neighb(self, x, y=None):
        """
        Print the neighbour states for visualization (not possible next states)

        :param x: discrete x coordinate of the tile (or pair of coordinates)
        :param y: discrete y coordinate of the tile (or None)
        """
        if y is None:
            y = x[1]
            x = x[0]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                t = "X "
                if (0 <= x+dx < self.size_x) and (0 <= y+dy < self.size_y):
                    t = self._tile_system.mazetile_tochar(self.get_tile(x+dx, y+dy)) + " "
                print(t, end='')
            print()

    def plot(self, ax, tiles=True, borders=False, grid=False):
        """
        Plot the maze

        :param ax: axis where to plot
        :param tiles: if True plot colored tiles
        :param borders: if True add maze borders
        :param grid: if True add grid
        """
        if tiles:
            data = np.zeros(self.shape)
            for x in range(self.size_x):
                for y in range(self.size_y):
                    data[x, y] = int(self.get_tile(x, y))
            ax.imshow(data.transpose(), origin="lower",
                      extent=[0, self.size_x * self.size_tile,
                              0, self.size_y * self.size_tile],
                      cmap=self._tile_system.colormap(),
                      norm=matplotlib.colors.Normalize(vmin=0, vmax=len(self._tile_system.TileType)))
        if borders:
            bwidth = 2
            # external borders
            for x in range(self.size_x):
                if self.is_visitable(x, 0):
                    ax.hlines(0, self.size_tile * x, self.size_tile * (x+1),
                              colors="k", linewidths=bwidth)
                if self.is_visitable(x, self.size_y-1):
                    ax.hlines(self.size_tile * self.size_y, self.size_tile * x, self.size_tile * (x + 1),
                              colors="k", linewidths=bwidth)
            for y in range(self.size_y):
                if self.is_visitable(0, y):
                    ax.vlines(0, self.size_tile * y, self.size_tile * (y+1),
                              colors="k", linewidths=bwidth)
                if self.is_visitable(self.size_x-1, y):
                    ax.vlines(self.size_tile * self.size_x, self.size_tile * y, self.size_tile * (y + 1),
                              colors="k", linewidths=bwidth)
            # internal borders
            for x in range(self.size_x-1):
                for y in range(self.size_y-1):
                    vis_tile = bool(self.is_visitable(x, y))
                    if vis_tile ^ bool(self.is_visitable(x+1, y)):  # sort of XOR
                        ax.vlines(self.size_tile * (x+1), self.size_tile * y, self.size_tile * (y + 1),
                                  colors="k", linewidths=bwidth)
                    if vis_tile ^ bool(self.is_visitable(x, y+1)):  # sort of XOR
                        ax.hlines(self.size_tile * (y+1), self.size_tile * x, self.size_tile * (x + 1),
                                  colors="k", linewidths=bwidth)

        if grid:
            gwidth = 1
            for x in range(self.size_x):
                for y in range(self.size_y):
                    vis_tile = self.is_visitable(x, y)
                    if vis_tile and x+1 < self.size_x and self.is_visitable(x+1, y):
                        ax.vlines(self.size_tile * (x+1), self.size_tile * y, self.size_tile * (y + 1),
                                  colors="k", linewidths=gwidth)
                    if vis_tile and y+1 < self.size_y and self.is_visitable(x, y+1):
                        ax.hlines(self.size_tile * (y+1), self.size_tile * x, self.size_tile * (x + 1),
                                  colors="k", linewidths=gwidth)

        # adjust visualization
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlim(-self.size_tile * 0.1, self.size_tile * (self.size_x + 0.1))
        ax.set_ylim(-self.size_tile * 0.1, self.size_tile * (self.size_y + 0.1))
        ax.set_aspect('equal')
