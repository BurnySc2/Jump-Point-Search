import numpy as np




def can_draw_line(y1, x1, y2, x2, array, wall):
    # Stolen from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]

    Modified so that it returns a tuple of two lists
    """
    # Setup initial conditions
    # x1, y1 = start
    # x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    # Recalculate differentials
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Calculate error
    error = dx // 2
    # error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    for x in range(x1, x2 + 1):
        # coord = (y, x) if is_steep else (x, y)

        # points.append(coord)
        if is_steep:
            # x_coords.append(x)
            # y_coords.append(y)
            if array[y, x] == wall:
                return False
        else:
            # x_coords.append(y)
            # y_coords.append(x)
            if array[x, y] == wall:
                return False

        error -= dy
        if error < 0:
            y += ystep
            error += dx
    return True

def get_line(y1, x1, y2, x2):
    # Stolen from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]

    Modified so that it returns a tuple of two lists
    """
    # Setup initial conditions
    # x1, y1 = start
    # x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Calculate error
    error = dx // 2
    # error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    y_coords = []
    x_coords = []

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        # coord = (y, x) if is_steep else (x, y)

        # points.append(coord)
        if is_steep:
            x_coords.append(x)
            y_coords.append(y)
        else:
            x_coords.append(y)
            y_coords.append(x)

        error -= dy
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    # return points
    return y_coords, x_coords




def get_line_aa(r0, c0, r1, c1):
    # Stolen from https://github.com/scikit-image/scikit-image/blob/28863d6e22b65787688de2101b9ddffe837380cf/skimage/draw/_draw.pyx#L108
    """Generate anti-aliased line pixel coordinates.
    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).
    Returns
    -------
    rr, cc, val : (N,) ndarray (int, int, float)
        Indices of pixels (`rr`, `cc`) and intensity values (`val`).
        ``img[rr, cc] = val``.
    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf
    """
    rr = list()
    cc = list()
    val = list()

    dc = abs(c0 - c1)

    dr = abs(r0 - r1)
    err = dc - dr


    if c0 < c1:
        sign_c = 1
    else:
        sign_c = -1

    if r0 < r1:
        sign_r = 1
    else:
        sign_r = -1

    if dc + dr == 0:
        ed = 1
    else:
        ed = (dc*dc + dr*dr)**0.5

    c, r = c0, r0
    while True:
        cc.append(c)
        rr.append(r)
        val.append(abs(err - dc + dr) / ed)

        err_prime = err
        c_prime = c

        if (2 * err_prime) >= -dc:
            if c == c1:
                break
            if (err_prime + dr) < ed:
                cc.append(c)
                rr.append(r + sign_r)
                val.append(abs(err_prime + dr) / ed)
            err -= dr
            c += sign_c

        if 2 * err_prime <= dr:
            if r == r1:
                break
            if (dc - err_prime) < ed:
                cc.append(c_prime + sign_c)
                rr.append(r)
                val.append(abs(dc - err_prime) / ed)
            err += dc
            r += sign_r

    return (np.array(rr, dtype=np.intp),
            np.array(cc, dtype=np.intp))#,
            # 1. - np.array(val, dtype=np.float))