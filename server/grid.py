import math

from pyproj import Transformer


def compute_grid(sw_lonlat, ne_lonlat, split, chip_size_m, crs, max_chips=None):
    """Compute a UTM-snapped chip grid for a bounding box.

    Returns a list of dicts with keys: id, split, geometry_wkt.
    geometry_wkt is the polygon in the projected CRS,
    vertex order: [NE, SE, SW, NW, NE] to match existing metadata.geojson.
    """
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    sw_e, sw_n = to_proj.transform(sw_lonlat[0], sw_lonlat[1])
    ne_e, ne_n = to_proj.transform(ne_lonlat[0], ne_lonlat[1])

    min_e = math.floor(sw_e / chip_size_m) * chip_size_m
    min_n = math.floor(sw_n / chip_size_m) * chip_size_m
    max_e = math.ceil(ne_e / chip_size_m) * chip_size_m
    max_n = math.ceil(ne_n / chip_size_m) * chip_size_m

    cols = round((max_e - min_e) / chip_size_m)
    rows = round((max_n - min_n) / chip_size_m)
    count = cols * rows

    if max_chips is not None and count > max_chips:
        raise ValueError(
            f"Requested area would create {count} chips, exceeding limit of {max_chips}"
        )

    chips = []
    e = min_e
    while e < max_e:
        n = min_n
        while n < max_n:
            e2 = e + chip_size_m
            n2 = n + chip_size_m

            chip_id = f"{e:.2f}e_{n:.2f}n"

            # WKT in projected CRS (NE, SE, SW, NW, NE)
            wkt = (
                f"POLYGON(({e2} {n2}, {e2} {n}, {e} {n}, {e} {n2}, {e2} {n2}))"
            )

            chips.append({
                "id": chip_id,
                "split": split,
                "geometry_wkt": wkt,
            })

            n += chip_size_m
        e += chip_size_m

    return chips
