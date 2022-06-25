from typing import Iterable, Union

from deep_utils.utils.logging_utils import log_print, value_error_log


def utm2wgs84(
    x: Union[float, Iterable],
    y: Union[float, Iterable],
    zone_number: int,
    northern: bool,
    zone_letter=None,
    strict=True,
    return_list=True,
    logger=None,
    verbose=0,
):
    import utm

    get_zero = False
    if isinstance(x, Iterable) and isinstance(y, Iterable):
        point_list = [(x_, y_) for x_, y_ in zip(x, y)]
    elif isinstance(x, float) and isinstance(y, float):
        point_list = [(x, y)]
        get_zero = True
    else:
        value_error_log(logger, "Input x and y are not supported!")

    new_points = [
        utm.to_latlon(
            x,
            y,
            zone_number=zone_number,
            northern=northern,
            strict=strict,
            zone_letter=zone_letter,
        )
        for x, y in point_list
    ]
    log_print(
        logger,
        f"Successfully projected input points from utm to lat-long",
        verbose=verbose,
    )
    if return_list:
        x, y = list(zip(*new_points))
        if get_zero:
            x, y = x[0], y[0]
        return x, y

    if get_zero:
        new_points = new_points[0]
    return new_points


def project_points(
    x: Union[float, Iterable],
    y: Union[float, Iterable],
    input_type="epsg:3857",
    out_type="wgs84",
    return_list=True,
    logger=None,
    verbose=0,
):
    """
    Changing the projection of input points.
    Args:
        x:
        y:
        input_type:
        out_type:
        return_list:
        logger:
        verbose:

    Returns:

    """
    from collections.abc import Iterable

    import geopandas as gpd
    from shapely.geometry import Point

    get_zero = False
    if isinstance(x, Iterable) and isinstance(y, Iterable):
        point_list = [Point(x_, y_) for x_, y_ in zip(x, y)]
    elif isinstance(x, float) and isinstance(y, float):
        point_list = [Point(x, y)]
        get_zero = True
    else:
        value_error_log(logger, "Input x and y are not supported!")

    in_points = gpd.GeoSeries(point_list, crs=input_type)
    output_points = in_points.to_crs(out_type)
    log_print(
        logger,
        f"Successfully projected input points from {input_type} to {out_type}",
        verbose=verbose,
    )
    output_points = [(point.x, point.y) for point in output_points]
    if return_list:
        x, y = list(zip(*output_points))
        if get_zero:
            x, y = x[0], y[0]
        return x, y

    if get_zero:
        output_points = output_points[0]

    return output_points
