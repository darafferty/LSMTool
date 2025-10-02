import numpy as np
import pytest
from lsmtool.facet import tessellate


def _flatten(iterable):
    return {tuple(point) for item in iterable for point in item}


@pytest.mark.parametrize(
    "ra_cal, ra_dec, ra_mid, dec_mid, width_ra, width_dec, "
    "expected_facet_points, expected_facet_polygons",
    [
        pytest.param(
            RA_CAL := [119.73, 138.08, 124.13, 115.74],
            DEC_CAL := [89.92, 89.91, 89.89, 89.89],
            126.52,
            90.0,
            0.3,
            0.3,
            # expected_facet_points
            np.transpose([RA_CAL, DEC_CAL]),
            #
            [
                [
                    [127.3, 89.9],
                    [278.9, 89.8],
                    [351.5, 89.8],
                    [51.9, 89.8],
                    [119.9, 89.9],
                    [127.3, 89.9],
                ],
                [
                    [127.3, 89.9],
                    [278.9, 89.8],
                    [261.5, 89.8],
                    [171.5, 89.8],
                    [146.9, 89.8],
                    [127.3, 89.9],
                ],
                [
                    [127.3, 89.9],
                    [146.9, 89.8],
                    [119.9, 89.8],
                    [119.9, 89.9],
                    [127.3, 89.9],
                ],
                [
                    [119.9, 89.9],
                    [51.9, 89.8],
                    [81.5, 89.8],
                    [119.9, 89.8],
                    [119.9, 89.9],
                ],
            ],
        )
    ],
)
def test_tessellate(
    ra_cal,
    ra_dec,
    ra_mid,
    dec_mid,
    width_ra,
    width_dec,
    expected_facet_points,
    expected_facet_polygons,
):
    """
    Test the tessellate function, using a region that encompasses the North
    Celestial Pole (NCP).
    """

    # Tessellate a region that encompasses the NCP.
    facet_points, facet_polys = tessellate(
        ra_cal,
        ra_dec,
        ra_mid,
        dec_mid,
        width_ra,
        width_dec,
    )

    # Check the facet points
    np.testing.assert_allclose(facet_points, expected_facet_points)

    # Check the facet polygons. Since the tessellate function is not
    # guaranteed to return the same order of vertices for identical
    # polygons, check only that the set of expected and actual vertices
    # are identical.
    facet_polys = [np.round(a, 1).tolist() for a in facet_polys]
    facet_polys_flat = _flatten(facet_polys)

    facet_polys_expected = _flatten(expected_facet_polygons)
    difference = facet_polys_flat.symmetric_difference(facet_polys_expected)
    assert not difference
