import numpy as np


def vtx2roi(rmap, vtx):
    """Map vertex activity to ROIs with sum.
    """
    out = np.zeros((rmap.max() + 1,) + vtx.shape[1:])
    np.add.at(out, rmap, vtx)
    return out


def roi2vtx(rmap, reg):
    """Map ROI activity to vertices.
    """
    # TODO div by vtx count st. vtx2roi(rmap, roi2vtx(rmap, x)) is unit
    return reg[rmap]



