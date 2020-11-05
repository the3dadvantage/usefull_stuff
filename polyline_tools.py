import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.pyplot import figure

import numpy as np
import os


import numpy as np


# redistribute ==========================
def measure_angle_at_each_vert(grid):
    """Provide mesh and anlge limit in degrees.
    Returns the indices of verts that are sharper than limit"""

    co = grid.co
    
    v1 = np.roll(co, 1, axis=0) - co
    v2 = np.roll(co, -1, axis=0) - co
    
    # use the vecs pointing away from each vertex later
    grid.vls = v1
    grid.vrs = v2
    
    ls_dots = np.einsum('ij, ij->i', v1, v1)
    rs_dots = np.einsum('ij, ij->i', v2, v2)
    
    uv1 = v1 / np.sqrt(ls_dots)[:, None]
    uv2 = v2 / np.sqrt(rs_dots)[:, None]
    
    con = np.pi /180
    limit = np.cos(con * grid.angle_limit)
    
    angle = np.einsum('ij, ij->i', uv1, uv2)
    sharps = angle > limit   
    sharps[0] = True
    
    return np.arange(co.shape[0])[sharps]


def get_segments(grid):
    """Generate a list of segments between sharp edges"""
    sharps = grid.sharps
    
    # in case there are no sharps    
    if len(sharps) == 0:
        sharps = np.array([0])

    sharp = sharps[0]
    v = sharp
    other_verts2 = np.array([sharp + 1, sharp -1])
    move2 = other_verts2[0]

    seg = [sharp]
    segs = []
    for i in range(len(sharps)):
        while True:    
            if move2 in sharps:
                seg.append(move2)
                segs.append(seg)
                seg = []
            
            seg.append(move2)
            other_verts2 = np.array([move2 + 1, move2 - 1])
            new = other_verts2[other_verts2 != sharp][0]
            v = move2
            move2 = new
            
            if move2 > (grid.co.shape[0] -1):
                offset = move2 - grid.co.shape[0]
                move2 = offset
                seg.append(move2)
                segs.append(seg)
                return segs


def get_seg_length(grid, seg):
    """returns the total length of a set
    of points that are in linear order"""
    co = grid.co
    vecs = co[seg[1:]] - co[seg[:-1]]
    grid.seg_vecs.append(vecs) # might as well save this for later
    seg_length = np.sqrt(np.einsum('ij, ij->i', vecs, vecs))
    grid.seg_lengths.append(seg_length) # saving this also
    total_length = np.sum(np.sqrt(np.einsum('ij, ij->i', vecs, vecs)))
    return total_length


def generate_perimeter(grid, current):
    """Place points around perimeter"""

    # get the length of each segments
    seg_lengths = np.array([get_seg_length(grid, s) for s in grid.segments])
    grid.point_counts = seg_lengths // grid.size
    grid.point_counts[grid.size / seg_lengths > 0.5] = 1
    grid.spacing = seg_lengths / grid.point_counts
    
    # add the first point in the segment (second one gets added next time)   
    seg_sets = np.empty((0,3), dtype=np.float32)
    seg_sets = move_point_on_path(grid, current, seg_sets)

    return seg_sets
    

def move_point_on_path(grid, idx, seg_sets):
    """Walk the points until we are at the distance
    we space them"""
    
    co = grid.co
    seg = grid.segments[idx]
    lengths = grid.seg_lengths[idx]
    spacing = grid.spacing[idx]
    vecs = grid.seg_vecs[idx]
    count = grid.point_counts[idx]

    seg_co_set = [co[seg[0]]] # the last one will be filled in by the first one next time.
    if count == 0:
        return seg_co_set

    growing_length = 0
    len_idx = 0
    build = spacing
    
    counter = 0
    for x in range(int(count) - 1):    
        growing_length = 0
        len_idx = 0
        counter += 1
        while growing_length < spacing:
            growing_length += lengths[len_idx]
            len_idx += 1    
            
        # back up to the last point now 
        len_idx -= 1
        growing_length -= lengths[len_idx] 
        point = co[len_idx]
        
        # move from the past point along the last vector until we
        # hit the proper spacing
        end_offset = spacing - growing_length 
        last_dif = lengths[len_idx] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!  
        along_last = end_offset / last_dif
        
        move = vecs[len_idx]
        
        loc = co[seg[len_idx]] + move * along_last
        
        seg_co_set.append(loc)
        
        # start from the beginning because it's easier
        spacing += build
    
    # join to master array:
    seg_sets = np.append(seg_sets, seg_co_set, axis=0) 
    return seg_sets


def mag_set(mag, v2):    
    '''Applys the magnitude of v1 to v2'''
    d1 = mag ** 2
    d2 = v2 @ v2
    div = d1/d2
    return v2 * np.sqrt(div)


class Distribute:
    pass
    

def redistribute(cut_polyline):
    # walk around the edges and and plot evenly spaced points
    # respecting the sharpness of the angle

    grid = Distribute()
    grid.co = np.zeros((len(cut_polyline), 3), dtype=np.float32)
    grid.co[:, :2] = cut_polyline
    
    grid.angle_limit = 160
    grid.sharps = measure_angle_at_each_vert(grid)
    grid.segments = get_segments(grid)
    grid.size = 0.45
    grid.seg_vecs = [] # gets filled by the function below
    grid.seg_lengths = [] # gets filled by the function below
    iters = len(grid.segments)
    new_co = np.empty((0,3), dtype=np.float32)    

    # create points for every segment between sharps --------------------
    for i in range(iters):
        x = generate_perimeter(grid, i)
        new_co = np.append(new_co, x, axis=0)
    
    return new_co
# end redistribute ======================


def edge_edge_intersect_2d(a1,a2, b1,b2, intersect=False):
    '''simple 2d line intersect'''    
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = da[::-1] * np.array([1,-1])
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    scale = (num / denom)
    if intersect:
        return b1 + db * scale, (scale > 0) & (scale < 1)
    else:
        return b1 + db * scale


def edges_edges_intersect_2d(a1,a2, b1,b2, intersect=False):
    '''simple 2d line intersect'''    
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = da[:, ::-1] * np.array([1,-1])    
    denom = np.einsum('ij,ij->i', dap, db)    
    num = np.einsum('ij,ij->i', dap, dp)
    scale = (num / denom)
    if intersect:
        return b1 + db * scale[:, None], (scale > 0) & (scale < 1)
    else:
        return b1 + db * scale[:, None]


def two_d_quat(rad, axis):
    theta = (rad * 0.5)
    w = np.cos(theta)
    axis[:, 2] *= np.sin(theta)
    return w, axis


def q_rotate(co, w, axis):
    """Takes an N x 3 numpy array and returns that array rotated around
    the axis by the angle in radians w. (standard quaternion)"""    
    move1 = np.cross(axis, co)
    move2 = np.cross(axis, move1)
    move1 *= w[:, None]
    return co + (move1 + move2) * 2


def round_polyline_corners(rc):

    polyline = rc.polyline
    rc.p = np.array(polyline[0], dtype=np.float32)
        
    ### Remove doubles -----------------
    roll_vec = np.roll(rc.p, 1, axis=0) - rc.p    
    roll_length = np.sqrt(np.einsum('ij,ij->i', roll_vec, roll_vec))
    
    keep = roll_length > rc.merge_threshold
    rc.p = rc.p[keep] # !!! start and end do not overlap now !!!

    
    ### Measure every angle ------------
    l_vec = np.roll(rc.p, 1, axis=0) - rc.p
    r_vec = np.roll(rc.p, -1, axis=0) - rc.p
    
    ld = np.einsum('ij,ij->i', l_vec, l_vec)
    rd = np.einsum('ij,ij->i', r_vec, r_vec)
    
    rc.ul = l_vec / np.sqrt(ld)[:, None]
    rc.ur = r_vec / np.sqrt(rd)[:, None]
        
    ad = np.einsum('ij,ij->i', rc.ul, rc.ur)
    ad[ad > 1] = 1
    ad[ad < -1] = -1
    
    angle = np.arccos(ad)    
    con = 180 / np.pi
    a = angle * con
        
    ### Find the sharp turns. Eliminate turns close to 180
    rc.sharp = (a <= (180 - rc.angle_threshold)) & (a > rc.straight_threshold)        


def round(rc):

    # get an angle perp to surround sharp
    uls = rc.ul[rc.sharp]
    urs = rc.ur[rc.sharp]
    
    uls3 = np.zeros((uls.shape[0], 3), dtype=np.float32)
    uls3[:, :2] = uls
    
    urs3 = np.zeros((urs.shape[0], 3), dtype=np.float32)
    urs3[:, :2] = urs

    cross = np.cross(uls3, urs3)
    rc.cross = cross
    
    l_perp = np.cross(uls3, cross)
    r_perp = np.cross(cross, urs3)
    
    lpd = np.einsum('ij,ij->i', l_perp, l_perp)
    rpd = np.einsum('ij,ij->i', r_perp, r_perp)    
    
    ulp = l_perp / np.sqrt(lpd)[:, None]
    urp = r_perp / np.sqrt(rpd)[:, None]
    
    piv0 = np.zeros((uls.shape[0], 3), dtype=np.float32)
    sharp_points = rc.p[rc.sharp]
    rc.sharp_points = sharp_points
    piv0[:, :2] = sharp_points
    
    piv1 = np.copy(piv0)
    piv2 = np.copy(piv0)
    
    if rc.radius_generate:
        rc.sharp_rads = np.zeros(ulp.shape[0], dtype=np.float32)
        rc.sharp_rads += rc.radius
        
    piv1 += (ulp * rc.sharp_rads[:, None])
    piv2 += (urp * rc.sharp_rads[:, None])

    rc.radius_generate = False
    
    move1 = piv0 - piv2
    piv3 = piv1 + move1
    
    # ------------------------------------
    a1 = piv3[:, :2]
    a2 = a1 + urs
    b1 = sharp_points
    b2 = sharp_points + uls[:, :2]
    
    intersect = edges_edges_intersect_2d(a1,a2, b1,b2, intersect=False)
    move2 = intersect - piv1[:, :2]
    
    p0 = piv0[:, :2] + move2
    p1 = intersect
    p2 = piv2[:, :2] + move2

    # p0 is the pivot. p1 and p2 are the start and end of the radius
    # They are stored in an Nx3 and indexed like: p0 = dco[1::3]
    dco = np.zeros((sharp_points.shape[0] * 3, 3), dtype=np.float32)
    dco[0::3][:, :2] = p0
    dco[1::3][:, :2] = p1
    dco[2::3][:, :2] = p2

    rc.pivots = dco


def check_overlap(rc):

    # 1. get distance from point to pivots
    
    p1 = rc.pivots[1::3][:, :2]
    p2 = rc.pivots[2::3][:, :2]
    
    ls = rc.sharp_points - p1 
    rs = rc.sharp_points - p2 
    ld = np.sqrt(np.einsum('ij,ij->i', ls, ls)) 
    rd = np.sqrt(np.einsum('ij,ij->i', rs, rs))
        
    idx = np.arange(rc.p.shape[0])
    sharp_idx = idx[rc.sharp]
    
    left_sets = []
    right_sets = []
    
    rc.left_edges = []
    rc.right_edges = []
    
    for i, s in enumerate(sharp_idx):

        # left side
        length = 0
        skipped_verts = []
        
        l_shift = 1
        for j in range(rc.p.shape[0]): # instead of while
            v = s - l_shift
            l_shift += 1
            
            vec = rc.p[v + 1] - rc.p[v]
            vl = np.sqrt(vec @ vec)
            length += vl

            skipped_verts.append(idx[v])            
            
            if length > ld[i]:
                edge = [idx[v + 1], idx[v]]
                abs_edge = [v + 1, v]

                rc.left_edges.append(edge)
                left_sets.append([s, ld[i], length - vl, edge, abs_edge])
                break
            
        # right side
        length = 0
        skipped_verts = []
        
        r_shift = 1
        for j in range(rc.p.shape[0]): # instead of while
            v = s + r_shift
            vabs = s + r_shift
            if v > (rc.p.shape[0] - 1):
                v = (s + r_shift) - rc.p.shape[0]
            
            r_shift += 1
            
            vec = rc.p[v - 1] - rc.p[v]
            vl = np.sqrt(vec @ vec)
            length += vl

            skipped_verts.append(idx[v])            
            
            if length > ld[i]:
                edge = [idx[v - 1], idx[v]]
                abs_edge = [vabs - 1, vabs]
                rc.right_edges.append(edge)
                right_sets.append([s, ld[i], length - vl, edge, abs_edge])
                break

    rc.left_sets = left_sets        
    rc.right_sets = right_sets        
    
    # if we are on a curve the pivot, start, and end will be updated to a new edge
    rc.new_pivot_locs = [[], []]
    rc.new_pivot_vecs = [[], []]
    
    for i in range(len(left_sets)):
        ls = left_sets[i]
        rs = right_sets[i]
        
        # check for collisions on both sides:
        l_shift = i - 1
        r_shift = i + 1
        if r_shift == len(left_sets):
            r_shift = 0
    
        r_last = right_sets[l_shift]
        l_last = left_sets[r_shift]
        
        l_overlap = ls[3][0] <= r_last[3][1]
        r_overlap = rs[3][0] >= l_last[3][1]

        # get left offset location            
        l_offset_loc, lsd, lvec, ladd = offset_past(rc, ls)
        l_tdl = get_length_along_edges(rc.p, ls[0], ls[3][0], reverse=True) + ladd

        rc.new_pivot_locs[0].append(l_offset_loc)
        rc.new_pivot_vecs[0].append(lvec)
        
        # get last right offset location
        lr_offset_loc, lrsd, lrvec, lradd = offset_past(rc, r_last)
        lr_tdl = get_length_along_edges(rc.p, r_last[0], r_last[3][0], reverse=False) + lradd
        
        l_sharp_tdl = get_length_along_edges(rc.p, r_last[0], ls[0], reverse=False)
        l_passed = (lr_tdl + l_tdl) > l_sharp_tdl

        # -----------------------------
        
        # get right offset location
        r_offset_loc, rsd, rvec, radd = offset_past(rc, rs)
        r_tdl = get_length_along_edges(rc.p, rs[0], rs[3][0], reverse=False) + radd

        rc.new_pivot_locs[1].append(r_offset_loc)
        rc.new_pivot_vecs[1].append(rvec)
        
        # get last left offset location
        ll_offset_loc, llsd, llvec, lladd = offset_past(rc, l_last)
        ll_tdl = get_length_along_edges(rc.p, l_last[0], l_last[3][0], reverse=True) + lladd
        
        r_sharp_tdl = get_length_along_edges(rc.p, rs[0], l_last[0], reverse=False)
        r_passed = (ll_tdl + r_tdl) > r_sharp_tdl
        
        r_combined = ll_tdl + r_tdl
        l_combined = lr_tdl + l_tdl
        x = r_combined / rc.radius
        y = l_combined / rc.radius
        
        rmax = r_sharp_tdl / x
        lmax = l_sharp_tdl / y
        too_big = min([rmax, lmax]) - rc.error_margin
        
        if rc.sharp_rads[i] > too_big:
            rc.adjust_radius = True
            rc.sharp_rads[i] = too_big
                    
        overage = r_combined - r_sharp_tdl
        adjust = overage / (rc.radius / 2)
        new = rc.radius - adjust

        if False:
            print(ls[0], "sharp")
            print(ls[1], "pivot length")
            print(ls[2], "length up to edge")
            print(ls[3], "edge verts")
            print(ls[4], "abs edge") # not used...


def get_pivots(pivots0, pivots1, sharps, rc=None):
    """Place pivot using angle half way between
    two vectors assuming we already know the
    distance along the vector.
    Uses lenght * (1/cos) (same as l/dot of uvecs)"""
    new_lv = pivots0 - sharps
    new_rv = pivots1 - sharps
    
    nld = np.sqrt(np.einsum('ij,ij->i', new_lv, new_lv))
    nrd = np.sqrt(np.einsum('ij,ij->i', new_rv, new_rv))
    
    uld = new_lv / nld[:, None]
    urd = new_rv / nrd[:, None]
    
    compare = np.zeros((uld.shape[0],2),dtype=np.float32)
    compare[:, 0] = nld
    compare[:, 1] = nrd
    
    new_min = np.min(compare, axis=1)
    vec_rad = new_min
    
    replot_l = sharps + (uld * new_min[:, None])
    replot_r = sharps + (urd * new_min[:, None])

    if rc is not None:    
        rc.l_uvecs = uld
        rc.r_uvecs = urd
        rc.final_rads = new_min
        
    mid = (replot_l + replot_r) / 2
    
    midv = mid - sharps
    umid = midv / (np.sqrt(np.einsum('ij,ij->i', midv, midv)))[:, None]
    
    angle = np.einsum('ij,ij->i', uld, umid)
    new_mid = (umid * new_min[:, None]) / angle[:, None]

    new_center = sharps + new_mid
    
    return replot_l, replot_r, new_center
    

def fix_curves(rc):

    v0 = np.array(rc.new_pivot_vecs[0])
    v1 = np.array(rc.new_pivot_vecs[1])
    
    led = np.array(rc.left_edges)
    red = np.array(rc.right_edges)
    a2 = rc.p[led[:, 0]]
    a1 = rc.p[led[:, 1]]
    b2 = rc.p[red[:, 0]]
    b1 = rc.p[red[:, 1]]
    
    new_sharps = edges_edges_intersect_2d(a1,a2, b1,b2)
    replot_l, replot_r, new_center = get_pivots(np.array(rc.new_pivot_locs[0]), np.array(rc.new_pivot_locs[1]), new_sharps, rc)
    
    rc.pivots[1::3][:, :2] = replot_l
    rc.pivots[2::3][:, :2] = replot_r   
    rc.pivots[0::3][:, :2] = new_center


def offset_past(rc, ls):
    l_add_len = ls[1] - ls[2]
    lvec = rc.p[ls[3][1]] - rc.p[ls[3][0]]
    lsd = np.sqrt(lvec @ lvec)
    lvec /= lsd
    l_last_p = rc.p[ls[3][0]]
    loc = l_last_p + lvec * l_add_len
    return loc, lsd, lvec, l_add_len 
    

def get_length_along_edges(co, v0, v1, reverse=False):
    """Assumes loop is in numerical order.
    Start and stop verts.
    Reverse counts from right to left.
    Can cross -1 in either direction."""
    
    if reverse:
        co = co[::-1]
        idx = np.arange(co.shape[0])
        v0 = idx[-v0 - 1]
        v1 = idx[-v1 - 1]
    
    over = v0 > v1
    if not over:    
        dif = co[v0+1:v1+1] - co[v0:v1]
    else:
        offset = -v1 - 1
        roll_co = np.roll(co, offset, axis=0)
        dif = roll_co[v0 + (offset + 1):] - roll_co[v0 + offset: -1]

    dists = np.sqrt(np.einsum('ij,ij->i', dif, dif))
    dist = np.sum(dists)
    return dist

        
def create_radius_points(rc):

    rc.radius_points = []
    pivots = rc.pivots[0::3]    
    lps = rc.pivots[1::3]
    rps = rc.pivots[2::3]
    
    lpv = lps - pivots
    rpv = rps - pivots
    ulpv = lpv / np.sqrt(np.einsum('ij,ij->i', lpv, lpv))[:, None]
    urpv = rpv / np.sqrt(np.einsum('ij,ij->i', rpv, rpv))[:, None]
    
    axis = np.zeros((rc.pivots.shape[0]//3, 3), dtype=np.float32)
    axis[:, 2] = np.sign(rc.cross[:, 2])
        
    ad = np.einsum('ij,ij->i', ulpv, urpv)
    ad[ad < -1] = -1
    ad[ad > 1] = 1
    rads = np.arccos(ad)
    recon = np.pi / 180
    rad_step = rc.step_angle * recon # this is angle between points in rads
    
    steps = np.nan_to_num(rads // rad_step) # number of segments (ignore last)
    tick = rads / steps

    for i in range(rads.shape[0]):
        div = int(steps[i])
        
        step = int(steps[i] - 1)
        if step < 1: # for handling 180 turns
            step = 0
        
        axis_t = np.zeros((step, 3), dtype=np.float32)
        axis_t[:, 2] = axis[i][2]
        
        ticks = np.zeros(step, dtype=np.float32)
        ticks[:] = tick[i]
        accumulated = np.cumsum(ticks)
        
        w, qaxis = two_d_quat(accumulated, axis_t)

        co = np.zeros((ticks.shape[0], 3), dtype=np.float32)

        co[:] = (rps[i] - pivots[i])
        
        co = q_rotate(co, w, qaxis)
        co += rc.pivots[0::3][i]
        rc.radius_points.append(co)


def make_new_indices(rc):
    rc.radius_points
    lps = rc.pivots[1::3]
    rps = rc.pivots[2::3]
    le = rc.left_edges
    re = rc.right_edges

    sharp_idx = np.arange(rc.p.shape[0])[rc.sharp]
    bool = np.ones(rc.p.shape[0], dtype=np.bool)
    
    for i in range(lps.shape[0]):
        start = le[i][1] + 1
        stop = re[i][1] - 1
        count = (stop - start) + 1

        if stop >= start:    
            idx = np.linspace(start, stop, count, dtype=np.int32)

            count = (stop - start) + 1
            bool[idx] = False
        
        else:

            stop1 = rc.p.shape[0] - 1
            count = (stop1 - start) + 1
            idx = np.linspace(start, stop1, count, dtype=np.int32)

            bool[idx] = False
            
            start = 0
            count = (stop - start) + 1
            idx = np.linspace(start, stop, count, dtype=np.int32)

            bool[idx] = False
    
    new_p = []
    count = 0
    for i, tf in enumerate(bool):
        if tf:
            new_p.append(rc.p[i])
        
        if i in sharp_idx:

            new_p.append(lps[count][:2])
            for qp in rc.radius_points[count][::-1]:
                new_p.append(qp[:2])
            new_p.append(rps[count][:2])
            count += 1

    rc.p = np.array(new_p)


class RoundCorners():
    def __init__(self,
                 polyline,
                 radius=0.87,
                 step_angle=10.0,
                 error_margin=0.001,
                 angle_threshold=10,
                 merge_threshold=0.5,
                 straight_threshold=1):
        
        if radius == 0:
            self.rounded = polyline
            self.sharp = None
            return
                
        self.radius = radius
        self.polyline = polyline
        self.adjust_radius = False
        self.step_angle = step_angle
        self.radius_generate = True
        self.error_margin = error_margin
        self.angle_threshold = angle_threshold
        self.merge_threshold = merge_threshold
        self.straight_threshold = straight_threshold        
        
        # ---
        round_polyline_corners(self)        
        round(self)
        check_overlap(self)
        
        if self.adjust_radius:
            round_polyline_corners(self)
            round(self)
            check_overlap(self)

        # compensate for pivots moving along curve
        fix_curves(self)
        
        create_radius_points(self)
        make_new_indices(self)
        
        new_spacing = True
        if new_spacing:
            self.p = redistribute(self.p)[:, :2]
        
        ### Add the start point back to the end ----------
        self.polyline = self.p.tolist()
        self.polyline += [self.polyline[0]]
        self.rounded = [self.polyline]


def rounded_polyline(cut_polyline):

    rc = RoundCorners(cut_polyline, # The original polyline
                      radius=0.87, # Radius of the curve
                      step_angle=10, # Degrees between radius points
                      angle_threshold=10, # Identify sharp corners by degrees
                      merge_threshold=0.5, # For removing doubles
                      straight_threshold=1) # if the turn is nearly 180 skip rounding (otherwise the radius pivot is an infinite distance from the corner)
    
    rounded = rc.rounded
    return rounded


def smoothed(ar, zeros, iters, reduce_only=False):
    """Reduce only based on theory that
    we only want to reduce vel in sharp
    turns not allowing vel to increase
    where we may still be turning."""
    end = ar.shape[0] - 1
    
    sh = ar.shape[0]
    idx1 = np.empty((sh, 3), dtype=np.int32)
    
    id = np.arange(sh)
    
    idx1[:, 0] = id - 1
    idx1[:, 1] = id
    idx1[:, 2] = id + 1
    idx1[idx1 < 1] = 0
    idx1[idx1 > (sh - 1)] = sh - 1
    means1 = np.mean(ar[idx1], axis=1)
    
    for k in range(iters):
        means1 = np.mean(ar[idx1], axis=1)
        move = (np.array(means1) - ar)
        if reduce_only:    
            move[move > 0] = 0.0
        ar += move    
        ar[zeros] = 0.0

    return ar        
        

class GeneratePVT:
    def __init__(self, cut_polyline, max_vel=100.0, smooth_iters=500):
        
        cp = cut_polyline[0]
        
        pl = np.array(cp)
        
        xV = np.ones(pl.shape[0], dtype=np.float32)
        yV = np.ones(pl.shape[0], dtype=np.float32)
        xV[0] = 0.0
        xV[-1] = 0.0
        yV[0] = 0.0
        yV[-1] = 0.0

        xswitch = np.ones(pl.shape[0], dtype=np.bool)
        yswitch = np.ones(pl.shape[0], dtype=np.bool)

        vecs = pl[1:] - pl[:-1]
        length = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
        uvecs = vecs / length[:, None]
        self.uvecs = uvecs
        
        sign = np.sign(uvecs[1:] * uvecs[:-1])
        xswitch[1:-1] = sign[:, 0] == -1
        yswitch[1:-1] = sign[:, 1] == -1
        # -------------------------
        
        xV[1:] = np.abs(uvecs[:, 0])
        yV[1:] = np.abs(uvecs[:, 1])
            
        smxV = smoothed(xV, xswitch, smooth_iters)
        smyV = smoothed(yV, yswitch, smooth_iters)
        
        smxV[-1] = 0
        smyV[-1] = 0
        
        averageXV = (smxV[0:-1] + smxV[1:]) / 2
        averageYV = (smyV[0:-1] + smyV[1:]) / 2
                
        # final ----------------------------
        self.max_vel = max_vel
        self.xP = np.append(np.array([0.0], dtype=np.float32), vecs[:, 0])
        self.yP = np.append(np.array([0.0], dtype=np.float32), vecs[:, 1])
        self.xV = smxV * max_vel
        self.yV = smyV * max_vel

        self.xAC = self.xV[1:] - self.xV[:-1]
        self.xAC = np.append(np.array([0.0]), self.xAC)

        self.yAC = self.yV[1:] - self.yV[:-1]
        self.yAC = np.append(np.array([0.0]), self.yAC)

        self.xJ = np.abs(self.xAC[1:] - self.xAC[:-1])
        self.xJ = np.append(np.array([0.0]), self.xJ)

        self.yJ = np.abs(self.yV[1:] - self.yV[:-1])
        self.yJ = np.append(np.array([0.0]), self.yJ)
        
        self.ti = np.zeros(pl.shape[0], dtype=np.float32)
        combined = np.sqrt((averageXV ** 2) + (averageYV ** 2))

        self.ti[1:] = length / combined / max_vel
        
        # head speed
        self.hs = np.zeros(pl.shape[0], dtype=np.float32)
        #self.hs[1:] = length / self.ti[1:]
        self.hs[1:] = combined * max_vel
    

def vel_interp(pvt):
    """Creates extra animation frames based on
    velocity at each point so the animation shows
    changes in speed."""
    p = pvt.p
    vel = pvt.hs
    
    p_acc = p[0:1]
    v_acc = vel[0]
    xV = pvt.xV[0]
    yV = pvt.yV[0]
    xJ = pvt.xJ[0]
    yJ = pvt.yJ[0]
    ti = pvt.ti[0]
    xP = pvt.xP[0]
    yP = pvt.yP[0]

    top = np.max(vel)
    bottom = np.min(vel[vel > 0.00001])
    dif = top - bottom
    spread = 10
    val = dif / spread
    
    for i, j, in enumerate(p):
        if i == 0:
            continue
        
        num = (((top - vel[i]) * val) // 50) + 2
        #print(num, vel[i], top, bottom)
        
        lx = np.linspace(p[i - 1][0], p[i][0], int(num))[1:]
        ly = np.linspace(p[i - 1][1], p[i][1], int(num))[1:]
        ar = np.empty((lx.shape[0], 2), dtype=np.float32)
        
        new_v = np.linspace(vel[i - 1], vel[i], int(num))[1:]
        v_acc = np.append(v_acc, new_v)

        new_xV = np.linspace(pvt.xV[i - 1], pvt.xV[i], int(num))[1:]
        xV = np.append(xV, new_xV)        

        new_yV = np.linspace(pvt.yV[i - 1], pvt.yV[i], int(num))[1:]
        yV = np.append(yV, new_yV)        

        new_xJ = np.linspace(pvt.xJ[i - 1], pvt.xJ[i], int(num))[1:]
        xJ = np.append(xJ, new_xJ)

        new_yJ = np.linspace(pvt.yJ[i - 1], pvt.yJ[i], int(num))[1:]
        yJ = np.append(yJ, new_yJ)

        new_ti = np.linspace(pvt.ti[i - 1], pvt.ti[i], int(num))[1:]
        ti = np.append(ti, new_ti)

        # don't really need to show interpolated P polyline.
        # It's just for verifying the P value in PVT
        #new_xP = np.linspace(pvt.xP[i - 1], pvt.xP[i], int(num))[1:]
        #xP = np.append(xP, new_xP)

        #new_yP = np.linspace(pvt.yP[i - 1], pvt.yP[i], int(num))[1:]
        #yP = np.append(yP, new_yP)

        ar[:, 0] = lx
        ar[:, 1] = ly
        
        p_acc = np.append(p_acc, ar, axis=0)

    pvt.xV = xV
    pvt.yV = yV
    pvt.xJ = xJ
    pvt.yJ = yJ
    pvt.ti = ti
    #pvt.xP = xP 
    #pvt.yP = yP
    pvt.hs = v_acc
    pvt.p = p_acc
    
    return


def manage_pvt(pvt):

    xp = pvt.xP
    yp = pvt.yP

    #if hasattr(pvt, 'uvecs'):
    if False:
        uvecs = np.append(np.array([[0.0, 0.0]], dtype=np.float32), pvt.uvecs, axis=0)
    
    else:    
        #vecs = np.zeros((xp.shape[0], 2), dtype=np.float32)
        #vecs[:, 0] = xp
        #vecs[:, 1] = yp
        vecs = pvt.p[1:] - pvt.p[:-1]
        vecs = np.append(np.array([[0.0, 0.0]], dtype=np.float32), vecs, axis=0)
        length = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
        uvecs = vecs / length[:, None]
        pvt.length = length

    xv = pvt.xV * pvt.ti * pvt.max_vel
    yv = pvt.yV * pvt.ti * pvt.max_vel
    vel_line = np.zeros((xv.shape[0], 2), dtype=np.float32)
    uxv = pvt.xP * xv
    uyv = pvt.yP * yv
    vel_line[:, 0] = np.nan_to_num(uxv)
    vel_line[:, 1] = np.nan_to_num(uyv)
    vel_line = np.cumsum(vel_line, axis=0)
    
    p_line = np.zeros((xp.shape[0], 2), dtype=np.float32)
    p_line[:, 0] = np.cumsum(xp)
    p_line[:, 1] = np.cumsum(yp)
    
    return p_line, vel_line
    

def offset_p_line(p):
    """Scales and moves the poly lines
    So they fit in the frame."""
    
    bottom = np.min(p, axis=0)
    p -= bottom
    m = np.max(p)
    scale = 200 / m
    p *= scale
    
    lb = np.min(p) # left bottom corner
    tr = np.max(p) # top right corner
    fudge = (tr - lb) * 0.1
    
    #bc = lb - fudge
    bc = 0.0
    tc = tr + fudge * 3
    tcf = tc - fudge * 0.2
    bcf = bc + fudge * 0.2
    
    # move the figure to the right side
    x_shift = tcf - np.max(p[:, 0])
    y_shift = bcf - np.min(p[:, 1])
    p[:, 0] += x_shift
    p[:, 1] += y_shift
    
    return p, bc, bcf, tc, tcf, fudge


def create_anim(cut_polyline=None, pvt=None, save_path=None, save_name=None, save_type='mp4', skip_frames=20):
    """Save type as 'gif' or 'mp4' """
    
    # for testing
    if cut_polyline is None:
        cut_polyline = test_poly()
        p = np.array(cut_polyline[0])
        import generate_pvt
        import imp
        imp.reload(generate_pvt)
        pvt = generate_pvt.GeneratePVT(cut_polyline, smooth_iters=500)
    
    else:
        p = np.array(cut_polyline[0])    
    
    pvt.p = p
    
    cp_p_line, cp_vel_line = manage_pvt(pvt)
    cp_p_line = offset_p_line(cp_p_line)[0]
    cp_vel_line = offset_p_line(cp_vel_line)[0]

    vel_interp(pvt)
    p = pvt.p
    
    p, bc, bcf, tc, tcf, fudge = offset_p_line(p)
        
    interpolate = False
    if interpolate:
        new_ar = spread_array(p, 5)
    else:
        new_ar = p
    
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    fig.set_size_inches(8, 8, forward=True)
    fig.set_dpi(100)
    ax.set_xlim((bc, tc))
    ax.set_ylim((bc, tc))

    # lines ---------------------------------
    line, = ax.plot([], [], lw=4.1, label='cut_polyline', color='deeppink') # cut_polyline
    p_line, = ax.plot([], [], lw=2.1, label='P polyline', color='yellow') # cut_polyline
    vel_line, = ax.plot([], [], lw=.5, label='vel * time', color='black') # cut_polyline
    xs_line, = ax.plot([], [], lw=5.0, label='x velocity', color='red') # x speed
    ys_line, = ax.plot([], [], lw=5.0, label='y velocity', color='greenyellow') # x speed
    hs_line, = ax.plot([], [], lw=5.0, label='head speed', color='dodgerblue') # x speed
    xj_line, = ax.plot([], [], lw=5.0, label='x jerk', color='crimson') # x speed
    yj_line, = ax.plot([], [], lw=5.0, label='y jerk', color='gold') # x speed
    time_line, = ax.plot([], [], lw=5.0, label='time * 20', color='purple') # x speed

    # points --------------------------------
    cp_point, = ax.plot([0], [0], 'go', color='deeppink') # cut_polyline
    p_point, = ax.plot([0], [0], 'go', color='yellow') # cut_polyline
    v_point, = ax.plot([0], [0], 'go', color='black') # cut_polyline
    x_point, = ax.plot([0], [0], 'go', color='red')    
    y_point, = ax.plot([0], [0], 'go', color='greenyellow')
    # ---------------------------------------

    ax.legend(loc=2)  # Add a legend.
    
    idexer = np.arange(p.shape[0])[::skip_frames] # skip frames to save animation time
    def animate(i):
        i = idexer[i]
        # poly lines
        line.set_data(new_ar[:, 0], new_ar[:, 1])
        p_line.set_data(cp_p_line[:, 0], cp_p_line[:, 1])
        vel_line.set_data(cp_vel_line[:, 0], cp_vel_line[:, 1])

        # bar graphs
        q = np.array([[bcf, 0.0],[bcf, 1.2]])
        r = np.array([[bcf * 2, 0.0],[bcf * 2, 1.2]])
        s = np.array([[bcf * 4, 0.0],[bcf * 4, 1.2]])
        t = np.array([[bcf * 6, 0.0],[bcf * 6, 1.2]])
        u = np.array([[bcf * 7, 0.0],[bcf * 7, 1.2]])
        v = np.array([[bcf * 9, 0.0],[bcf * 9, 1.2]])

        q[1][-1] = pvt.xV[i]
        r[1][-1] = pvt.yV[i]
        s[1][-1] = pvt.hs[i]
        t[1][-1] = pvt.xJ[i]
        u[1][-1] = pvt.yJ[i]
        v[1][-1] = pvt.ti[i] * 10

        xs_line.set_data(q[:, 0], q[:, 1])
        ys_line.set_data(r[:, 0], r[:, 1])
        hs_line.set_data(s[:, 0], s[:, 1])
        xj_line.set_data(t[:, 0], t[:, 1])
        yj_line.set_data(u[:, 0], u[:, 1])
        time_line.set_data(v[:, 0], v[:, 1])

        # points        
        cp_point.set_data(new_ar[:,0][i], new_ar[:,1][i])
        #p_point.set_data(cp_p_line[:,0][i], cp_p_line[:,1][i])
        #v_point.set_data(cp_vel_line[:,0][i], cp_vel_line[:,1][i])
        x_point.set_data(new_ar[:,0][i], [bc + fudge * 0.1])
        y_point.set_data([tc - fudge * 0.1], new_ar[:,1][i])
        
        return (line,
                p_line,
                vel_line,
                xs_line,
                ys_line,
                hs_line,
                xj_line,
                yj_line,
                time_line,
                cp_point,
                p_point,
                v_point,
                x_point,
                y_point,
                )

    anim = animation.FuncAnimation(fig, animate, frames=idexer.shape[0], interval=20, blit=True)
    path = save_path
    name = save_name

    if path is None:
        path = os.path.expanduser('~/Desktop/')
    if save_name is None:
        name = 'PVT_movie'
    
    if save_type == 'gif':
        anim.save(path + name + '.gif', writer='imagemagick', fps=60)    

    if save_type == 'mp4':
        anim.save(path + name + '.mp4',writer=animation.FFMpegWriter(fps=60))

    #plt.show()
    plt.close(fig=None)


def test_poly(pidx=777):
    """Can load an np.savetxt file
    for the cut_polyline"""
    import pathlib

    path = os.path.expanduser("~/Desktop")
    mc_path = pathlib.Path(path).joinpath('MC_cache_files')
    idx = np.arange(20)
    for i in idx:
        if False:    
            if i != pidx:
                continue

        txt = mc_path.joinpath(str(i))
        pl = np.loadtxt(txt).tolist()
        cut_polyline = [pl]
    
        pvt_from_polyline(cut_polyline, create_animation=True, save_path=path + '/pvt_movies/', save_name=str(i))
    
    return cut_polyline


# call this function with polyline    
def pvt_from_polyline(cut_polyline, max_velocity=200, create_animation=False, save_path=None, save_name=None):
    if cut_polyline is None:
        cut_polyline = test_poly()
    
    rounded = rounded_polyline(cut_polyline)
    pvt = GeneratePVT(rounded, max_vel=max_velocity, smooth_iters=500)
    
    # Path for saving video:
    if create_animation:
        skip_frames = 50 # save every Nth frame in video
        save_type = 'mp4' # or 'gif' but 'gif' takes forever to save
        create_anim(rounded, pvt, save_path=save_path, save_name=save_name, save_type=save_type, skip_frames=skip_frames)
    
    # currently all numpy arrays, float32
    pvt.xP # delta move on x
    pvt.yP # delta move on y
    pvt.xV # x velocity
    pvt.yV # y velocity
    pvt.ti # time between points (same for x and y)
    
    return pvt

test_call = False
if test_call: # will save "pvt_move.mp4" to desktop by default
    pvt_from_polyline(cut_polyline,
                      max_velocity=200,
                      create_animation=True,
                      save_path=None,
                      save_name=None)
