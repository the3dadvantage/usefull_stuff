import numpy as np

try:
    import bpy
except:
    pass


def final_cut_(existing_points, max_depth=10, min_distance=2, straightness=0.5):
    divisions = 16
    
    ep = existing_points
    
    # could get left and right max and add padding
    #   
    ls = np.min(ep[:, 0]) - 0.0001
    rs = np.max(ep[:, 0]) + 0.0001
    
    spread = np.linspace(ls, rs, divisions)
    
    idx = np.searchsorted(spread, ep[:, 0])
    numidx = np.arange(idx.shape[0])
    nums = []
    peaks = []
    for i in range(1, divisions):    
        slice = ep[idx==i]
        if slice.shape[0] == 0:
            peak = peaks[-1]
            nums.append(nums[-1])
            peaks.append(peaks[-1])
        else:    
            peak = np.max(slice[:, 1])
            peaks.append(peak)
            apeak = np.argmax(slice[:, 1])
            num = numidx[idx==i][apeak]
            nums.append(num)
        #bool[idx==i] = True        
    
    return idx, slice, nums
    
    
    # questions:
    # 1. how do we know the left and right start and
    #   stop locations for the laser?
    
    pass    
    
    
    
    
    
a = bpy.data.objects['a']
a = bpy.data.objects['b']
a = bpy.data.objects['p']

co = np.zeros((len(a.data.vertices), 3), dtype=np.float32)
a.data.vertices.foreach_get('co', co.ravel())

idx, slice, peaks = final_cut_(co[:, :2])

bool = np.zeros(idx.shape[0], dtype=np.bool)
#print(idx==0)
bool[peaks] = True
#bool[idx==5] = True
#print(slice)
#for i in range(1, 12):    
    #bool[idx==i] = True    
    
#a.data.vertices.foreach_set('select', bool)
#a.data.update()


def double_check(ep, lp, md, k, iters):
    """Checks if any edges dip below the pattern
    and moves them up."""
    #return
    dists = np.zeros(lp.shape, dtype=np.float32)
    middle = lp[:, 0][1:-1]
    ls = middle - lp[:, 0][:-2]
    rs = lp[:, 0][2:] - middle
    
    dists[:, 0][1:-1] = ls
    dists[:, 1][1:-1] = rs
    dists[0][0] = ls[0]
    dists[-1][0] = rs[-1]
    dis = np.sort(dists, axis=0)[:, 1]
    x_dif = lp[:, 0][:, None] - ep[:, 0]
    bool = np.abs(x_dif) < dis[:, None]
    
    for i in range(lp.shape[0]):
        idx = np.arange(ep.shape[0])[bool[i]]
        ey = ep[:, 1][idx]
        ly = lp[i][1] - md
        
        mult = 1
        if iters > 10:    
            if k > iters - 3:
                mult = 0
        #if k == iters - 1:
            #mult == 0
        
        if np.any(ey > ly):
            t = np.max(ey)
            lp[i][1] += ((t-ly) + md) * mult


def smoothed(ar, ep, zeros, md, iters, reduce_only=False):
    """Reduce only based on theory that
    we only want to reduce vel in sharp
    turns not allowing vel to increase
    where we may still be turning."""

    # get weights so sharp angles move less
    vecs = ar[1:] - ar[:-1]
    length = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
    uvecs = vecs / length[:, None]
    d = np.einsum('ij,ij->i', uvecs[1:], uvecs[:-1])

    d[d > 1] = 1
    d[d < 0] = 0
    weights = np.ones(ar.shape[0], dtype=np.float32)
    #weights[1:-1] = d# ** 1.5

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
            #move *= weights[:, None]
            pass
            #move[move < 0] = 0.0
        ar[:,1] += move[:, 1]    
        ar[zeros] = 0.0
        
        #if k < iters - 4:    
        double_check(ep, ar, md, k, iters)
        
    return ar
    

def pad_ends(ar, overcut=4):
    
    ls = ar[0][0]
    rs = ar[-1][0]
    
    safe_margin = 10
    overlap_size = 30 # area to go back and forth
    
    pad = 20
    
    # start = start_location
    start = ls - safe_margin - overlap_size - pad
    # end = end_location    
    end = rs + safe_margin + overlap_size + pad
    
    ls_start = ls - safe_margin - 

# or: for i in range(N):
#         walk(i)

#!!! Make args to pad the edges so like:
#    edge_dist = 10 # centimeters
#!!! make args for number of times to go
#    back and forth over the edge where it curls up
#!!! Put in the args for max depth and stuff...

def final_cut(existing_points=co, max_depth=350, min_distance=10, straightness=0.0):
    "straightness must be from 0.0 - 1.0, 1.0 is a straight line"
    smoothing_iters = 20 
    md = min_distance

    flat =.5
    s = flat

    convert = straightness * 200    
    
    
    
    divisions = int((200 - convert)) + 10
    ep = existing_points
    
    ab_max = np.max(ep[:, 1])
    bottom = ab_max - max_depth
    
    ls = np.min(ep[:, 0]) - 0.0001
    rs = np.max(ep[:, 0]) + 0.0001
    
    spread = np.linspace(ls, rs, divisions)
    
    f = bpy.context.scene.frame_current - 1
    
    #print(spread[f])
    
    e = bpy.data.objects['e']
    e2 = bpy.data.objects['e2']
    dir = -1
    look = (rs - ls) / divisions
    drop = look * s * dir
    vec = np.array([look * s, drop])

    idx = np.searchsorted(spread, ep[:, 0])
    numidx = np.arange(idx.shape[0])
    nums = []
    peaks = []
    vecs = [0.0]
    expand = 9
    epeaks = []
    xy = []
    for i in range(divisions):
        b = i
        if b > 4:
            b = 5
        slice = ep[np.in1d(idx, np.arange(i-b, i + expand))]
        #slice = ep[np.in1d(idx, np.arange(i, i + expand))]
        if slice.shape[0] == 0:
            peak = peaks[-1] + vecs[-1]
            vec = vecs[-1]
            if peak < bottom:
                peak = bottom
                vec = bottom - vec
            vecs.append(vec)
            #peaks.append(peak)
        else:
            peak = np.max(slice[:, 1]) + md
            if peak < bottom:
                peak = bottom
            #peaks.append(peak)
        
        if i > 0:
            dif = peak - peaks[i - 1]
            vec = dif * flat
            vecs.append(vec)
            peak = peaks[i - 1] + vec
        
        # edges go back and forth where it curls
        # number of edge passes. how wide an area.
        # edge width var
        peaks.append(peak)    
            
        xy += [[spread[i], peaks[i]]]
        
    npxy = np.array(xy)
    zeros = np.zeros(npxy.shape[0], dtype=np.bool)
    sm_ar = smoothed(npxy, ep, zeros, md=min_distance, iters=smoothing_iters, reduce_only=True)

    back_forth = pad_ends(sm_ar)

    line = bpy.data.objects['l']
    lco = np.zeros((len(line.data.vertices), 3), dtype=np.float32)
    line.data.vertices.foreach_get('co', lco.ravel())

    lco[:, :2][:len(xy)] = sm_ar
    line.data.vertices.foreach_set('co', lco.ravel())
    line.data.update()
    
    return [sm_ar.tolist()] # formatted like cut_polyline
        

cut_polyline = final_cut()


