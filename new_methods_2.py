import bpy
import numpy as np
from numpy import newaxis as nax


def get_uv_index_from_3d(ob):
    """Creates a two dimensional including where each vert
    occurs in the uv layers. Second dimension is N size so not numpy"""
    # figure out every index where the 3d verts occur in the uv maps
    obm = get_bmesh(ob)
    obm.verts.ensure_lookup_table()
    obm.faces.ensure_lookup_table()
    
    # currently works on a selected set of verts
    selected_verts = get_selected_verts(ob)
    sel_idx = np.arange(selected_verts.shape[0])[selected_verts]
    
    indexed_sum = []
    cum_sum = 0
    for i in obm.faces:
        indexed_sum.append(cum_sum)
        cum_sum += len(i.verts)
    
    v_sets = []
    for i in sel_idx:
        uv = []
        for f in obm.verts[i].link_faces:
            vidx = np.array([v.index for v in f.verts])
            idx = (np.arange(vidx.shape[0])[vidx == i])[0]
            uv.append(idx + indexed_sum[f.index])
        v_sets.append(uv)    
    # v_sets is now the uv index of each vert wherever it occurs in the uv map
    # v_sets[0] is vert zero and [uv[5], uv[390], uv[25]] or something like that
    # v_sets[0] looks like [5, 390, 16]
    return v_sets
    

def grid_sample(ob, box_count=10, offset=0.00001):
    """divide mesh into grid and sample from each segment.
    offset prevents boxes from excluding any verts"""
    co = get_co(ob)
    
    # get bounding box corners
    min = np.min(co, axis=0)
    max = np.max(co, axis=0)
    
    # box count is based on largest dimension
    dimensions = max - min
    largest_dimension = np.max(dimensions)
    box_size = largest_dimension / box_count
    
    # get box count for each axis
    xyz_count = dimensions // box_size # number of boxes on each axis
    
    # number of boxes on each axis:
    box_dimensions = dimensions / xyz_count # each box is this size
    
    line_end = max - box_dimensions # we back up one from the last value
    
    x_line = np.linspace(min[0], line_end[0], num=xyz_count[0], dtype=np.float32)
    y_line = np.linspace(min[1], line_end[1], num=xyz_count[1], dtype=np.float32)
    z_line = np.linspace(min[2], line_end[2], num=xyz_count[2], dtype=np.float32)
    
    idxer = np.arange(co.shape[0])
    
    # get x bools
    x_grid = co[:, 0] - x_line[:,nax]
    x_bools = (x_grid + offset > 0) & (x_grid - offset < box_dimensions[0])
    cull_x_bools = x_bools[np.any(x_bools, axis=1)] # eliminate grid sections with nothing
    xb = cull_x_bools

    x_idx = np.tile(idxer, (xyz_count[0], 1))
    
    samples = []
    
    for boo in xb:
        xidx = idxer[boo]
        y_grid = co[boo][:, 1] - y_line[:,nax]
        y_bools = (y_grid + offset > 0) & (y_grid - offset < box_dimensions[1])
        cull_y_bools = y_bools[np.any(y_bools, axis=1)] # eliminate grid sections with nothing
        yb = cull_y_bools
        for yboo in yb:
            yidx = xidx[yboo]
            z_grid = co[yidx][:, 2] - z_line[:,nax]
            z_bools = (z_grid + offset > 0) & (z_grid - offset < box_dimensions[2])
            cull_z_bools = z_bools[np.any(z_bools, axis=1)] # eliminate grid sections with nothing
            zb = cull_z_bools        
            for zboo in zb:
                samples.append(yidx[zboo][0])
            
                # !!! to use this for collisions !!!:
                if False:    
                    samples.extend(yidx[zboo])
    
    return np.unique(samples) # if offset is zero we don't need unique... return samples


def edge_to_edge(e1, e2, e3, e4 ):
    """Takes two edges defined by four vectors.
    Returns the two points that describe the shortest
    distance between the two edges. The two points comprise
    a segment that is orthagonal to both edges."""
    v1 = e2 - e1
    v2 = e3 - e4
    v3 = e3 - e1
    cross = np.cross(v1, v2)
    d = (v3 @ cross) / (cross @ cross)
    spit = cross * d # spit because if you stand on cp1 and spit this is where it lands.
    cp1 = e1 + spit
    vec2 = cp1 - e3
    d = (vec2 @ v2) / (v2 @ v2)
    nor = v2 * d
    cp2 = e3 + nor 
    normal = cp1 - cp2
    or_vec = e1 - cp2
    e_dot = normal @ v1
    e_n_dot = normal @ or_vec
    scale = e_n_dot / e_dot  
    p_on_p =  (or_vec - v1 * scale) + cp2
    return p_on_p, p_on_p + spit


def curve_gen(scalars, type=0, height=1):
    """Takes points between zero and 1 and plots them on a curve"""
    if type == 0: # smooth middle
        mid = scalars ** 2
        mid_flip = (-scalars + 1) ** 2
        return mid * mid_flip * 16 * height
        
    if type == 1: # half circle
        return np.sqrt(scalars) * np.sqrt(-scalars + 1)
    
    if type == 2: # smooth bottom to top
        reverse = -scalars + 1
        c1 = (scalars ** 2) * reverse
        c2 = (-reverse ** 2 + 1) * scalars
        smooth = c1 + c2
        return smooth

    if type == 3: # smooth top to bottom flip
        reverse = -scalars + 1
        c1 = (scalars ** 2) * reverse
        c2 = (-reverse ** 2 + 1) * scalars
        smooth = c1 + c2
        return -smooth + 1

    if type == 4: # 1/4 circle top left
        return np.sqrt(-scalars + 2) * np.sqrt(scalars)

    if type == 5: #1/4 circle bottom right
        x = np.sqrt(-scalars + 1) * np.sqrt(scalars + 1)
        return -x + 1

    if type == 6: #1/4 circle bottom left
        return -(np.sqrt(-scalars + 2) * np.sqrt(scalars)) + 1

    if type == 7: #1/4 circle top right
        x = np.sqrt(-scalars + 1) * np.sqrt(scalars + 1)
        return x 


def get_selected_poly_verts(ob):
    """returns a list of lists of verts in each selected polygon.
    Works in any mode."""
    if ob.type != "MESH":
        return []

    if ob.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(ob.data)
        return [[v.index for v in f.verts] for f in bm.faces if f.select]

    return [[i for i in p.vertices] for p in ob.data.polygons if p.select]


def get_poly_verts(ob):
    """returns a list of lists of verts in each polygon.
    Works in any mode."""
    if ob.type != "MESH":
        return []

    if ob.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(ob.data)
        return [[v.index for v in f.verts] for f in bm.faces]

    return [[i for i in p.vertices] for p in ob.data.polygons]


def get_eidx():
    ec = len(ob.data.edges)
    ed = np.zeros(ec * 2, dtype=np.int32)
    ob.data.edges.foreach_get('vertices', ed)
    ed.shape = (ec, 2)
    return ed


def select_all(ob, select=False):
    """Fast select/deselect in object mode"""
    atts = [ob.data.vertices, ob.data.edges, ob.data.polygons]
    fun = np.zeros
    if select:
        fun = np.ones
    for att in atts:
        c = len(att)
        arr = fun(c, dtype=np.bool)
        att.foreach_set('select', arr)
    

def hide_all(ob, hide=False):
    """Fast hide/unhide in object mode"""
    atts = [ob.data.vertices, ob.data.edges, ob.data.polygons]
    fun = np.zeros
    if hide:
        fun = np.ones
    for att in atts:
        c = len(att)
        arr = fun(c, dtype=np.bool)
        att.foreach_set('hide', arr)


def coincident_points(group_a, group_b, threshold=.0001, inverse=True):
    """finds the index of points in group a that match the location of at
    least one point in group b. Returns the inverse by default: points that have no match
    returns a bool array matching the first dimension of group_a"""
    x = group_b - group_a[:, nax]
    dist = np.einsum('ijk, ijk->ij', x, x)
    min_dist = np.min(dist, axis=1)
    if inverse:
        return min_dist > threshold
    return min_dist < threshold


def remove_doubles(group, threshold=.0001):
    """finds coincident points and returns a bool array eliminating all but the first
    occurance of the coincident points"""
    x = group - group[:, nax]
    dist = np.einsum('ijk, ijk->ij', x, x)
    pairs = dist < threshold
    doubles = np.sum(pairs, axis=0) > 1
    idx = np.arange(len(group))[doubles]
    all_true = np.ones(len(group), dtype=np.bool)
    for i in idx:
        this = np.all((group[i] - group[idx]) == 0, axis=1)
        all_true[idx[this][1:]] = False
    return all_true


def get_quat(rad, axis):
    u_axis = axis / np.sqrt(axis @ axis)
    theta = (rad * 0.5)
    w = np.cos(theta)
    q_axis = u_axis * np.sin(theta)
    return w, q_axis

 
def get_quat_2(v1, v2, rot_mag=1, convert=True, axis=None):
    """Returns the quaternion that will rotate the object based on two vectors.
    If axis is provided: it can be non-unit. The quaternion will rotate around the axis
    until v1 lines up with v2 on the axis    
    To rotate part way rot mag can be a value between -1 and 1.
    For specific angles, rot_mag can be the cosine of the angle in radians (haven't tested this theory)"""  
    if convert: # if vectors are non-unit
        v1 = v1 / np.sqrt(v1 @ v1)
        v2 = v2 / np.sqrt(v2 @ v2)
    if axis is None:
        mid = v1 + v2 * rot_mag # multiply for slerping  
        Umid = mid / np.sqrt(mid @ mid)
        w = Umid @ v1
        xyz = np.cross(v1, Umid)
        return w, xyz
    vc1 = np.cross(axis, v1)        
    vc2 = np.cross(axis, v2) 
    v1 = vc1 / np.sqrt(vc1 @ vc1)
    v2 = vc2 / np.sqrt(vc2 @ vc2)
    mid = v1 + v2 * rot_mag # multiply for slerping  
    Umid = mid / np.sqrt(mid @ mid)
    w = Umid @ v1
    xyz = np.cross(v1, Umid)
    return w, xyz


def get_quat_from_perp_vecs(v1, v2):
    x = np.array([1, 0, 0])
    z = np.array([0, 0, 1])
    uv1 = v1 / np.sqrt(v1 @ v1)
    norm = np.cross(v1, v2)
    uv3 = norm / np.sqrt(norm @ norm)
    w1, axis1 = get_quat_2(uv3, z)
    rot = q_rotate(uv1, w1, axis1)
    w2, axis2 = get_quat_2(rot, x)
    n_w, n_xyz = quaternion_add(-w1, axis1, -w2, axis2)


def q_rotate(co, w, axis):
    """Takes an N x 3 numpy array and returns that array rotated around
    the axis by the angle in radians w. (standard quaternion)"""    
    move1 = np.cross(axis, co)
    move2 = np.cross(axis, move1)
    move1 *= w
    return co + (move1 + move2) * 2


def quaternion_subtract(w1, v1, w2, v2):
    """Get the quaternion that rotates one object to another"""
    w = w1 * w2 - np.dot(v1, v2)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return w, -v


# -------------------------------------------->>>

def get_co(ob, arr=None, key=None): # key
    """Returns vertex coords as N x 3"""
    c = len(ob.data.vertices)
    if arr is None:    
        arr = np.zeros(c * 3, dtype=np.float32)
    if key is not None:
        ob.data.shape_keys.key_blocks[key].data.foreach_get('co', arr.ravel())        
        arr.shape = (c, 3)
        return arr
    ob.data.vertices.foreach_get('co', arr.ravel())
    arr.shape = (c, 3)
    return arr


def get_proxy_co(ob, arr):
    """Returns vertex coords with modifier effects as N x 3"""
    me = ob.to_mesh(bpy.context.scene, True, 'PREVIEW')
    c = len(me.vertices)
    me.vertices.foreach_get('co', arr.ravel())
    bpy.data.meshes.remove(me)
    arr.shape = (c, 3)
    return arr


def apply_transforms(ob, co=None):
    """Get vert coords in world space"""
    if co is None:
        co = get_co(ob)
    m = np.array(ob.matrix_world)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


def revert_transforms(ob, co):
    """Set world coords on object. 
    Run before setting coords to deal with object transforms
    if using apply_transforms()"""
    m = np.linalg.inv(ob.matrix_world)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc  


def closest_points_edge(vec, origin, p):
    '''Returns the location of the point on the edge'''
    vec2 = p - origin
    d = np.einsum('j,ij->i', vec, vec2) / (vec @ vec)
    cp = origin + vec * d[:, nax]
    return cp, d


def cp_scalar(vec, origin, p, unitize=False):
    '''Returns the dot that would put the point on the edge.
    Useful for sorting the order of verts if they were
    projected to the closest point on the edge'''
    vec2 = p - origin
    if unitize:
        vec2 = vec2 / np.sqrt(np.einsum('ij,ij->i', vec2, vec2))[:, nax]
    d = np.einsum('j,ij->i', vec, vec2)
    return d


def in_line_bounds(vec, origin, p):
    '''Returns returns a bool array indicating if points
    are in the range of the start and end of a vector'''
    vec2 = p - origin
    d = np.einsum('j,ij->i', vec, vec2)
    vd = vec @ vec 
    bool = (d > 0) & (d < vd)    
    return bool


def loop_order(ob):
    """takes an object consisting of a single loop of edges and gives the order"""
    obm = get_bmesh(ob)
    obm.edges.ensure_lookup_table()
    e = obm.edges[0]
    v = e.verts[0]
    order = []
    for i in range(len(obm.edges)):
        other = e.other_vert(v)
        order.append(other.index)
        e = [ed for ed in v.link_edges if ed != e][0]
        v = [ve for ve in e.verts if ve != other][0]
    return order


def circular_order(co, v1, v2, center=None, edges=False, convex=False, normal=None):
    """Return an array that indexes the points in circular order.
    v1 and v2 must be perpindicular and their normal defines the axis.
    if edges is True, return the edges to connect the points"""
    if co.shape[0] is 0:
        return
    #if center is None:
    center = np.mean(co, axis=0)
    if convex:
        center_vecs = co - center
        center_dots = np.einsum('ij,ij->i', center_vecs, center_vecs)
        max = np.argmax(center_dots)
        out_vec = center_vecs[max]
        cross = np.cross(out_vec, normal)
        con_set = [max]
        point = max 
        
        for i in range(co.shape[0]):
            spread = co - co[point]
            h = np.einsum('ij,ij->i', spread, spread)    
            Uspread = np.nan_to_num(spread / np.sqrt(h)[:, nax])
            dots = np.einsum('j,ij->i', cross, Uspread)
            new = np.argmax(dots)
            if new == point:
                new = np.argsort(dots)[-2] # for when the point gets narcissistic and finds itself
            if new == max:
                break

            con_set.append(new)
            cross = co[new] - co[point]
            point = new
 
        idxer = np.arange(len(con_set))
        eidx = np.append([idxer],[np.roll(idxer, -1)], 0).T       

        return con_set, eidx, center
    
    count = co.shape[0]
    idxer = np.arange(count)
    on_p1, center_vecs = cp_scalar(v1, center, co, False, True) # x_vec off center
    pos_x = on_p1 > 0
    co_pos = co[pos_x]
    co_neg = co[-pos_x]
    p_on_p2 = cp_scalar(v2, center, co_pos, True)
    n_on_p2 = cp_scalar(v2, center, co_neg, True)
    p_y_sort = np.argsort(p_on_p2)
    n_y_sort = np.argsort(n_on_p2)
    order = np.append(idxer[pos_x][p_y_sort], idxer[-pos_x][n_y_sort][::-1])
            
    idxer = np.arange(len(order))
    eidx = np.append([idxer],[np.roll(idxer, -1)], 0).T

    return order, eidx


# Can get a speedup reusing the existing array
def get_att(att, name, dim2=None, vecs=None, dtype=None, proxy=None, shape=None):
    """Returns a numpy array full of data with given shape
    att:                                      (example) ob.data.vertices
    name:  string in foreach_get              (example) 'co'
    dim2:  final shape of the array           (example) co would be 3 for 3d vectors
    vecs:  include the existing array speedup (example) ob_co
    dtype: numpy data type                    (example) np.float32
    proxy: object. uses modifier effects      (example) ob
    shape: include for arrays with 3 dimensions
    If proxy is used it must be a mesh object."""
    if proxy is not None:
        data = proxy.to_mesh(bpy.context.scene, True, 'PREVIEW')
        att = data.vertices
        dim1 = len(att)
        if vecs is None:
            vecs = np.zeros(dim1 * dim2, dtype=dtype)
        att.foreach_get(name, vecs.ravel())
        bpy.data.meshes.remove(data)
        vecs.shape = (dim1, dim2)
        return vecs
    
    dim1 = len(att)        
    if vecs is None:
        vecs = np.zeros(dim1 * dim2, dtype=dtype)
    att.foreach_get(name, vecs.ravel())
    if shape is None:
        vecs.shape = (dim1, dim2)
        return vecs
    vecs.shape = (dim1, shape[0], shape[1])
    return vecs
    
    
def set_att(vecs, att, name):
    att.foreach_set(name, vecs.ravel())
