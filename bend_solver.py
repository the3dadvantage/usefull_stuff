import bpy
import bmesh
import numpy as np


# universal ---------------------
def get_obm(ob):
    """gets bmesh in editmode or object mode"""
    ob.update_from_editmode()
    obm = bmesh.new()
    obm.from_mesh(ob.data)
    return obm
    

# universal ---------------------
def get_weights(tris, points):
    """Find barycentric weights for triangles.
    Tris is a Nx3x3 set of triangle coords.
    points is the same N in Nx3 coords"""
    origins = tris[:, 0]
    cross_vecs = tris[:, 1:] - origins[:, None]
    v2 = points - origins

    # ---------
    v0 = cross_vecs[:,0]
    v1 = cross_vecs[:,1]

    d00_d11 = np.einsum('ijk,ijk->ij', cross_vecs, cross_vecs)
    d00 = d00_d11[:,0]
    d11 = d00_d11[:,1]
    d01 = np.einsum('ij,ij->i', v0, v1)
    d02 = np.einsum('ij,ij->i', v0, v2)
    d12 = np.einsum('ij,ij->i', v1, v2)

    div = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * div
    v = (d00 * d12 - d01 * d02) * div

    weights = np.array([1 - (u+v), u, v, ]).T
    return weights


# universal ---------------------
def get_co_shape(ob, key=None, ar=None):
    """Get vertex coords from a shape key"""
    if ar is not None:
        ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
        return ar
    c = len(ob.data.vertices)
    ar = np.empty((c, 3), dtype=np.float32)
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
    return ar


# universal ---------------------
def get_poly_centers(ob, co, data=None):
    """Get poly centers. Data is meant
    to be built the first time then
    passed in. (dynamic)"""

    if data is not None:
        data[0][:] = 0
        np.add.at(data[0], data[2], co[data[3]])
        data[0] /= data[1][:, None]    
        return data[0]

    pc = len(ob.data.polygons)
    pidex = np.hstack([[v for v in p.vertices] for p in ob.data.polygons])

    div = [len(p.vertices) for p in ob.data.polygons]

    indexer = []
    for i, j in enumerate(div):
        indexer += [i] * j
    div = np.array(div, dtype=np.float32)

    centers = np.zeros((pc, 3), dtype=np.float32)

    np.add.at(centers, indexer, co[pidex])
    centers /= div[:, None]    
    
    return [centers, div, indexer, pidex]


# universal ---------------------
def pairs_idx(ar):
    """Eliminates duplicates and mirror duplicates.
    for example, [1,4], [4,1] or duplicate occurrences of [1,4]
    Returns ar (array) and the index that removes the duplicates."""
    # no idea how this works (probably sorcery) but it's really fast
    a = np.sort(ar, axis=1) # because it only sorts on the second acess the index still matches other arrays.
    #x = np.random.rand(a.shape[1])
    x = np.linspace(1, 2, num=a.shape[1])
    y = a @ x
    unique, index = np.unique(y, return_index=True)
    return a[index], index
#                                                         #
#                                                         #
# --------------------end universal---------------------- #


# precalculated ------------------------
def get_j_surface_offset(cloth):
    """Get the vecs to move the plotted
    wieghts off the surface."""
    
    ax = cloth.j_axis_vecs
    ce = cloth.j_ce_vecs # has the faces swapped so the normal corresponds to the other side of the axis
    cross = np.cross(ax, ce)

    cloth.j_normals = cross / np.sqrt(np.einsum('ij,ij->i', cross, cross))[:, None]
    cloth.plot_normals = cloth.j_normals[cloth.j_tiler]
    
    cloth.plot_vecs = cloth.sco[cloth.swap_jpv] - cloth.j_plot
    cloth.plot_dots = np.einsum('ij,ij->i', cloth.plot_normals, cloth.plot_vecs)[:, None]


# dynamic ------------------------------
def measure_linear_bend(cloth):
    """Takes a set of coords and an edge idx and measures segments"""
    l = cloth.sp_ls # left side of the springs (Full moved takes the place of the right side)
    v = cloth.full_moved - cloth.co[l]
    d = np.einsum("ij ,ij->i", v, v)
    return v, d, np.sqrt(d)    


# dynamic ------------------------------
def get_eq_tri_tips(cloth, co, centers, skip=False):
    """Slide the centers of each face along
    the axis until it's in the middle for
    using as a triangle. (dynamic)"""
    
    skip = True # set to false to use eq tris. 
    if skip: # skip will test if it really makes any difference to move the tris to the center
        cloth.j_axis_vecs = co[cloth.stacked_edv[:,1]] - co[cloth.stacked_edv[:,0]]
        cloth.j_tips = centers[cloth.stacked_faces]
        cloth.j_ce_vecs = centers[cloth.stacked_faces] - co[cloth.stacked_edv[:,0]]
        return cloth.j_tips, cloth.j_axis_vecs, cloth.j_ce_vecs
    
    # creates tris from center and middle of edge. 
    # Not sure if it makes any difference... 
    j_axis_vecs = co[cloth.stacked_edv[:,1]] - co[cloth.stacked_edv[:,0]]
    j_axis_dots = np.einsum('ij,ij->i', j_axis_vecs, j_axis_vecs)
    j_ce_vecs = centers[cloth.stacked_faces] - co[cloth.stacked_edv[:,0]]
    cloth.swap_ce_vecs = centers[cloth.swap_faces] - co[cloth.stacked_edv[:,0]]
    j_cea_dots = np.einsum('ij,ij->i', j_axis_vecs, j_ce_vecs)
    
    j_div = j_cea_dots / j_axis_dots
    j_spit = j_axis_vecs * j_div[:,None]
    
    j_cpoe = co[cloth.stacked_edv[:,0]] + j_spit    
    jt1 = centers[cloth.stacked_faces] - j_cpoe
    j_mid = co[cloth.stacked_edv[:,0]] + (j_axis_vecs * 0.5)    
    
    cloth.j_tips = j_mid + jt1
    cloth.j_axis_vecs = j_axis_vecs
    cloth.j_ce_vecs = j_ce_vecs
    # ---------------------
    return cloth.j_tips, cloth.j_axis_vecs, cloth.j_ce_vecs


# precalculated ------------------------
def eq_bend_data(cloth):
    """Generates face pairs around axis edges.
    Supports edges with 2-N connected faces.
    Can use internal structures this way."""
    ob = cloth.ob
    obm = get_obm(ob)
    sco = cloth.sco
    
    # eliminate sew edges and outer edges:
    ed = [e for e in obm.edges if len(e.link_faces) > 1]
    
    first_row = []
    e_tiled = []
    f_ls = []
    f_rs = []
    for e in ed:
        ls = []        
        for f in e.link_faces:
            otf = [lf for lf in e.link_faces if lf != f]
            for lf in otf:    
                f_ls += [f.index]
                f_rs += [lf.index]
                e_tiled += [e.index]
    
    shape1 = len(f_ls)
    paired = np.empty((shape1, 2), dtype=np.int32)
    paired[:, 0] = f_ls
    paired[:, 1] = f_rs
    
    # faces grouped left and right
    cloth.face_pairs, idx = pairs_idx(paired)
    cloth.stacked_faces = cloth.face_pairs.T.ravel()
    jfps = cloth.stacked_faces.shape[0]
    
    # swap so we get wieghts from tris opposite axis
    cloth.swap_faces = np.empty(jfps, dtype=np.int32)
    cloth.swap_faces[:jfps//2] = cloth.face_pairs[:, 1]
    cloth.swap_faces[jfps//2:] = cloth.face_pairs[:, 0]
    
    # remove duplicate pairs so edges match face pairs
    tiled_edges = np.array(e_tiled)[idx] 
    
    # v1 and v2 for each face pair (twice as many faces because each pair shares an edge)
    obm.edges.ensure_lookup_table()
    cloth.edv = np.array([[obm.edges[e].verts[0].index,
                     obm.edges[e].verts[1].index]
                     for e in tiled_edges], dtype=np.int32)
    
    shape = cloth.edv.shape[0]
    cloth.stacked_edv = np.tile(cloth.edv.ravel(), 2)
    cloth.stacked_edv.shape = (shape * 2, 2)


def get_poly_vert_tilers(cloth):
    """Get an index to tile the left and right sides.
    ls and rs is based on the left and right sides of
    the face pairs."""
    
    cloth.swap_jpv = []
    cloth.jpv_full =[]
    ob = cloth.ob
    
    cloth.ab_faces = []
    cloth.ab_edges = []
    
    count = 0
    for i, j in zip(cloth.swap_faces, cloth.stacked_edv): # don't need to swap edv because both sides share the same edge
        
        pvs = [v for v in ob.data.polygons[i].vertices]
        nar = np.array(pvs)
        b1 = nar != j[0]
        b2 = nar != j[1]
        
        nums = np.arange(nar.shape[0]) + count 
        cloth.ab_faces += nums[b1 & b2].tolist()
        cloth.ab_edges += nums[~(b1)].tolist()
        cloth.ab_edges += nums[~(b2)].tolist()
        
        count += nar.shape[0]
        r = [v for v in ob.data.polygons[i].vertices if v not in j]
        cloth.swap_jpv += r

    for i in cloth.swap_faces:
        r = [v for v in ob.data.polygons[i].vertices]
        cloth.jpv_full += r


def tiled_weights(cloth):
    """Tile the tris with the polys for getting
    barycentric weights"""
    
    ob = cloth.ob
    face_pairs = cloth.face_pairs
    
    # counts per poly less the two in the edges
    cloth.full_counts = np.array([len(p.vertices) for p in ob.data.polygons], dtype=np.int32)
    cloth.full_div = np.array(cloth.full_counts, dtype=np.float32)[cloth.swap_faces][:, None]
    cloth.plot_counts = cloth.full_counts - 2 # used by plotted centers

    # joined:
    jfps = cloth.stacked_faces.shape[0]

    jsc = cloth.plot_counts[cloth.swap_faces]    
    cloth.j_tiler = np.hstack([[i] * jsc[i] for i in range(jfps)])
    cloth.js_tris = cloth.j_tris[cloth.j_tiler]
    
    jscf = cloth.full_counts[cloth.swap_faces]
    cloth.ab_tiler = np.hstack([[i] * jscf[i] for i in range(jfps)])
    cloth.sp_ls = np.hstack([[v for v in cloth.ob.data.polygons[f].vertices] for f in cloth.swap_faces])
    cloth.sp_rs = np.arange(cloth.sp_ls.shape[0])    
    

def triangle_data(cloth):

    sco = cloth.sco
    edv = cloth.edv
    
    # joined tris:
    j_tris = np.zeros((cloth.j_tips.shape[0], 3, 3), dtype=np.float32)
    j_tris[:, :2] = sco[cloth.stacked_edv]
    j_tris[:, 2] = cloth.j_tips
    cloth.j_tris = j_tris
    #cloth.js_tris = j_tris
    #-----------------
    
    # get the tilers for creating tiled weights
    tiled_weights(cloth)

    jw = get_weights(cloth.js_tris, sco[cloth.swap_jpv])
    cloth.j_plot = np.sum(cloth.js_tris * jw[:,:,None], axis=1)
    get_j_surface_offset(cloth)
    cloth.jw = jw
    

def linear_bend_set(cloth):
    
    cloth.bend_stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32)
    ab = np.array(cloth.jpv_full)
    
    springs = []
    vc = len(cloth.ob.data.vertices)
    for i in range(vc):
        w = np.where(i == ab)[0]
        for j in w:
            springs.append([i,j])
    
    cloth.linear_bend_springs = np.array(springs, dtype=np.int32)
    cloth.bend_v_fancy = cloth.linear_bend_springs[:,0]
    

def ab_setup(cloth):
    cloth.ab_centers = np.empty((cloth.stacked_faces.shape[0], 3), dtype=np.float32)
    cloth.ab_coords = np.empty((len(cloth.jpv_full), 3), dtype=np.float32)
    

def dynamic(cloth):

    # get centers from MC_current
    centers = get_poly_centers(cloth.ob, cloth.co, cloth.center_data)
    co = cloth.co
    
    #cloth.j_tris[:] = 0
    cloth.j_tris[:, :2] = co[cloth.stacked_edv]
    tips, ax, ce = get_eq_tri_tips(cloth, co, centers, skip=False)
    cloth.j_tris[:, 2] = tips
    
    jw = cloth.jw
    j_plot = np.sum(cloth.j_tris[cloth.j_tiler] * jw[:,:,None], axis=1)

    # for just flattening
    final_plot = j_plot
    flat = False
    if not flat:
        cross = np.cross(ax, ce)
        normals = cross / np.sqrt(np.einsum('ij,ij->i', cross, cross))[:, None]
        plot_normals = normals[cloth.j_tiler]
        final_plot = j_plot + (plot_normals * cloth.plot_dots)
        
    # get centers from plot
    cloth.ab_centers[:] = 0
    cloth.ab_centers += co[cloth.stacked_edv[:, 0]]
    cloth.ab_centers += co[cloth.stacked_edv[:, 1]]    
    np.add.at(cloth.ab_centers, cloth.j_tiler, final_plot)

    cloth.ab_centers /= cloth.full_div

    c_vecs = centers[cloth.swap_faces] - cloth.ab_centers

    cloth.ab_coords[cloth.ab_faces] = final_plot
    cloth.ab_coords[cloth.ab_edges] = cloth.co[cloth.stacked_edv.ravel()]
    
    full_moved = cloth.ab_coords + c_vecs[cloth.ab_tiler]
    
    cloth.full_moved = full_moved
        

def bend_setup(cloth):
    cloth.center_data = get_poly_centers(cloth.ob, cloth.sco, data=None)
    cloth.source_centers = np.copy(cloth.center_data[0]) # so we can overwrite the centers array when dynamic
    eq_bend_data(cloth)
    get_poly_vert_tilers(cloth)
    get_eq_tri_tips(cloth, cloth.sco, cloth.source_centers)
    triangle_data(cloth)
    ab_setup(cloth)
    linear_bend_set(cloth)
    

def linear_bash(cloth):
    
    stretch = cloth.ob.MC_props.bend * .2
        
    basic_set = cloth.linear_bend_springs
    basic_v_fancy = cloth.sp_ls
    stretch_array = cloth.bend_stretch_array
    
    # (current vec, dot, length)
    cv, cd, cl = measure_linear_bend(cloth) # from current cloth state
    move_l = cl * stretch

    # mean method -------------------
    cloth.bend_stretch_array[:] = 0.0

    #rock_hard_abs = np.abs(move_l)
    np.add.at(cloth.bend_stretch_array, basic_v_fancy, move_l)
    weights = move_l / cloth.bend_stretch_array[basic_v_fancy]
    # mean method -------------------

    # apply forces ------------------

    move = cv * (move_l / cl)[:,None]
    move *= weights[:,None]
    np.add.at(cloth.co, basic_v_fancy, np.nan_to_num(move))



print()
print("------------------ new eq ------------------")

class Cloth:
    pass

cloth = Cloth()
cloth.ob = bpy.data.objects['nn']
#cloth.ob = bpy.data.objects['nnn']
cloth.co = get_co_shape(cloth.ob, key='MC_current', ar=None)
cloth.sco = get_co_shape(cloth.ob, key='MC_source', ar=None)

bend_setup(cloth)
dynamic(cloth)
linear_bash(cloth)


cloth.ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', cloth.co.ravel())
cloth.ob.data.update()
