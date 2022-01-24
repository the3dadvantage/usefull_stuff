## =============================== ##
## =============================== ##
# play mega tri mesh cache
# !!! don't forget to delete shape keys !!!
import bpy
import numpy as np

# for using pip in python:
from pip._internal import main as pipmain
pipmain(['install', 'package-name'])


def add_empty(name, loc, rot, size=0.05, type="SPHERE"):
    """Create an empty and link it
    to the scene"""
    if name in bpy.data.objects:
        o = bpy.data.objects[name]
    else:    
        o = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(o)

        # empty_draw was replaced by empty_display
    o.empty_display_size = size
    o.empty_display_type = type

    o.location.xy = loc
    o.rotation_euler[2] = rot


def redistribute_polyline(co, spacing=None, respaced=None):
    """Walk the points in a polygon
    assuming they are not spaced
    evenly and space them evenly.
    If spacing is None use average len.
    Respaced can be an array of distances
    for spacing unevenly."""

    vecs = co[1:] - co[:-1]
    dots = np.einsum('ij,ij->i', vecs, vecs)

    # cull zero length
    booly = np.ones(co.shape[0], dtype=bool)
    booly[1:] = dots > 0.0
    co = co[booly]

    vecs = co[1:] - co[:-1]
    dots = np.einsum('ij,ij->i', vecs, vecs)

    lengths = np.sqrt(dots)
    u_vecs = vecs / lengths[:, None]
    av_len = np.mean(lengths)
    total_len = np.sum(lengths)

    if spacing is None:
        spacing = av_len

    # get the number of points
    point_count = int(total_len // spacing)
    
    if respaced is None:    
        respaced = np.linspace(0.0, total_len, point_count)
    
    cumulated = np.cumsum(lengths)
    with_zero = np.zeros(cumulated.shape[0] + 1, dtype=np.float32)
    with_zero[1:] = cumulated
    
    idxer = np.arange(with_zero.shape[0])
    hits = []
    for i in range(respaced.shape[0] - 1):
        hit = respaced[i] >= with_zero
        idx = idxer[hit][-1]
        
        dif = respaced[i] - with_zero[idx]
        crawl = co[idx] + (u_vecs[idx] * dif)
        hits.append(crawl)
                
    return hits + [co[-1]]


class Bezier():
    def TwoPoints(t, P1, P2):
        """
        Returns a point between P1 and P2, parametised by t.
        """

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        """
        Returns a list of points interpolated by the Bezier process
        """
        newpoints = []
        for i1 in range(0, len(points) - 1):
        #for i1 in range(0, len(points)):
            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
        return newpoints

    def Point(t, points):
        """
        Returns a point interpolated by the Bezier process
        """
        newpoints = points
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
        return newpoints[0]

    def Curve(t_values, points):
        """
        Returns a point interpolated by the Bezier process
        """

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)

        curve = np.delete(curve, 0, 0)

        return curve
    
    # example
    bez_test = False
    if bez_test:
        t_points = np.arange(0, 2, 0.1)
        test = np.array([[0, 5], [4, 10], [8, 10]])#, [4, 0], [6, 0], [10, 5]]) # can be lots of points
        test_set_1 = Bezier.Curve(t_points, test)


def percent_solve(result, decimal):
    """Find the value needed for
    the end result given a percentage.
    Example: if I want 100 and 20 percent
    is going to be taken: solve(100, .2)."""
    mult = 1 / decimal # 5 here
    div = result / (mult - 1)
    return div * mult


def get_linked(obm, idx, op=None):
    """put in the index of a vert. Get everything
    linked just like 'select_linked_pick()'"""
    vboos = np.zeros(len(obm.verts), dtype=np.bool)
    cvs = [obm.verts[idx]]
    escape = False
    while not escape:
        new = []
        for v in cvs:
            if not vboos[v.index]:
                vboos[v.index] = True
                lv = [e.other_vert(v) for e in v.link_edges]
                culled = [v for v in lv if not vboos[v.index]]
                new += culled
        cvs = new
        if len(cvs) == 0:
            escape = True
    idxer = np.arange(len(obm.verts))[vboos]
    if op == "DELETE":    
        verts = [obm.verts[i] for i in idxer]
        bmesh.ops.delete(obm, geom=verts)    
    return idxer


def map_ranges(r1b, r1t, r2b, r2t, val):
    """Find the value of range 1 where
    it maps to range two.
    r1b: range 1 bottom
    r1t: range 1 top"""

    dif1 = r1t - r1b
    dif2 = r2t - r2b
    vd = val - r1b
    dv1 = vd / dif1
    
    return dif2 * dv1 + r2b


#----------------------------------------------
def spread_array(ar=None, steps=6):
    """Create interpolated points
    between points in an array
    of vectors."""
    v = ar[1:] - ar[:-1]
    div = v / (steps + 1)
    if len(ar.shape) == 1:    
        new_ar = np.zeros(ar.shape[0] * (steps + 1))
    else:    
        new_ar = np.zeros((ar.shape[0] * (steps + 1), ar.shape[1]))
    new_ar[::steps + 1] = ar
    for i in range(steps):
        ph = ar[:-1] + (div * (i + 1))
        new_ar[:-(steps +1)][i+1::steps+1] = ph
    
    return new_ar[: -steps]

xy = np.arange(4)
ar = np.arange(8)
ar.shape = (4,2)
ar[:,0] = xy
ar[:,1] = xy

new_ar = spread_array(ar)

print('------ new ------')
print(new_ar)
#----------------------------------------------



def get_proxy_co(ob, co=None, proxy=None):
    """Gets co with modifiers like cloth"""
    if proxy is None:
        dg = bpy.context.evaluated_depsgraph_get()
        prox = ob.evaluated_get(dg)
        proxy = prox.to_mesh()

    if co is None:
        vc = len(proxy.vertices)
        co = np.empty((vc, 3), dtype=np.float32)

    proxy.vertices.foreach_get('co', co.ravel())
    ob.to_mesh_clear()
    return co


def np_co_to_text(ob, co, rw='w'):
    """Read or write cache file"""
    name = ob.name + 'cache.npy'
    
    if rw == 'w':
        if name not in bpy.data.texts:
            bpy.data.texts.new(name)
        
        txt = bpy.data.texts[name]
        np.savetxt(txt, co)
        
        return
    
    vc = len(ob.data.vertices)
    txt = bpy.data.texts[name].as_string()
    frame = bpy.context.scene.frame_current
    start = (frame -1) * vc * 3
    
    co = np.fromstring(txt, sep='\n')[start: start + (vc * 3)]
    co.shape = (co.shape[0]//3, 3)

    ob.data.vertices.foreach_set('co', co.ravel())
    ob.data.update()

    
def play_internal_cache(scene):
    ob = bpy.data.objects['Cube']
    np_co_to_text(ob, co=None, rw='r')


bpy.app.handlers.frame_change_post.append(play_internal_cache)
## =============================== ##
## =============================== ##


def read_flap_target():
    import bpy
    import json
    ob = bpy.data.objects['g8193']

    file = bpy.data.texts['flap_ptrs.json']
    slices = json.loads(file.as_string())

    for k, v in slices.items():
        for ve in v:
            ob.data.vertices[ve].select = True

        ob.data.update()
        #error 

def bmesh_proxy(ob):
    """Get a bmesh contating modifier effects"""
    dg = bpy.context.evaluated_depsgraph_get()
    prox = ob.evaluated_get(dg)
    proxy = prox.to_mesh()    
    obm = bmesh.new()
    obm.from_mesh(proxy)
    return obm


def select_edit_mode(sc, ob, idx, type='v', deselect=False, obm=None):
    """Selects verts in edit mode and updates"""
    
    if ob.data.is_editmode:
        if obm is None:
            obm = bmesh.from_edit_mesh(ob.data)
            obm.verts.ensure_lookup_table()
        
        if type == 'v':
            x = obm.verts
        if type == 'f':
            x = obm.faces
        if type == 'e':
            x = obm.edges
        
        if deselect:
            for i in x:
                i.select = False
        
        for i in idx:
            sc.select_counter[i] += 1
            x[i].select = True
        
        if obm is None:
            bmesh.update_edit_mesh(ob.data)
        #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


# ---------------------four functions below------------------------------- <<
def eliminate_duplicate_pairs(ar):
    """Eliminates duplicates and mirror duplicates.
    for example, [1,4], [4,1] or duplicate occurrences of [1,4]
    Returns an Nx2 array."""
    # no idea how this works (probably sorcery) but it's really fast
    a = np.sort(ar, axis=1)
    x = np.random.rand(a.shape[1])
    y = a @ x
    unique, index = np.unique(y, return_index=True)
    return a[index]
    # or
    strs = [str(e) + str(t) for e, t in zip(a[:,0], a[:,1])]
    uni = np.unique(strs, return_index=True)[1]



def tree(co, margin=0.001, _idx=None):
    
    # could test dividing up the world instead of dividing boxes
    b_min = np.min(co, axis=0)
    b_max = np.max(co, axis=0)
    mid = b_min + ((b_max - b_min) / 2)

    bpy.data.objects['a'].location = mid
    # l = left, r = right, f = front, b = back, u = up, d = down
    idx = np.arange(co.shape[0], dtype=np.int32)
    boxes = []

    # -------------------------------
    B = co[:,0] < mid[0] + margin
    il = idx[B]

    B = co[:,0] > mid[0] - margin
    ir = idx[B]

    # ------------------------------
    cil = co[:,1][il]
    B = cil > mid[1] - margin
    ilf = il[B]

    B = cil < mid[1] + margin
    ilb = il[B]

    cir = co[:,1][ir]
    B = cir > mid[1] - margin
    irf = ir[B]

    B = cir < mid[1] + margin
    irb = ir[B]

    # ------------------------------
    cilf = co[:,2][ilf]
    B = cilf > mid[2] - margin
    ilfu = ilf[B]
    B = cilf < mid[2] + margin
    ilfd = ilf[B]

    cilb = co[:,2][ilb]
    B = cilb > mid[2] - margin
    ilbu = ilb[B]
    B = cilb < mid[2] + margin
    ilbd = ilb[B]

    cirf = co[:,2][irf]
    B = cirf > mid[2] - margin
    irfu = irf[B]
    B = cirf < mid[2] + margin
    irfd = irf[B]

    cirb = co[:,2][irb]
    B = cirb > mid[2] - margin
    irbu = irb[B]
    B = cirb < mid[2] + margin
    irbd = irb[B]
    
    if _idx is None:
        boxes = [ilfu, ilfd, ilbu, ilbd, irfu, irfd, irbu, irbd]
        doubles = [i for i in boxes if i.shape[0] > 1]
            #return #[i.tolist() for i in boxes]
        return doubles
    
    boxes = [_idx[ilfu],
             _idx[ilfd],
             _idx[ilbu],
             _idx[ilbd],
             _idx[irfu],
             _idx[irfd],
             _idx[irbu],
             _idx[irbd]
             ]
    
    doubles = [i for i in boxes if i.shape[0] > 1]
    return doubles


def branches(co, margin):
    """Subsets of trees"""
    boxes = []
    b1 = tree(co, margin=margin)
    for i in b1:
        b2 = tree(co[i], margin=margin, _idx=i)
        for j in b2:
            b3 = tree(co[j], margin=margin, _idx=j)
            boxes += b3

    return boxes
    

def find_doubles(ob, margin=0.001):
    """Finds verts whose distance from each
    other is less than the margin.
    Returns an Nx2 numpy arry of close pairs."""
    
    vc = len(ob.data.vertices)
    co = np.empty((vc, 3), dtype=np.float32)
    ob.data.vertices.foreach_get('co', co.ravel())

    boxes = branches(co, margin)
    dubs = []
    m = margin ** 2
    
    for bz in boxes:
        if bz.shape[0] > 0:
            c = co[bz]
            b_vecs = c[:, None] - c
            d = np.einsum('ijk,ijk->ij', b_vecs, b_vecs)
            agw = np.argwhere(d <= m)
            cull = agw[:, 0] == agw[:, 1]
            agwc = agw[~cull]
            if agwc.shape[0] > 0:    
                
                dubs += bz[agwc].tolist()
        
    return eliminate_duplicate_pairs(np.array(dubs))
# ------------------------four functions above---------------------------- >>


def merge_verts(ob, margin=0.001, obm=None):

    if obm is None:
        obm = bmesh.new()
        obm.from_mesh(ob.data)
    
    bmesh.ops.remove_doubles(obm, verts=obm.verts, dist=margin)
    obm.to_mesh(ob.data)

    ob.data.update()
    obm.clear()
    obm.free()


def get_ordered_loop(poly_line, edges=None):
    """Takes a bunch of verts and gives the order
    based on connected edges.
    Or, takes an edge array of vertex indices
    and gives the vertex order."""

    if edges is not None:        
        v = edges[0][0]
        le = edges[np.any(v == edges, axis=1)]
        if len(le) != 2:
            print("requires a continuous loop of edges")
            return
            
        ordered = [v]
        for i in range(len(poly_line.data.vertices)):
            #le = v.link_edges
            le = edges[np.any(v == edges, axis=1)]
            if len(le) != 2:
                print("requires a continuous loop of edges")
                break

            ot1 = le[0][le[0] != v]
            ot2 = le[1][le[1] != v]
            v = ot1
            if ot1 in ordered[-2:]:    
                v = ot2
            if v == ordered[0]:
                break

            ordered += [v[0]]
        return ordered
        
    obm = get_bmesh(poly_line, refresh=True)
    v = obm.edges[0].verts[0]
    le = v.link_edges

    if len(le) != 2:
        print("requires a continuous loop of edges")
        return
        
    ordered = [v.index]
    for i in range(len(poly_line.data.vertices)):
        le = v.link_edges

        if len(le) != 2:
            print("requires a continuous loop of edges")
            break

        ot1 = le[0].other_vert(v)
        ot2 = le[1].other_vert(v)
        v = ot1
        if ot1.index in ordered[-2:]:    
            v = ot2
        if v.index == ordered[0]:
            break

        ordered += [v.index]
    
    return ordered    
    

def read_python_script(name=None):
    import bpy
    import inspect
    import pathlib
    """When this runs it makes a copy of this script
    and saves it to the blend file as a text"""

    p_ = pathlib.Path(inspect.getfile(inspect.currentframe()))
    py = p_.parts[-1]
    p = p_.parent.parent.joinpath(py)
    try:    
        o = open(p)
    except:
        p = p_.parent.joinpath(py) # linux or p1 (not sure why this is happening in p1)
        o = open(p)
    
    if name is None:
        name = 'new_' + py
        
    new = bpy.data.texts.new(name)
    
    r = o.read()
    new.write(r)


def cross_from_tris(tris):
    origins = tris[:, 0]
    vecs = tris[:, 1:] - origins[:, nax]
    cross = np.cross(vecs[:, 0], vecs[:, 1])
    return cross


def distance_along_normal(tris, points):
    """Return the distance along the cross
    product and the distance along normalized
    cross product"""
    origins = tris[:, 0]
    cross_vecs = tris[:, 1:] - origins[:, nax]    
    v2 = points - origins
    
    cross = np.cross(cross_vecs[:,0], cross_vecs[:,1])
    d_v2_c = np.einsum('ij,ij->i', v2, cross)
    d_v2_v2 = np.einsum('ij,ij->i', cross, cross) 
    div = d_v2_c / d_v2_v2        

    U_cross = cross / np.sqrt(d_v2_v2)[:, None]
    U_d = np.einsum('ij,ij->i', v2, U_cross)
        
    return div, U_d# for normalized


def connect_panels(self, s_norm_val=1.0, offset_steps=0, correct_rotation=True, reverse=False):
    """Offset steps is an int that allows backing up or moving forward
    the given number of edges.
    correct_rotation checks that the two arrays are paired correctly.
    reverse flips the direction of both arrays."""
    
    #==========================================
    if False:
        Bobj = self.garment.Bobj
        left_zipper_vert_ptrs = self.left_panel.get_connection_bmesh_vert_ptrs()
        right_zipper_vert_ptrs = self.right_panel.get_connection_bmesh_vert_ptrs()
    
    Bobj = bpy.context.object
    left_zipper_vert_ptrs = np.arange(48, 60)
    right_zipper_vert_ptrs = np.arange(36, 48)
    #==========================================


    if correct_rotation:
        # If first vert in left should pair with last vert in right
        # or just assume we need to correct: right_zipper_vert_ptrs = right_zipper_vert_ptrs[::-1]
        v_first_l = Bobj.data.vertices[left_zipper_vert_ptrs[0]].co
        v_first_r = Bobj.data.vertices[right_zipper_vert_ptrs[0]].co
        v_last_r = Bobj.data.vertices[right_zipper_vert_ptrs[-1]].co
        vec1 = v_first_l - v_first_r
        vec2 = v_first_l - v_last_r
        l1 = vec1 @ vec1
        l2 = vec2 @ vec2
        # print(l2, l1,"are we there yet???")

        if l2 < l1:

            right_zipper_vert_ptrs = right_zipper_vert_ptrs[::-1]

    if reverse:
        right_zipper_vert_ptrs = right_zipper_vert_ptrs[::-1]
        left_zipper_vert_ptrs = left_zipper_vert_ptrs[::-1]

    obm = get_bmesh(Bobj)
    obm.verts.ensure_lookup_table()

    # get the total length

    co = np.array([obm.verts[v].co for v in left_zipper_vert_ptrs])
    vecs = co[1:] - co[:-1]
    l = np.sqrt(np.einsum("ij ,ij->i", vecs, vecs))
    sums = np.cumsum(np.nan_to_num(l/np.sum(l)))

    bool = s_norm_val < sums
    indexer = np.where(bool)[0]
    removing = False

    # print(np.abs(offset_steps), bool.shape[0])

    stop = None
    if indexer.shape[0] == 0:
        stop = -1 # fill them all
        if offset_steps < 0:
            if np.abs(offset_steps) <= bool.shape[0]:
                stop = left_zipper_vert_ptrs[offset_steps -1]
            else:
                removing =True
    else:
        with_offset = indexer[0] + offset_steps
        set_stop = True

        if with_offset > indexer[-1]:
            stop = -1
            set_stop = False
        if with_offset < 0:
            removing = True
            set_stop = False
        if set_stop:
            stop = left_zipper_vert_ptrs[with_offset]

    if np.all(bool):
        removing = True
    for v1, v2 in zip(left_zipper_vert_ptrs, right_zipper_vert_ptrs):
        le = [e for e in obm.verts[v1].link_edges if e.other_vert(obm.verts[v1]).index == v2]
        existing = len(le) == 1

        if (removing & existing):
            obm.edges.remove(le[0])

        if not existing:
            if not removing:
                obm.edges.new([obm.verts[v1],obm.verts[v2]])

        if v1 == stop:
            removing = True
            
    obm.to_mesh(Bobj.data)
    Bobj.data.update()            


def apply_shape(ob, modifier_name='Cloth', update_existing_key=False, keep=['Cloth'], key_name='Cloth'):
    """Apply modifier as shape without using bpy.ops.
    Does not apply modifiers.
    Mutes modifiers not listed in 'keep.'
    Using update allows writing to an existing shape_key."""

    def turn_off_modifier(modifier, on_off=False):
        modifier.show_viewport = on_off

    mod_states = [mod.show_viewport for mod in ob.modifiers]
    [turn_off_modifier(mod, False) for mod in ob.modifiers if mod.name not in keep]

    dg = bpy.context.evaluated_depsgraph_get()
    proxy = ob.evaluated_get(dg)
    co = get_co(proxy)

    if update_existing_key:
        key = ob.data.shape_keys.key_blocks[key_name]
    else:
        key = new_shape_key(ob, name=key_name, arr=None, value=0)

    key.data.foreach_set('co', co.ravel())

    for i, j in zip(mod_states, ob.modifiers):
        j.show_viewport = i

    return key


def matrix_from_custom_orientation():
    """For using custom orientations as a matrix transform"""
    import bpy
    from bpy import context
    import mathutils   
    #Get the matrix of the transform orientation called 'name'
    custom_matrix = bpy.context.scene.orientations['name'].matrix
    #Copy the matrix to resize it from 3x3 matrix to 4x4 matrix
    custom_matrix_4 = custom_matrix.copy()
    custom_matrix_4.resize_4x4()
    #Set the matrix of the active object to match the resized matrix
    bpy.context.active_object.matrix_world = custom_matrix_4

    
def verts_in_group(ob, name='Group'):
    """Returns np array of indices for vertices in the group"""
    ob.update_from_editmode() # in case someone has assigned verts in editmode
    idx = ob.vertex_groups[name].index
    idxer = np.arange(len(ob.data.vertices))
    this = [[j.group for j in v.groups if j.group == idx] for v in ob.data.vertices]
    idxs = [i for i in idxer if len(this[i]) > 0]
    return np.array(idxs)


def save_data(name='saved_data.py', var='some_variable', data={'key': [1,2,3]}, overwrite=True):
    """Saves a dictionary as a variable in a python file
    as a blender internal text file. Can later import
    module and call all data as global variables."""
    if name not in bpy.data.texts:
        bpy.data.texts.new(name)

    data_text = bpy.data.texts[name]

    m = json.dumps(data, sort_keys=True, indent=2)

    if overwrite:
        data_text.from_string(var + ' = ' + m)
        return

    # can also just add to the text
    data_text.cursor_set(-1, character=-1) # in case someone moves the cursor
    data_text.write(var + ' = ' + m)

    data_text.cursor_set(-1, character=-1) # add the new line or we can't read it as a python module
    data_text.write('\n')


# to save an external text in a blend file
def save_text_in_blend_file(path, file_name='my_text.py'):
    """Run this then save the blend file.
    file_name is the key blender uses to store the file:
    bpy.data.texts[file_name]"""
    t = bpy.data.texts.new(file_name)
    read = open(path).read()
    t.write(read)


# to import the text as a module from within the blend file
def get_internal_text_as_module(filename, key):
    """Load a module and return a dictionary from
    that module."""
    module = bpy.data.texts[filename].as_module()
    return module.points[key]


def get_co(ob):
    """Returns Nx3 numpy array of vertex coords as float32"""
    v_count = len(ob.data.vertices)
    co = np.empty(v_count * 3, dtype=np.float32)
    ob.data.vertices.foreach_get('co', co)
    co.shape = (v_count, 3)
    return co


def new_shape_key(ob, name, arr=None, value=1):
    """Create a new shape key, set it's coordinates
    and set it's value"""
    new_key = ob.shape_key_add(name=name)
    new_key.value = value
    if arr is not None:
        new_key.data.foreach_set('co', arr.ravel())
    return new_key


def get_verts_in_group(ob, name):
    """Returns the indices of the verts that belong to the group"""
    idx = ob.vertex_groups[name].index
    vg = [v.index for v in ob.data.vertices if idx in [vg.group for vg in v.groups]]
    return np.array(vg)


# -------- group debug ----------------#
def new_shape_key(ob, name, arr=None, value=1):
    """Create a new shape key, set it's coordinates
    and set it's value"""
    new_key = ob.shape_key_add(name=name)
    new_key.value = value
    if arr is not None:
        new_key.data.foreach_set('co', arr.ravel())
    return new_key

# -------- group debug ----------------#
def link_mesh(verts, edges=[], faces=[], name='!!! Debug Mesh !!!'):
    """Generate and link a new object from pydata"""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)  
    mesh.update()
    mesh_ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(mesh_ob)
    return mesh_ob

# -------- group debug ----------------#
def create_debug_mesh(numpy_coords=[np.array([[1,2,3]]), np.array([[4,5,6]])],
    shape_keys=['Basis', 'key_1']):
    """Use a list of sets of numpy coords and matching list of shape key names.
    Creates a mesh point cloud with shape keys for each numpy coords set.
    !!! Adds this objet to the blend file !!!"""    
    key_count = len(shape_keys)
    ob = link_mesh(numpy_coords[0])
    keys = ob.data.shape_keys

    for i in range(key_count):
        new_shape_key(ob, shape_keys[i], numpy_coords[i], value=0)
        
    ob.data.update()


def offset_face_indices(faces=[]):
    """Sorts the original face vert indices
    for a new mesh from subset."""
    # Example: face[n].verts = [[20, 10, 30], [10, 30, 100]]
    # Converts to [[1, 0, 2], [0, 2, 3]]

    def add(c):
        c['a'] += 1
        return c['a']

    flat = np.hstack(faces)
    idx = np.unique(flat, return_inverse=True)[1]
    c = {'a': -1}
    new_idx = [[idx[add(c)] for j in i] for i in faces]


# get depsgraph co with various modifiers turned off
def get_co_with_modifiers(ob, types=[], names=[], include_mesh=False):
    """Get the coordinates of modifiers with
    specific modifiers turned on or off.
    List mods by type or name.
    This lets you turn off all mods of a type
    or just turn off by name."""

    debug = True
    if debug:
        # verify modifier names and types
        mod_types = [mod.type for mod in ob.modifiers]
        mod_names = [mod.name for mod in ob.modifiers]
        # if the arg names ar not right return
        type_check = np.all(np.in1d(types, mod_types))
        name_check = np.all(np.in1d(names, mod_names))

        if not (type_check & name_check):
            print("!!! Warning. Mods not listed correctly !!!")
            print("!!! Warning. Mods not listed correctly !!!")
            return

    # save mod states for reset
    mod_states = [mod.show_viewport for mod in ob.modifiers]

    def turn_off_modifier(modifier):
        modifier.show_viewport = False

    [turn_off_modifier(mod) for mod in ob.modifiers if mod.name in names]
    [turn_off_modifier(mod) for mod in ob.modifiers if mod.type in types]

    # get the coordinates with the current modifier state
    dg = bpy.context.evaluated_depsgraph_get()
    proxy = ob.evaluated_get(dg)
    co = get_co(proxy)

    for i, j in zip(mod_states, ob.modifiers):
        j.show_viewport = i

    if include_mesh:
        return co, proxy.data

    return co


def dots(a,b):
    #N x 3 - N x 3
    x = np.einsum('ij,ij->i', a, b)
    #3 - N x 3
    y = np.einsum('j,ij->i', a, b) # more simply b @ n
    #N x 3 - N x N x 3
    z = np.einsum('ij,ikj->ik', a, b)    
    #N x N x 3 - N x N x 3    
    w = np.einsum('ijk,ijk->ij', a, b)
    #N x 2 x 3 - N x 2 x 3
    a = np.einsum('ijk,ijk->ij', a, b)    
    
    #N x 2 x 3 - N x 3
    np.einsum('ij, ikj->ik', axis_vecs, po_vecs)
    
    #mismatched N x 3 - N2 x 3 with broadcasting so that the end result is tiled
    mismatched = np.einsum('ij,i...j->...i', a, np.expand_dims(b, axis=0))    
    # 4,3,3 - 4,2,3 with broadcasting    
    mismatched_2 = np.einsum('ijk,ij...k->i...j', a, np.expand_dims(b, axis=1))
    return x,y,z,w,a, mismatched, mismatched_2
    

