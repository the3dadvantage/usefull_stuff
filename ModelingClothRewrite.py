# when extruding:
#   currently have to switch to the source shape
#   This is bceause the coords will be in the same location
#       in the source shape. Could pause existing selection
#       along with new selection. Maybe play with source being
#       relative only while that's happening then switch it back
#       to the current being relative to source?
#   To confirm maybe wait until deselct all or wait a given
#       number of iterations?

# cache keyframing could be really cool. So do a linear
# interpolation between cached frames.

# !!!!!!!!!!!!!!!
# mesh keyframing might actually be what I need for
# my procedural animation stuff.
# think about the barycentric springs with equalateral tris applied to an armature
# !!!!!!!!!!!!!!!

# would it make sense to run cloth physics on a lattice or curve object?????

# undo !!!!!!
#def b_log(a=None,b=None):
    #return

# !!! OPTIMISE: lots of placed to provide and "out" array. Start with all the einsums.
# !!! might also check the speed of linalg against v / sqrt(einsum(vec vec))

doing_self_collisions = False
doing_self_collisions = True

try:
    import bpy
    from bpy.ops import op_as_string
    from bpy.app.handlers import persistent
    import os
    import shutil
    import pathlib
    import inspect
    import bmesh
    import functools as funky
    import numpy as np
    from numpy import newaxis as nax
    import time
    import copy # for duplicate cloth objects
    if doing_self_collisions:    
        self_collide = bpy.data.texts['EA_method.py'].as_module()
        #self_collide = bpy.data.texts['self_collide.py'].as_module()

except ImportError:
    pass

try:
    from garments_blender.utils.rich_blender_utils import B_log
    internal_log = B_log()
    internal_log.module = 'MC_tools'
    internal_log.active = True
    internal_log.name = 'log_MC_tools'
    b_log = internal_log.b_log

except:
    print("must be calling outside of p1")
    pass


# global data
MC_data = {}
MC_data['colliders'] = {}
MC_data['cloths'] = {}
MC_data['iterator'] = 0

# recent_object allows cloth object in ui
#   when selecting empties such as for pinning.
MC_data['recent_object'] = None


# developer functions ------------------------
def reload():
    """!! for development !! Resets everything"""
    # when this is registered as an addon I will want
    #   to recaluclate these objects not set prop to false

    # Set all props to False:
    reload_props = ['continuous', 'collider', 'animated', 'cloth', 'cache', 'cache_only', 'play_cache']
    if 'MC_props' in dir(bpy.types.Object):
        for i in reload_props:
            for ob in bpy.data.objects:
                ob.MC_props[i] = False

    for i in bpy.data.objects:
        if "MC_cloth_id" in i:
            del(i["MC_cloth_id"])
        if "MC_collider_id" in i:
            del(i["MC_collider_id"])

    #for detecting deleted colliders or cloths
    MC_data['cloth_count'] = 0
    MC_data['collider_count'] = 0

reload()


def np_co_to_text(ob, co, rw='w', cloth=None):
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

    # right here could put in a feature to overwrite in edit mode per frame.
    if ob.data.is_editmode:
        for i, v in enumerate(cloth.obm.verts):
            v.co = co[i]
        bmesh.update_edit_mesh(ob.data)
        return
            
    ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', co.ravel())
    ob.data.update()


# developer functions ------------------------
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


def get_proxy(ob):
    """Gets proxy mesh with mods"""
    # use: ob.to_mesh_clear() to remove
    dg = bpy.context.evaluated_depsgraph_get()
    prox = ob.evaluated_get(dg)
    proxy = prox.to_mesh()
    return proxy


# developer functions ------------------------
def read_python_script(name=None):
    """When this runs it makes a copy of this script
    and saves it to the blend file as a text
    Not a virus... well sort of like a virus"""

    p_ = pathlib.Path(inspect.getfile(inspect.currentframe()))
    py = p_.parts[-1]
    p = p_.parent.joinpath(py)
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


# developer functions ------------------------
def create_object_cache(ob):
    print('cache only function')


#print("new--------------------------------------")


# Cache functions ---------------
def cache_interpolation(cloth):
    # !!! have to finish this one
    # Might as well throw in the ability to scale the
    #   cache up and down.
    #   Compute on playback so we don't screw with the files

    ob = cloth.ob
    f = ob.MC_props.current_cache_frame
    fp = cloth.cache_dir
    idx = np.array([int(i.parts[-1]) for i in fp.iterdir()])


# Cache functions ---------------
def cache_only(ob, frame=None):

    self = ob.MC_props

    path = os.path.expanduser("~/Desktop")
    self['cache_folder'] = path
    mc_path = pathlib.Path(path).joinpath('MC_cache_files')

    if not mc_path.exists():
        mc_path.mkdir()

    final_path = mc_path.joinpath(ob.name)
    self['cache_name'] = ob.name
    final_path = mc_path.joinpath(ob.name)

    # create dir if it doesn't exist
    if not final_path.exists():
        final_path.mkdir()

    #f = bpy.context.scene.frame_current
    f = ob.MC_props.current_cache_frame
    ob.MC_props['current_cache_frame'] = f + 1
    if frame is not None:
        f = frame

    txt = final_path.joinpath(str(f))

    np.savetxt(txt, get_proxy_co(ob))


# Cache functions ---------------
def cache(cloth, keying=False):
    """Store a text file of Nx3 numpy coords."""
    ob = cloth.ob
    fp = cloth.cache_dir

    maf = ob.MC_props.max_frames
    con = ob.MC_props.continuous

    # update ccf either to current frame or to +1 if continuous
    ccf = ob.MC_props.current_cache_frame
    f = bpy.context.scene.frame_current
    ob.MC_props['current_cache_frame'] = f

    if keying:
        f = ccf
        ob.MC_props['current_cache_frame'] = ccf

    if con:
        f = ccf
        ob.MC_props['current_cache_frame'] = ccf + 1

    txt = fp.joinpath(str(f))

    #sf = cloth.ob.MC_props.start_frame
    #ef = cloth.ob.MC_props.end_frame
    #if (f >= sf) & (f <= ef):

    nonexistent = not txt.exists()

    #np.savetxt(txt, cloth.co)
    if (nonexistent | ob.MC_props.overwrite_cache):
        np.savetxt(txt, cloth.co)
        print('saved a cache file: ', txt)


# Cache functions ---------------
def play_cache(cloth, cb=False):
    """Load a text file of Nx3 numpy coords."""
    if cloth.ob.MC_props.internal_cache:
        np_co_to_text(cloth.ob, cloth.co, rw='r', cloth=cloth)
        return
        print('play cache is running')

    ob = cloth.ob

    if not hasattr(cloth, "cache_dir"):
        ob.MC_props['play_cache'] = False
        return

    cache_interpolation(cloth) # Finish this !!!

    f = bpy.context.scene.frame_current

    if ob.MC_props.continuous:
        f = ob.MC_props.current_cache_frame
        ob.MC_props['current_cache_frame'] = f + 1

    if cb: # when running the callback
        f = ob.MC_props.current_cache_frame

    fp = cloth.cache_dir

    txt = fp.joinpath(str(f))

    if txt.exists():
        cloth.co = np.loadtxt(txt)

    key = 'MC_current'
    # cache only playback
    if ob.MC_props.cache_only:
        key = 'cache_key'
        if not ob.data.is_editmode:
            ob.data.shape_keys.key_blocks['cache_key'].data.foreach_set('co', cloth.co.ravel())
            ob.data.update()
            return

    if ob.data.is_editmode:
        index = ob.data.shape_keys.key_blocks.find(key)
        if ob.active_shape_key_index == index:

            try:
                cloth.obm.verts
            except:
                #print("updated cloth.obm")
                cloth.obm = bmesh.from_edit_mesh(ob.data)

            for i, j in enumerate(cloth.co):
                cloth.obm.verts[i].co = j
        return

    ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set("co", cloth.co.ravel())
    ob.data.update()


# debugging
def T(type=1, message=''):
    if type == 1:
        return time.time()
    print(time.time() - type, message)


# ============================================================ #
#                    universal functions                       #
#                                                              #


def get_tri_bounds(co, tridex):
    """Convert to tri mesh.
    Get the bounding corners
    of each face."""

    tri_coords = co[tridex]
    # optimize # can provide out arrays for min and max
    mins = np.min(tri_coords, axis=1)
    maxs = np.max(tri_coords, axis=1)

    return mins, maxs


def to_boxes(min, max, mid, bool):
    # [ilfu, ilfd, ilbu, ilbd, irfu, irfd, irbu, irbd]
    #so a box consists of a min and max bound as two 3d vecs

    # need to add the margin to these before checking
    #   if the tris are in there.

    #         [max]
    #     [mid]
    # [min]


    #so I could run it again with the mins
    #and maxs of the tris.
    #I would get a bunch of idxs of tris




    r = +x
    l = -x
    f = +y
    b = -y
    u = +z
    d = -z

    [ilfu, ilfd, ilbu, ilbd, irfu, irfd, irbu, irbd]




# universal ---------------------
def octree(co, margin=0.001, idx=None, bounds=None):
    """Adaptive octree. Good for finding doubles or broad
    phase collision culling""" # first box is based on bounds so first box could be any shape rectangle
    # could test dividing up the world instead of dividing boxes
    # to run extra depth run this again on subsets.
    # if the count in a single box is too high that might make sense.

    # instead of fancy indexing with subsets I could try a bool array of the entire index...
    if bounds is None:
        b_min = np.min(co, axis=0)
        b_max = np.max(co, axis=0)
        mid = b_min + ((b_max - b_min) / 2)
        mid_ = mid + margin
        _mid = mid - margin

        x_, y_, z_ = mid_[0], mid_[1], mid_[2]
        _x, _y, _z = _mid[0], _mid[1], _mid[2]
        #x, y, z = mid[0], mid[1], mid[2]

    else:
        b_min = bounds[0]
        b_max = bounds[1]
        mid = bounds[2]

        mid_ = mid # + margin
        _mid = mid # - margin

        x_, y_, z_ = mid_[0], mid_[1], mid_[2]
        _x, _y, _z = _mid[0], _mid[1], _mid[2]
        #x, y, z = mid[0], mid[1], mid[2]

    # l = left, r = right, f = front, b = back, u = up, d = down
    if idx is None:
        idx = np.arange(co.shape[0], dtype=np.int32)

    bounds = []

    # -------------------------------
    B = co[:,0] < x_# + margin
    il = idx[B]

    B = co[:,0] > _x# - margin
    ir = idx[B]

    # ------------------------------
    cil = co[:,1][il]
    B = cil > _y# - margin
    ilf = il[B]

    B = cil < y_# + margin
    ilb = il[B]

    cir = co[:,1][ir]
    B = cir > _y# - margin
    irf = ir[B]

    B = cir < y_# + margin
    irb = ir[B]

    # ------------------------------
    cilf = co[:,2][ilf]
    B = cilf > _z# - margin
    ilfu = ilf[B]
    B = cilf < z_# + margin
    ilfd = ilf[B]

    cilb = co[:,2][ilb]
    B = cilb > _z# - margin
    ilbu = ilb[B]
    B = cilb < z_# + margin
    ilbd = ilb[B]

    cirf = co[:,2][irf]
    B = cirf > _z# - margin
    irfu = irf[B]
    B = cirf < z_# + margin
    irfd = irf[B]

    cirb = co[:,2][irb]
    B = cirb > _z# - margin
    irbu = irb[B]
    B = cirb < z_# + margin
    irbd = irb[B]

    boxes = [ilfu, ilfd, ilbu, ilbd, irfu, irfd, irbu, irbd]
    full = [i for i in boxes if i.shape[0] > 0]
    bool = [i.shape[0] > 0 for i in boxes]

    return full, b_min, b_max, mid, bool


def octree_2(co, bounds=None, idx=None):

    # I think there is a way to pass in
    # the bounds of the collision object.
    # too late in the afternoon as I write this...

    if bounds is None:
        b_min = np.min(co, axis=0)
        b_max = np.max(co, axis=0)

    else:
        b_min = bounds[0]
        b_max = bounds[1]

    if idx is None:
        idx = np.arange(co.shape[0], dtype=np.int32)

    mid = b_min + ((b_max - b_min) / 2)
    x, y, z = mid[0], mid[1], mid[2]

    # -------------------------------
    B = co[:,0] < x
    il = idx[B]
    ir = idx[~B]

    # ------------------------------
    cil = co[:,1][il]
    B = cil > y
    ilf = il[B]
    ilb = il[~B]

    cir = co[:,1][ir]
    B = cir > y
    irf = ir[B]
    irb = ir[~B]

    # ------------------------------
    cilf = co[:,2][ilf]
    B = cilf > z
    ilfu = ilf[B]
    ilfd = ilf[~B]

    cilb = co[:,2][ilb]
    B = cilb > z
    ilbu = ilb[B]
    ilbd = ilb[~B]

    cirf = co[:,2][irf]
    B = cirf > z
    irfu = irf[B]
    irfd = irf[~B]

    cirb = co[:,2][irb]
    B = cirb > z
    irbu = irb[B]
    irbd = irb[~B]

    boxes = [ilfu, ilfd, ilbu, ilbd, irfu, irfd, irbu, irbd]
    full = [i for i in boxes if i.shape[0] > 0]
    bool = [i.shape[0] > 0 for i in boxes]

    return full, b_min, b_max, mid, bool


# update the cloth ---------------
def collide(cloth, colliders):

    t = time.time()

    if not colliders:
        return

    ob = cloth.ob
    co = cloth.co

    np.copyto(cloth.wco, cloth.co)
    apply_in_place(ob, cloth.wco)
    min = np.min(cloth.wco, 0)
    max = np.max(cloth.wco, 0)

    for c in colliders:
        obc = c.ob
        om = obc.MC_props.outer_margin
        im = obc.MC_props.inner_margin

        if obc.data.is_editmode:
            obc.update_from_editmode()

        # check for changes. (should just need faces...)
        proxy = get_proxy(obc)
        face_count = len(proxy.polygons)
        if c.face_count != face_count:
            c.refresh()

        # refresh vert coords
        proxy.vertices.foreach_get('co', c.pco.ravel())
        obc.to_mesh_clear()
        apply_in_place(obc, c.pco)

        minc = np.min(c.pco, 0) - om
        maxc = np.max(c.pco, 0) + om

        in_min = min < maxc
        in_max = max > minc

        # only check objects whose bounds intersect
        bounds = np.all(in_min & in_max)
        if bounds:

            # only check faces intersecting the cloth bounds.
            tco = c.pco[c.tridex]

            c.tridexer = np.arange(tco.shape[0]) # precalculate this

            tz = tco[:, :, 2]
            tx = tco[:, :, 0]
            ty = tco[:, :, 1]

            #print(c.tridexer.shape, "shape start")
            zmax = np.max(tz[c.tridexer], 1)
            zmaxboo = zmax > min[2]
            c.tridexer = c.tridexer[zmaxboo]
            #print(c.tridexer.shape, "shape mid")
            zmin = np.min(tz[c.tridexer], 1)
            zminboo = zmin < max[2]
            c.tridexer = c.tridexer[zminboo]
            #print(c.tridexer.shape, "shape end")

            #print('y ------------')
            #print(c.tridexer.shape, "shape start")
            ymax = np.max(ty[c.tridexer], 1)
            ymaxboo = ymax > min[1]
            c.tridexer = c.tridexer[ymaxboo]
            #print(c.tridexer.shape, "shape mid")
            ymin = np.min(ty[c.tridexer], 1)
            yminboo = ymin < max[1]
            c.tridexer = c.tridexer[yminboo]
            #print(c.tridexer.shape, "shape end")

            #print('x ------------')
            #print(c.tridexer.shape, "shape start")
            xmax = np.max(tx[c.tridexer], 1)
            xmaxboo = xmax > min[0]
            c.tridexer = c.tridexer[xmaxboo]
            #print(c.tridexer.shape, "shape mid")
            xmin = np.min(tx[c.tridexer], 1)
            xminboo = xmin < max[0]
            c.tridexer = c.tridexer[xminboo]

            print(time.time()-t)
            #return

        bmn = np.empty(6, dtype=np.float32)
        bmn.shape = (3,2)
        bmn[:,0] = min
        bmn[:,1] = minc
        bmin = np.sort(bmn, 1)[:, 0]

        bmx = np.empty(6, dtype=np.float32)
        bmx.shape = (3,2)
        bmx[:,0] = max
        bmx[:,1] = maxc
        bmax = np.sort(bmx, 1)[:, 0]

        ot, min, max, mid, bool = octree_2(co, bounds=[bmin, bmax])

        p1 = bpy.data.objects['p1']
        p2 = bpy.data.objects['p2']
        p3 = bpy.data.objects['p3']
        p4 = bpy.data.objects['p4']
        p5 = bpy.data.objects['p5']
        p6 = bpy.data.objects['p6']
        p7 = bpy.data.objects['p7']
        p8 = bpy.data.objects['p8']
        #print(cco)

        pps = [p1, p2, p3, p4, p5, p6, p7, p8]

        for i, j in enumerate(ot):
            pco = get_co(pps[i])
            pco[:] = 0
            pco[np.arange(j.shape[0])] = co[j]

            pps[i].data.vertices.foreach_set('co', pco.ravel())
            pps[i].data.update()


        c1 = bpy.data.objects['c1']
        c2 = bpy.data.objects['c2']
        c3 = bpy.data.objects['c3']
        c4 = bpy.data.objects['c4']
        c5 = bpy.data.objects['c5']
        c6 = bpy.data.objects['c6']
        c7 = bpy.data.objects['c7']
        c8 = bpy.data.objects['c8']

        cs = [c1, c2, c3, c4, c5, c6, c7, c8]

        xn, yn, zn = min[0], min[1], min[2]
        xx, yx, zx = max[0], max[1], max[2]
        xd, yd, zd = mid[0], mid[1], mid[2]

        b0 = []




        cs[0].location = min
        cs[1].location = max
        cs[2].location = mid
        #we have eight boxes.
        #they all share the same center: mid
        #b1 might be


#        using the bounds of the collision
#        object as the max and min of the octree...
#        so like and stuff.... if the max of the collision
#        is poking into the cloth box, I don't need
#        to check anything but the cloth verts in
#        that box so I could use the box max
#        as the vert max if I add the margen from
#        the max and if I substract the margin
#        from the min...
#        maybe use either the collider
#        or the cloth depending on which is smaller?
#        so like if the avatar feet hang way down
#        there is no point checking the bounds all
#        the way down at the feet.
#        but if the cloth is out past the chest
#        the chest verts + margin make a better box.
#        We are getting the collider bounds anyway...
#
#        so collider min - margin, collider max + margin
#        cloth min, cloth max.
#
#        lesser of the two maxes.
#        greater of the two mins.



# universal ---------------------
def get_bary_weights(tris, points):
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
def get_normals_from_tris(tris):
    origins = tris[:, 0]
    vecs = tris[:, 1:] - origins[:, nax]
    cross = np.cross(vecs[:, 0], vecs[:, 1])
    mag = np.sqrt(np.einsum("ij ,ij->i", cross, cross))[:, nax]
    return np.nan_to_num(cross/mag)


# universal ---------------------
def cross_from_tris(tris):
    origins = tris[:, 0]
    vecs = tris[:, 1:] - origins[:, nax]
    cross = np.cross(vecs[:, 0], vecs[:, 1])
    return cross


# universal ---------------------
def apply_rotation(object, normals):
    """When applying vectors such as normals we only need
    to rotate"""
    m = np.array(object.matrix_world)
    mat = m[:3, :3].T
    #object.v_normals = object.v_normals @ mat
    return normals @ mat


# universal ---------------------
def revert_rotation(ob, co):
    """When reverting vectors such as normals we only need
    to rotate"""
    m = np.array(ob.matrix_world)
    mat = m[:3, :3] # rotates backwards without T
    return co @ mat


# universal ---------------------
def revert_transforms(ob, co):
    """Set world coords on object.
    Run before setting coords to deal with object transforms
    if using apply_transforms()"""
    m = np.linalg.inv(ob.matrix_world)
    mat = m[:3, :3]# rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


# universal ---------------------
def revert_in_place(ob, co):
    """Revert world coords to object coords in place."""
    m = np.linalg.inv(ob.matrix_world)
    mat = m[:3, :3]# rotates backwards without T
    loc = m[:3, 3]
    co[:] = co @ mat + loc


# universal ---------------------
def apply_in_place(ob, arr):
    """Overwrite vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    # optimize: np.dot(arr, mat, out=dot_arr)
    # optimize: np.add(dot_arr, loc, out=add_arr)
    arr[:] = arr @ mat + loc
    return arr


# universal ---------------------
def apply_transforms(ob, co):
    """Get vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


# universal ---------------------
def copy_object_transforms(ob1, ob2):
    M = ob1.matrix_world.copy()
    ob2.matrix_world = M


# universal ---------------------
def manage_locations_as_strings(old, new):
    # vertex coords for each point in old to a key as string
    """ !!! not finished !!! """
    idx_co_key = {}
    vs = ob.data.vertices
    flat_faces = np.hstack(sel_faces)
    for i in range(len(flat_faces)):
        idx_co_key[str(vs[flat_faces[i]].co)] = flat_faces[i]


# universal ---------------------
def offset_face_indices(faces=[]):
    """Sorts the original face vert indices
    for a new mesh from subset. Works with N-gons"""
    # Example: face_verts = [[20, 10, 30], [10, 30, 100, 105]]
    # Converts to [[1, 0, 2], [0, 2, 3, 4]]

    def add(c):
        c['a'] += 1
        return c['a']

    flat = np.hstack(faces)
    idx = np.unique(flat, return_inverse=True)[1]
    c = {'a': -1}
    new_idx = [[idx[add(c)] for j in i] for i in faces]
    return new_idx


# universal ---------------
def link_mesh(verts, edges, faces, name='name'):
    """Generate and link a new object from pydata"""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    mesh_ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(mesh_ob)
    return mesh_ob


# universal ---------------
def mesh_from_selection(ob, name='name'):
    """Generate a new mesh from selected faces"""
    obm = get_bmesh(ob)
    obm.faces.ensure_lookup_table()
    faces = [[v.index for v in f.verts] for f in obm.faces if f.select]
    if len(faces) == 0:
        print("No selected faces in mesh_from_selection")
        return # no faces

    obm.verts.ensure_lookup_table()
    verts = [i.co for i in obm.verts if i.select]
    idx_faces = offset_face_indices(faces)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], idx_faces)
    mesh.update()
    if False:
        mesh_ob = ob.copy()
    else:
        mesh_ob = bpy.data.objects.new(name, mesh)

    mesh_ob.data = mesh
    mesh_ob.name = name
    bpy.context.collection.objects.link(mesh_ob)
    return mesh_ob, faces, idx_faces


# universal ---------------
def get_bmesh(ob=None):
    """gets bmesh in editmode or object mode
    by checking the mode"""
    if ob.data.is_editmode:
        return bmesh.from_edit_mesh(ob.data)
    obm = bmesh.new()
    obm.from_mesh(ob.data)
    return obm


# universal ---------------------
def get_quad_obm(ob):
    """gets bmesh in editmode or object mode
    by updating the edit mesh"""
    ob.update_from_editmode()
    obm = bmesh.new()
    obm.from_mesh(ob.data)
    bmesh.ops.join_triangles(obm, faces=obm.faces, angle_shape_threshold=0.9, angle_face_threshold=0.9)
    return obm


# universal ---------------
def get_co(ob, ar=None):
    """Get vertex coordinates from an object in object mode"""
    c = len(ob.data.vertices)
    ar = np.empty((c, 3), dtype=np.float32)
    ob.data.vertices.foreach_get('co', ar.ravel())
    return ar


# universal ---------------
def co_overwrite(ob, ar):
    """Fast way to overwrite"""
    ob.data.vertices.foreach_get('co', ar.ravel())


# universal ---------------
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
        data[0] /= data[1]
        return data[0]

    pc = len(ob.data.polygons)
    pidex = np.hstack([[v for v in p.vertices] for p in ob.data.polygons])

    div = [len(p.vertices) for p in ob.data.polygons]

    indexer = []
    for i, j in enumerate(div):
        indexer += [i] * j
    div = np.array(div, dtype=np.float32)[:, None]

    centers = np.zeros((pc, 3), dtype=np.float32)

    np.add.at(centers, indexer, co[pidex])
    centers /= div

    return [centers, div, indexer, pidex]


# universal ---------------------
def get_poly_centers_bmesh(obm, co, data=None):
    """Get poly centers. Data is meant
    to be built the first time then
    passed in. (dynamic)"""

    if data is not None:
        data[0][:] = 0
        np.add.at(data[0], data[2], co[data[3]])
        data[0] /= data[1]
        return data[0]

    pc = len(obm.faces)
    pidex = np.hstack([[v.index for v in f.verts] for f in obm.faces])

    div = [len(f.verts) for f in obm.faces]

    indexer = []
    for i, j in enumerate(div):
        indexer += [i] * j
    div = np.array(div, dtype=np.float32)[:, None]

    centers = np.zeros((pc, 3), dtype=np.float32)

    np.add.at(centers, indexer, co[pidex])
    centers /= div

    return [centers, div, indexer, pidex]


# universal ---------------
def key_overwrite(ob, ar, key):
    """Fast way to overwrite"""
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())


# universal ---------------
def get_co_edit__(ob, obm=None):
    """Get vertex coordinates from an object in edit mode"""
    if obm is None:
        obm = get_bmesh(ob)
        obm.verts.ensure_lookup_table()
    co = np.array([i.co for i in obm.verts])
    return co


# universal ---------------
def Nx3(ob):
    """For generating a 3d vector array"""
    if ob.data.is_editmode:
        ob.update_from_editmode()
    count = len(ob.data.vertices)
    ar = np.zeros((count, 3), dtype=np.float32)
    return ar


# universal ---------------
def get_co_edit(ob, ar=None, key='MC_current'):
    if ar is None:
        c = len(ob.data.vertices)
        ar = np.empty((c, 3), dtype=np.float32)
    ob.update_from_editmode()
    #ob.data.vertices.foreach_get('co', ar.ravel())
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
    return ar


# universal ---------------
def get_co_mode(ob=None, key='MC_current'):
    """Edit or object mode"""
    if ob is None: # cloth.target object might be None
        return
    if ob.data.is_editmode:
        ob.update_from_editmode()
    c = len(ob.data.vertices)
    ar = np.empty((c, 3), dtype=np.float32)
    #ob.data.vertices.foreach_get('co', ar.ravel())
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
    return ar


# universal ---------------
def compare_geometry(ob1, ob2, obm1=None, obm2=None, all=False):
    """Check for differences in verts, edges, and faces between two objects"""
    # if all is false we're comparing a target objects. verts in faces
    #   and faces must match.
    def get_counts(obm):
        v_count = len([v for v in obm.verts if len(v.link_faces) > 0])
        #v_count = len(obm.verts) # not sure if I need to separate verts...
        e_count = len(obm.edges)
        f_count = len(obm.faces)
        if all:
            return np.array([v_count, e_count, f_count])
        return np.array([v_count, f_count]) # we can still add sew edges in theory...

    if obm1 is None:
        obm1 = get_bmesh(ob1)
    if obm2 is None:
        obm2 = get_bmesh(ob2)

    c1 = get_counts(obm1)
    c2 = get_counts(obm2)

    return np.all(c1 == c2)


# universal ---------------
def detect_changes(counts, obm):
    """Compare mesh data to detect changes in edit mode"""
    # counts in an np array shape (3,)
    v_count = len(obm.verts)
    e_count = len(obm.edges)
    f_count = len(obm.faces)
    new_counts = np.array([v_count, e_count, f_count])
    # Return True if all are the same
    return np.all(counts == new_counts), f_count < 1


# universal ---------------
def get_mesh_counts(ob, obm=None):
    """Returns information about object mesh in edit or object mode"""
    if obm is not None:
        v_count = len(obm.verts)
        e_count = len(obm.edges)
        f_count = len(obm.faces)
        return np.array([v_count, e_count, f_count])
    v_count = len(ob.data.vertices)
    e_count = len(ob.data.edges)
    f_count = len(ob.data.polygons)
    return np.array([v_count, e_count, f_count])


# universal ---------------
def get_weights(ob, name, obm=None, default=0, verts=None, weights=None):
    """Get vertex weights. If no weight is assigned
    set array value to zero. If default is 1 set
    all values to 1"""
    if obm is None:
        obm = get_bmesh(ob)
        obm.faces.ensure_lookup_table()
        obm.verts.ensure_lookup_table()

    # might want to look into using map()
    count = len(obm.verts)
    if name not in ob.vertex_groups:
        ob.vertex_groups.new(name=name)

    g_idx = ob.vertex_groups[name].index
    arr = np.zeros(count, dtype=np.float32)

    obm.verts.layers.deform.verify()

    deform = obm.verts.layers.deform.active

    dvert_lay = obm.verts.layers.deform.active

    if verts is not None:
        for i, v in enumerate(verts):
            dvert = obm.verts[v][dvert_lay]
            dvert[g_idx] = weights[i]
        return # will run again once the weights are set and build the arrays.

    if dvert_lay is None: # if there are no assigned weights
        return arr

    for v in obm.verts:
        idx = v.index
        dvert = v[dvert_lay]

        if g_idx in dvert:
            arr[idx] = dvert[g_idx]
        else:
            if default == 1:
                dvert[g_idx] = default
            arr[idx] = default

    return arr


# universal !!! broken !!! ---------------------
def box_bary_weights(poly, point, vals=[]):
    """Get values to plot points from tris
    or return the new plotted value."""
    # note: could use a lot of add.at/subtract.at and
    #   obscenely complex indexing to do bend springs
    #   between n-gons. It would be a riot!

    if vals: # two scalar values
        ba = poly[1] - poly[0]
        ca = poly[2] - poly[0]
        v1, v2 = vals[0], vals[1]
        plot = poly[0] + (((ba * v1) + (ca * v2)) * .5)
        #plot = poly[0]# + (((ba * v1) + (ca * v2)) * .5)
        return plot

    pa = point - poly[0]
    ba = poly[1] - poly[0]
    ca = poly[2] - poly[0]

    v1 = np.nan_to_num(pa / ba)
    v2 = np.nan_to_num(pa / ca)
    return [v1, v2]


# cloth setup -------------
def pairs_idx(ar):
    """Eliminates duplicates and mirror duplicates.
    for example, [1,4], [4,1] or duplicate occurrences of [1,4]
    Returns ar (array) and the index that removes the duplicates."""
    # no idea how this works (probably sorcery) but it's really fast
    a = np.sort(ar, axis=1) # because it only sorts on the second acess the index still matches other arrays.
    x = np.random.rand(a.shape[1])
    #x = np.linspace(1, 2, num=a.shape[1])
    y = a @ x
    unique, index = np.unique(y, return_index=True)
    return a[index], index



# cloth setup -------------
def reset_shapes(ob):
    """Create shape keys if they are missing"""
    if ob.MC_props.p1_bend:
        return

    if ob.data.shape_keys == None:
        ob.shape_key_add(name='Basis')

    keys = ob.data.shape_keys.key_blocks
    if 'MC_source' not in keys:
        ob.shape_key_add(name='MC_source')
        keys['MC_source'].value = 1.0

    if 'MC_current' not in keys:
        ob.shape_key_add(name='MC_current')
        keys['MC_current'].value = 1.0
        keys['MC_current'].relative_key = keys['MC_source']


# cloth setup -------------
def extend_bend_springs():
    #is there a way to get bend relationships from
    #just the basis springs...
    #Or could I do something like virtual springs
    #but for bend sets...
    pass


# abstract bend setup ----------------------------
# precalculated ------------------------
def get_j_surface_offset(cloth):
    """Get the vecs to move the plotted
    wieghts off the surface."""

    ax = cloth.j_axis_vecs
    ce = cloth.j_ce_vecs # has the faces swapped so the normal corresponds to the other side of the axis
    cross = np.cross(ax, ce)

    cloth.j_normals = cross / np.sqrt(np.einsum('ij,ij->i', cross, cross))[:, None]
    cloth.plot_normals = cloth.j_normals[cloth.j_tiler]

    cloth.bend_flat = False
    cloth.plot_vecs = cloth.sco[cloth.swap_jpv] - cloth.j_plot
    cloth.plot_dots = np.einsum('ij,ij->i', cloth.plot_normals, cloth.plot_vecs)[:, None]
    if np.all(cloth.plot_dots < 0.000001):
        cloth.bend_flat = True


# abstract bend setup ----------------------------
# dynamic ------------------------------
def measure_linear_bend(cloth):
    """Takes a set of coords and an edge idx and measures segments"""
    l = cloth.sp_ls # left side of the springs (Full moved takes the place of the right side)
    a,b,c = np.unique(l, return_counts=True, return_inverse=True)

    x = c[b]
    cloth.bend_multiplier = ((x - 2) / 2) + 2
    return

    v = (cloth.full_moved - cloth.co[cloth.sp_ls]) / cloth.divy
    d = np.einsum("ij ,ij->i", v, v)
    return v, d, np.sqrt(d)


# abstract bend setup ----------------------------
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


# abstract bend setup ----------------------------
# precalculated ------------------------
def eq_bend_data(cloth):
    """Generates face pairs around axis edges.
    Supports edges with 2-N connected faces.
    Can use internal structures this way."""
    ob = cloth.ob
    if ob.MC_props.quad_bend:
        obm = cloth.quad_obm
    else:
        obm = get_bmesh(ob)
    sco = cloth.sco

    # eliminate: sew edges, outer edges, bend_stiff group weight 0:
    ed = [e for e in obm.edges if
         (len(e.link_faces) > 1) &
         (cloth.bend_cull[e.verts[0].index]) &
         (cloth.bend_cull[e.verts[1].index])]

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
    t = time.time()
    cloth.face_pairs, idx = pairs_idx(paired)
    print(time.time()-t)
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


# abstract bend setup ----------------------------
# precalculated ------------------------
def get_poly_vert_tilers(cloth):
    """Get an index to tile the left and right sides.
    ls and rs is based on the left and right sides of
    the face pairs."""

    if cloth.ob.MC_props.quad_bend:
        obm = cloth.quad_obm
    else:
        obm = cloth.obm
        obm.faces.ensure_lookup_table()

    cloth.swap_jpv = []
    cloth.jpv_full = []
    ob = cloth.ob

    cloth.ab_faces = []
    cloth.ab_edges = []

    count = 0
    for i, j in zip(cloth.swap_faces, cloth.stacked_edv): # don't need to swap edv because both sides share the same edge

        pvs = [v.index for v in obm.faces[i].verts]
        nar = np.array(pvs)
        b1 = nar != j[0]
        b2 = nar != j[1]

        nums = np.arange(nar.shape[0]) + count
        cloth.ab_faces += nums[b1 & b2].tolist()
        cloth.ab_edges += nums[~(b1)].tolist()
        cloth.ab_edges += nums[~(b2)].tolist()

        count += nar.shape[0]
        r = [v.index for v in obm.faces[i].verts if v.index not in j]
        cloth.swap_jpv += r

    for i in cloth.swap_faces:
        r = [v.index for v in obm.faces[i].verts]
        cloth.jpv_full += r


# abstract bend setup ----------------------------
# precalculated ------------------------
def tiled_weights(cloth):
    """Tile the tris with the polys for getting
    barycentric weights"""
    if cloth.ob.MC_props.quad_bend:
        obm = cloth.quad_obm
    else:
        obm = cloth.obm

    ob = cloth.ob
    face_pairs = cloth.face_pairs

    # counts per poly less the two in the edges
    cloth.full_counts = np.array([len(f.verts) for f in obm.faces], dtype=np.int32)
    cloth.full_div = np.array(cloth.full_counts, dtype=np.float32)[cloth.swap_faces][:, None]
    cloth.plot_counts = cloth.full_counts - 2 # used by plotted centers

    # joined:
    jfps = cloth.stacked_faces.shape[0]

    jsc = cloth.plot_counts[cloth.swap_faces]
    cloth.j_tiler = np.hstack([[i] * jsc[i] for i in range(jfps)])

    if False:
        cloth.js_tris = cloth.j_tris[cloth.j_tiler]

    jscf = cloth.full_counts[cloth.swap_faces]
    cloth.ab_tiler = np.hstack([[i] * jscf[i] for i in range(jfps)])
    cloth.sp_ls = np.hstack([[v.index for v in obm.faces[f].verts] for f in cloth.swap_faces])


# abstract bend setup ----------------------------
# precalculated ------------------------
def triangle_data(cloth):

    sco = cloth.sco
    edv = cloth.edv

    # joined tris:
    j_tris = np.zeros((cloth.j_tips.shape[0], 3, 3), dtype=np.float32)
    j_tris[:, :2] = sco[cloth.stacked_edv]
    j_tris[:, 2] = cloth.j_tips
    cloth.j_tris = j_tris
    #-----------------

    # get the tilers for creating tiled weights
    tiled_weights(cloth)

    if False:
        jw = get_bary_weights(cloth.js_tris, sco[cloth.swap_jpv])
        cloth.j_plot = np.sum(cloth.js_tris * jw[:,:,None], axis=1)
        get_j_surface_offset(cloth)
        cloth.jw = jw

    trial = False
    trial = True
    if trial:
        # can probably speed this up by merging the arrays then slicing
        tips, ax, ce = get_eq_tri_tips(cloth, cloth.sco, cloth.source_centers, skip=False)
        c1 = np.cross(ax, ce)
        c2 = np.cross(c1, ax)

        Uax = ax / np.sqrt(np.einsum('ij,ij->i', ax, ax))[:, None]
        Uc1 = c1 / np.sqrt(np.einsum('ij,ij->i', c1, c1))[:, None]
        Uc2 = c2 / np.sqrt(np.einsum('ij,ij->i', c2, c2))[:, None]

        j_mid = sco[cloth.stacked_edv[:,0]] + (ax * 0.5)

        vecs = sco[cloth.swap_jpv] - j_mid[cloth.j_tiler]

        cloth.d1 = np.einsum('ij,ij->i', vecs, Uax[cloth.j_tiler])
        cloth.d2 = np.einsum('ij,ij->i', vecs, Uc1[cloth.j_tiler])
        cloth.d3 = np.einsum('ij,ij->i', vecs, Uc2[cloth.j_tiler])
        cloth.is_flat = np.all(np.abs(cloth.d2) < 0.00001)

# abstract bend setup ----------------------------
# precalculated ------------------------
def linear_bend_set(cloth): # !!! not currently used !!!
    ab = np.array(cloth.jpv_full)

    springs = []
    vc = len(cloth.ob.data.vertices)
    for i in range(vc):
        w = np.where(i == ab)[0]
        for j in w:
            springs.append([i,j])

    cloth.linear_bend_springs = np.array(springs, dtype=np.int32)
    cloth.bend_v_fancy = cloth.linear_bend_springs[:,0]


# abstract bend setup ----------------------------
# precalculated ------------------------
def ab_setup(cloth):
    cloth.ab_centers = np.empty((cloth.stacked_faces.shape[0], 3), dtype=np.float32)
    cloth.ab_coords = np.empty((len(cloth.jpv_full), 3), dtype=np.float32)
    #cloth.bend_stretch_array = np.zeros((cloth.co.shape[0], 3), dtype=np.float32)
    l = cloth.sp_ls # left side of the springs (Full moved takes the place of the right side)
    a,b,c = np.unique(l, return_counts=True, return_inverse=True)
    x = c[b] # number of times each vert occurs
    #cloth.bend_multiplier = 1 / (((x - 2.5) / 2.5) + 2.5)[:, None]
    if False:
        print(x[x == 1].shape, " == 1")
        print(x[x == 2].shape, " == 2")
        print(x[x == 3].shape, " == 3")
        print(x[x == 4].shape, " == 4")
        print(x[x == 5].shape, " == 5")
        print(x[x == 6].shape, " == 6")
        print(x[x == 7].shape, " == 7")
        print(x[x == 8].shape, " == 8")

    #x[x < 4] *= 4


    x[x < 8] *= 2

    #print(x[x > 2], "printign this")
    cloth.bend_multiplier = 1 / x[:, None] #(((x - 2.5) / 2.5) + 2.5)[:, None]

    #cloth.bend_multiplier = 1 / (x + (10 / x))[:, None] #(((x - 2.5) / 2.5) + 2.5)[:, None]
    #cloth.bend_multiplier = 1 / ((x / 2) + 1)[:, None] #(((x - 2.5) / 2.5) + 2.5)[:, None]

    # multiplying the vertex group here (for some reason...)
    cloth.bend_group_mult = cloth.bend_group[l]
    cloth.bend_multiplier *= cloth.bend_group_mult


    #cloth.divy = np.clip(c[b][:, None], 0, 12)


    #cloth.bend_group_mult = cloth.bend_group[cloth.sp_ls]
    #for i in cloth.bend_group_mult:
        #print(i)



# abstract bend setup ----------------------------
# precalculated ------------------------
def bend_setup(cloth):
    # if I end up using cross products and unit
    #   vecs instead of wieghts, I can
    #   add the center of the edge
    #   once at the end
    #   instead of doing it for all three

    if cloth.ob.MC_props.quad_bend:
        cloth.quad_obm = get_quad_obm(cloth.ob)
        cloth.quad_obm.faces.ensure_lookup_table()
        cloth.center_data = get_poly_centers_bmesh(cloth.quad_obm, cloth.sco, data=None)
    else:
        cloth.center_data = get_poly_centers(cloth.ob, cloth.sco, data=None)

    cloth.source_centers = np.copy(cloth.center_data[0]) # so we can overwrite the centers array when dynamic
    eq_bend_data(cloth)
    get_poly_vert_tilers(cloth)
    get_eq_tri_tips(cloth, cloth.sco, cloth.source_centers)
    triangle_data(cloth)
    ab_setup(cloth)
    #linear_bend_set(cloth)


# abstract bend setup ----------------------------
# dynamic ------------------------------
def dynamic(cloth):

    # get centers from MC_current
    centers = get_poly_centers(cloth.ob, cloth.co, cloth.center_data)
    co = cloth.co

    if False:
        #cloth.j_tris[:] = 0
        cloth.j_tris[:, :2] = co[cloth.stacked_edv]
        tips, ax, ce = get_eq_tri_tips(cloth, co, centers, skip=False)

        cloth.j_tris[:, 2] = tips
        jw = cloth.jw
        j_plot = np.sum(cloth.j_tris[cloth.j_tiler] * jw[:,:,None], axis=1)

        # for just flattening
        final_plot = j_plot
        flat = False

        if not cloth.bend_flat:
            cross = np.cross(ax, ce)
            normals = cross / np.sqrt(np.einsum('ij,ij->i', cross, cross))[:, None]
            plot_normals = normals[cloth.j_tiler]
            final_plot = j_plot + (plot_normals * cloth.plot_dots)

    trial = False
    trial = True
    if trial:
        tips, ax, ce = get_eq_tri_tips(cloth, co, centers, skip=False)
        c1 = np.cross(ax, ce)
        c2 = np.cross(c1, ax)

        Uax = ax / np.sqrt(np.einsum('ij,ij->i', ax, ax))[:, None]
        Uc1 = c1 / np.sqrt(np.einsum('ij,ij->i', c1, c1))[:, None]
        Uc2 = c2 / np.sqrt(np.einsum('ij,ij->i', c2, c2))[:, None]

        #j_axis_vecs = co[cloth.stacked_edv[:,1]] - co[cloth.stacked_edv[:,0]]

        j_mid = co[cloth.stacked_edv[:,0]] + (ax * 0.5)

        p1 = Uax[cloth.j_tiler] * cloth.d1[:, None]
        p3 = Uc2[cloth.j_tiler] * cloth.d3[:, None]

        origin = j_mid[cloth.j_tiler]

        if not cloth.is_flat:
            p2 = Uc1[cloth.j_tiler] * cloth.d2[:, None]
            final_plot = p1 + p2 + p3 + origin

        if cloth.is_flat:
            final_plot = p1 + p3 + origin


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


# abstract bend setup ----------------------------
# dynamic ------------------------------
def abstract_bend_(cloth):
    dynamic(cloth)
    stretch = cloth.ob.MC_props.bend# * 100
    cv = (cloth.full_moved - cloth.co[cloth.sp_ls])# * mult

    if False:
        w = cloth.bend_stretch_array
        w[:] = 0.0

        d = np.einsum("ij ,ij->i", cv, cv)
        l = np.sqrt(d)
        #l = d #np.sqrt(d)
        np.add.at(w, cloth.sp_ls, l)
        # !!! don't forget to add the vertex weights back in
        mult = (l / w[cloth.sp_ls]) * stretch # * weights

        #print(cv[0])
        cv *= mult[:, None]
        #print(cv[0])
    else:
        mult = cloth.bend_multiplier * stretch
        cv *= mult
    #print(cloth.bend_multiplier)
    #np.add.at(cloth.co, cloth.sp_ls, np.nan_to_num(cv * stretch))
    #print(np.sum(cv, axis=0))
    np.add.at(cloth.co, cloth.sp_ls, np.nan_to_num(cv))# * stretch))
    return

def abstract_bend(cloth):
    dynamic(cloth)
    # (current vec, dot, length)
    #cv = measure_edges(cloth.co, cloth.basic_set, cloth) # from current cloth state
    cv = (cloth.full_moved - cloth.co[cloth.sp_ls])
    cd = np.einsum("ij ,ij->i", cv, cv)
    cl = np.sqrt(cd)

    move_l = cl

    # mean method -------------------
    cloth.bend_stretch_array[:] = 0.0

    rock_hard_abs = move_l
    np.add.at(cloth.bend_stretch_array, cloth.sp_ls, rock_hard_abs)
    weights = (rock_hard_abs / cloth.bend_stretch_array[cloth.sp_ls]) * cloth.ob.MC_props.bend

    move = cv# * ((move_l / cl) * cloth.ob.MC_props.bend)[:,None]

    move *= weights[:,None]

    np.add.at(cloth.co, cloth.sp_ls, np.nan_to_num(move))


    # there is a difference between the edges that get evaled
    #   and the points that get moved by the bend weights...
    # In some cases this creates imbalance in the forces...

#                                                                #
#                                                                #
# ------------------- end abstract bend data ------------------- #



# universal ---------------
def inside_triangles(tris, points, check=True, surface_offset=False, cloth=None):
    """Can check inside triangle.
    Can find barycentric weights for triangles.
    Can find multiplier for distance off surface
        from non-unit cross product."""
    origins = tris[:, 0]
    cross_vecs = tris[:, 1:] - origins[:, nax]
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
    if not check | surface_offset:
        return weights

    if not check:
        cross = np.cross(cross_vecs[:,0], cross_vecs[:,1])
        d_v2_c = np.einsum('ij,ij->i', v2, cross)
        d_v2_v2 = np.einsum('ij,ij->i', cross, cross)
        div = d_v2_c / d_v2_v2

        U_cross = cross / np.sqrt(d_v2_v2)[:, None]
        U_d = np.einsum('ij,ij->i', v2, U_cross)


        return weights, div, U_d# for normalized

    check = np.all(weights > 0, axis=0)
    # check if bitwise is faster when using lots of tris
    if False:
        check = (u > 0) & (v > 0) & (u + v < 1)

    return weights.T, check


def cpoe_bend_plot_values(cloth):

    # get axis vecs
    co = get_co_shape(cloth.ob, key="MC_source")
    be = cloth.bend_edges
    tt = cloth.bend_tri_tips.reshape(be.shape)   #[:, ::-1] # flip it to the other side
    po_vecs = co[tt] - co[be[:, 0]][:, None]
    tris = co[tt]
    origins = co[be[:, 0]]

    # get axis plot (non-unit vecs so it will scale? I think this makes sense...)
    axis_vecs = co[be[:, 1]] - origins
    d_po = np.einsum('ij, ikj->ik', axis_vecs, po_vecs)
    d_axis = np.einsum('ij,ij->i', axis_vecs, axis_vecs)
    # --------------------------------------
    cloth.axis_div = np.nan_to_num(d_po / d_axis[:, None])
    """
    A pair of axis dots for each axis. One for each tri tip, and a partridge in a pair tree.
    """

    # get cross plot (unit vecs here stabilize)
    cross_vecs = tris - origins[:, None]
    cross = np.roll(np.cross(cross_vecs, axis_vecs[:, None]), 1, axis=1)
    n_2_3 = cross.shape
    n_3 = (n_2_3[0] *2, 3)
    cross.shape = n_3
    po_vecs.shape = cross.shape
    U_cross = cross / np.sqrt(np.einsum('ij, ij->i', cross, cross))[:, None]
    # --------------------------------------
    cloth.cross_div = np.einsum('ij, ij->i', U_cross, po_vecs)[:, None]
    """
    How far off the surface the opposite tri tip is along the normal
    """

    # get tri plot (unit vecs here stabilize)
    cross.shape = n_2_3
    tri_vecs = np.cross(cross, axis_vecs[:, None])
    tri_vecs.shape = n_3
    cross.shape = n_3
    d_tv = np.einsum('ij, ij->i', tri_vecs, tri_vecs)
    U_tri = tri_vecs / np.sqrt(d_tv)[:, None]
    # --------------------------------------
    cloth.tri_div = np.einsum('ij, ij->i', U_tri, po_vecs)[:, None]
    """
    Cross of the normal and the axis pointing perpindicular to the axis
    along the surface of the tri opposite the tip.
    """


def cpoe_bend_plot(cloth):
    """Plot values based on cpoe using axis and cross products"""

    axis_div = cloth.axis_div
    tri_div = cloth.tri_div
    cross_div = cloth.cross_div

    co = cloth.co
    be = cloth.bend_edges
    tt = cloth.bend_tri_tips.reshape(be.shape)   #[:, ::-1] # flip it to the other side
    po_vecs = co[tt] - co[be[:, 0]][:, None]
    tris = co[tt]
    origins = co[be[:, 0]]

    # plot axis (origin added here)
    axis_vecs = co[be[:, 1]] - origins
    axis_plot = origins[:, None] + axis_vecs[:, None] * axis_div[:,:,None]

    # plot normal
    cross_vecs = tris - origins[:, None]
    cross = np.roll(np.cross(cross_vecs, axis_vecs[:, None]), 1, axis=1)
    n_2_3 = cross.shape
    n_3 = (n_2_3[0] * 2, 3)
    cross.shape = n_3
    po_vecs.shape = cross.shape
    U_cross = cross / np.sqrt(np.einsum('ij, ij->i', cross, cross))[:, None]
    cross_plot = np.nan_to_num(U_cross) * cross_div

    # plot along tri surface
    cross.shape = n_2_3
    tri_vecs = np.cross(cross, axis_vecs[:, None])
    tri_vecs.shape = n_3
    cross.shape = n_3

    d_tv = np.sqrt(np.einsum('ij, ij->i', tri_vecs, tri_vecs))
    U_tri = tri_vecs / d_tv[:, None]

    tri_plot = np.nan_to_num(U_tri) * tri_div

    axis_plot.shape = n_3
    plot = tri_plot + axis_plot + cross_plot

    return plot


def get_cloth(ob):
    """Return the cloth instance from the object"""
    return MC_data['cloths'][ob['MC_cloth_id']]

# ^                                                          ^ #
# ^                 END universal functions                  ^ #
# ============================================================ #




# ============================================================ #
#                   precalculated data                         #
#                                                              #

# precalculated ---------------
def closest_point_mesh(obm, ob, idx, target):
    """Uses bmesh method to get CPM"""
    context = bpy.context
    scene = context.scene

    obm.verts.ensure_lookup_table()
    tmwi = target.matrix_world.inverted()

    locs = []
    faces = []
    norms = []
    cos = []
    for i in idx:
        co = ob.matrix_world @ obm.verts[i].co # global face median
        #co = obm.verts[i].co # global face median
        cos.append(co)
        local_pos = tmwi @ co  # face cent in target local space

        (hit, loc, norm, face_index) = target.closest_point_on_mesh(local_pos)
        #print(hit, loc, norm, face_index)
        if hit:
            #locs.append(target.matrix_world @ loc)
            locs.append(loc)
            faces.append(face_index)
            norms.append(norm)
            empties = True
            empties = False
            if empties:
                bpy.ops.object.empty_add(location=target.matrix_world @ loc)

    print("need to create dtypes for these array below")
    return np.array(locs), np.array(faces), np.array(norms), np.array(cos)


def get_tridex(ob, tobm=None):
    """Return an index for viewing the verts as triangles"""
    free = False
    if tobm is None:
        tobm = bmesh.new()
        tobm.from_mesh(ob.data)
        free = True
    bmesh.ops.triangulate(tobm, faces=tobm.faces[:])
    tridex = np.array([[v.index for v in f.verts] for f in tobm.faces], dtype=np.int32)
    if free:
        tobm.free()
    return tridex


def get_tridex_2(ob, mesh=None): # faster than get_tridex()
    """Return an index for viewing the
    verts as triangles using a mesh and
    foreach_get"""
    if ob.data.is_editmode:
        ob.update_from_editmode()

    if mesh is not None:
        tobm = bmesh.new()
        tobm.from_mesh(mesh)
        bmesh.ops.triangulate(tobm, faces=tobm.faces)
        me = bpy.data.meshes.new('tris')
        tobm.to_mesh(me)
        p_count = len(me.polygons)
        tridex = np.empty((p_count, 3), dtype=np.int32)
        me.polygons.foreach_get('vertices', tridex.ravel())

        # clear unused tri mesh
        bpy.data.meshes.remove(me)
        return tridex

    tobm = bmesh.new()
    tobm.from_mesh(ob.data)
    bmesh.ops.triangulate(tobm, faces=tobm.faces[:])
    me = bpy.data.meshes.new('tris')
    tobm.to_mesh(me)
    p_count = len(me.polygons)
    tridex = np.empty((p_count, 3), dtype=np.int32)
    me.polygons.foreach_get('vertices', tridex.ravel())

    # clear unused tri mesh
    bpy.data.meshes.remove(me)

    return tridex, tobm


def get_bend_sets(cloth):
    """Each edge with two faces makes a bend edge.
    The mesh is triangluated so the tip of each triangle
    whose base is the edge is moved like a hinge
    around the bend edge."""

    ob = cloth.ob
    tridex, obm = get_tridex_2(ob)
    obm.edges.ensure_lookup_table()
    T = time.time()
    bend_data = np.array([[e.verts[0].index, e.verts[1].index, e.index] for e in obm.edges if len(e.link_faces) == 2], dtype=np.int32)
    if bend_data.shape[0] == 0:
        cloth.bend_data = None
        return
    bend_edges = bend_data[:, :2]
    ed_idx = bend_data[:, -1:].ravel()

    lf = [obm.edges[e].link_faces for e in ed_idx]
    fidx = np.array([[i[0].index, i[1].index] for i in lf])

    c = tridex[fidx.ravel()]
    a = bend_edges.ravel()[:, None]
    b = bend_edges[:, ::-1].ravel()[:, None]

    bool1 = a == c
    bool2 = b == c

    tri_tips = c[~(bool1 | bool2)]
    # switch tri_tips so the tip pairs with the tri
    #   on the opposite side of the edge.
    shape = tri_tips.shape
    tri_tips.shape = (shape[0]//2, 2)
    tri_tips = tri_tips[:,::-1].ravel()

    # write to class instance
    cloth.tridex = tridex
    cloth.bend_ed_idx = ed_idx
    cloth.bend_edges = bend_edges
    cloth.bend_tri_tips = tri_tips
    cloth.bend_tris = c
    obm.free()

    # can be dynamic
    bary_bend_springs(cloth)
    cloth.bend_tri_tip_array = np.zeros(len(cloth.ob.data.vertices), dtype=np.float32)#[:, None]


def bary_bend_springs(cloth):
    """Dynamic portion of bend springs
    (as opposed to index groups)"""
    co = get_co_shape(cloth.ob, key='MC_source')
    tris = co[cloth.bend_tris]
    points = co[cloth.bend_tri_tips]
    weights, surface_offset, U_d = inside_triangles(tris, points, check=False, surface_offset=True, cloth=cloth)
    # 44444
    cloth.zero_surface_bool = None
    cloth.bend_weights = weights[:,:,None]
    cloth.bend_surface_offset = surface_offset[:, None]
    cloth.bend_U_d = U_d


# precalculated ---------------
def create_surface_follow_data(active, cloths):
    """Need to run this every time
    we detect 'not same' """

    # !!! need to make a global dg shared between cloth objects???
    selection = active.MC_props.surface_follow_selection_only

    for c in cloths:
        cloth = MC_data['cloths'][c['MC_cloth_id']]
        proxy = active.evaluated_get(cloth.dg)
        cloth.surface = True
        cloth.surface_object = proxy

        # temp -------------------------------------------
        #mesh_ob = proxy.copy()
        #bpy.context.collection.objects.link(mesh_ob)
        #mesh_ob.data.update()
        # temp -------------------------------------------

        cloth.surface_object = proxy
        vw = get_weights(c, 'MC_surface_follow')
        idx = np.where(vw != 0)[0]
        cloth.bind_idx = idx

        print("have to run the surface calculations when updating groups")

        if selection:
            sel_object, sel_faces, idx_faces = mesh_from_selection(proxy, "temp_delete_me")
            if sel_object is None:
                return 'Select part of the mesh or turn off "Selection Only"'

            # triangulate selection object mesh for barycentric weights
            sel_obm = bmesh.new()
            sel_obm.from_mesh(sel_object.data)
            bmesh.ops.triangulate(sel_obm, faces=sel_obm.faces[:])
            sel_obm.to_mesh(sel_object.data)
            sel_object.data.update()
            copy_object_transforms(proxy, sel_object)
            sel_tridex = get_tridex(sel_object, sel_obm)
            sel_obm.free()

            # find barycentric data from selection mesh
            locs, faces, norms, cos = closest_point_mesh(cloth.obm, c, idx, sel_object)

            # converts coords to dict keys so we can find corresponding verts in surface follow mesh
            idx_co_key = {}
            pvs = proxy.data.vertices
            flat_faces = np.hstack(sel_faces)
            for i in range(len(flat_faces)):
                idx_co_key[str(pvs[flat_faces[i]].co)] = flat_faces[i]

            # use keys to get correspoinding vertex indices
            flat_2 = np.hstack(sel_tridex[faces])
            co_key_2 = []
            vs = sel_object.data.vertices
            tri_keys = [str(vs[i].co) for i in flat_2]
            surface_follow_tridex = [idx_co_key[i] for i in tri_keys]

            # convert to numpy and reshape
            cloth.surface_tridex = np.array(surface_follow_tridex)

            cloth.surface_tris_co = np.array([pvs[i].co for i in cloth.surface_tridex], dtype=np.float32)
            shape = cloth.surface_tridex.shape[0]
            cloth.surface_tridex.shape = (shape // 3, 3)
            cloth.surface_tris_co.shape = (shape // 3, 3, 3)
            #v = proxy.data.vertices
            #for i in cloth.surface_tridex.ravel():
                #bpy.ops.object.empty_add(location=proxy.matrix_world @ v[i].co)
            #print(tri_keys)

            # delete temp mesh
            me = sel_object.data
            bpy.data.objects.remove(sel_object)
            bpy.data.meshes.remove(me)

            # generate barycentric weights
            cloth.surface_bary_weights = inside_triangles(cloth.surface_tris_co, locs, check=False)


            dif = cos - apply_transforms(active, locs)
            #cloth.surface_norms = norms
            cloth.surface_norm_vals = np.sqrt(np.einsum("ij ,ij->i", dif, dif))[:, nax]

            # Critical values for barycentric placement:
            #   1: cloth.surface_tridex          (index of tris in surface object)
            #   2: cloth.surface_bary_weights    (barycentric weights)
            #   3: cloth.surface_object          (object we're following)
            #   4: cloth.surface                 (bool value indicating we should run surface code)
            #   5: cloth.bind_idx = idx          (verts that are bound to the surface)
            #   6: cloth.surface_norms = norms   (direction off surface of tris)
            #   7: cloth.surface_normals = norms (mag of surface norms)


def virtual_springs(cloth):
    """Adds spring sets checking if springs
    are already present using strings.
    Also stores the set so we can check it
    if there are changes in geometry."""

    # when we detect changes in geometry we
    #   regenerate the springs, then add
    #   cloth.virtual_springs to the basic
    #   array after we check that all the
    #   verts in the virtual springs are
    #   still in the mesh.
    verts = cloth.virtual_spring_verts
    ed = cloth.basic_set

    string_spring = [str(e[0]) + str(e[1]) for e in ed]

    strings = []
    new_ed = []
    for v in verts:
        for j in verts:
            if j != v:
                stringy = str(v) + str(j)
                strings.append(stringy)
                #if stringy not in string_spring:
                new_ed.append([v,j])

    in1d = np.in1d(strings, string_spring)
    cull_ed = np.array(new_ed)[~in1d]
    cloth.virtual_springs = cull_ed # store it for checking when changing geometry
    cloth.basic_set = np.append(cloth.basic_set, cull_ed, axis=0)
    cloth.basic_v_fancy = cloth.basic_set[:,0]
    # would be nice to have a mesh or ui magic to visualise virtual springs
    # !!! could do a fixed type sew spring the same way !!!
    # !!! maybe use a vertex group for fixed sewing? !!!
    # !!! Could also use a separate object for fixed sewing !!!


def get_springs_2(cloth):
    """Create index for viewing stretch springs"""
    obm = cloth.obm
    obm.verts.ensure_lookup_table()

    if not cloth.do_stretch:
        return


    TT = time.time()
    verts = np.arange(len(obm.verts))[cloth.linear_cull.ravel()]
    ed = []
    for vc in verts:
        v = obm.verts[vc]
        #if cloth.linear_cull[v.index]:
        v_set = []
        for f in v.link_faces:
            for vj in f.verts:
                if vj != v:
                    stri = str(v.index) + str(vj.index)
                    if stri not in v_set:
                        ed.append([vc, vj.index])
                        v_set.append(stri)


    cloth.basic_set = np.array(ed)
    cloth.basic_v_fancy = cloth.basic_set[:,0]
    cloth.stretch_group_mult = cloth.stretch_group[cloth.basic_v_fancy]


# ^                                                          ^ #
# ^                 END precalculated data                   ^ #
# ============================================================ #

# ============================================================ #
#                      cloth instance                          #
#                                                              #

def manage_vertex_groups(cloth):
    """Create vertex groups and generate
    numpy arrays for them."""

    ob = cloth.ob
    #obm = cloth.obm
    #if ob.data.is_editmode:
    #ob.data.update()
    obm = get_bmesh(ob)

    vw = 1 # so we avoid calculating except where the spring values are more than zero
    if ob.MC_props.dense:
        vw = 0

    np_groups = []
    groups = [('MC_pin',            0.0),
              ('MC_drag',           0.0),
              ('MC_surface_follow', 0.0),
              ('MC_bend_stiff',      vw),
              ('MC_stretch',         vw)]

    for i in groups:
        np_groups.append(get_weights(ob, i[0], obm, i[1]))

    cloth.pin            = np_groups[0][:, None]
    cloth.drag           = np_groups[1]
    cloth.surface_follow = np_groups[2]
    cloth.bend_group     = np_groups[3][:, None]
    cloth.stretch_group  = np_groups[4][:, None]

    cloth.bend_cull = cloth.bend_group > 0

    cloth.do_bend = False
    if np.any(cloth.bend_cull):
        cloth.do_bend = True
        if not ob.MC_props.p1_bend:
            cloth.sco = get_co_shape(cloth.ob, key='MC_source', ar=None)
        bend_setup(cloth)

    cloth.linear_cull = cloth.stretch_group > 0
    cloth.do_stretch = False
    if np.any(cloth.linear_cull):
        cloth.do_stretch = True
        get_springs_2(cloth)

    cloth.skip_bend_wieght = np.all(cloth.bend_group == 1) # for optimizing by elimating multiplier
    cloth.skip_stretch_wieght = np.all(cloth.stretch_group == 1) # for optimizing by elimating multiplier
    # when running in p1 just write to these two groups
    # before turning on cloth.

    if ob.data.is_editmode:
        bmesh.update_edit_mesh(ob.data)
        return

    cloth.obm.to_mesh(ob.data)
    cloth.ob.data.update()


# collider instance -----------------
class Collider(object):
    # The collider object
    name = 'inital name'

    def __init__(self):
        self.ob = bpy.context.object
        self.name = 'inital name'

    def refresh(self):
        # for building the collider data
        print(self.ob.name, "Object name of collider")
        self.name = "changed name"
        if self.ob.data.is_editmode:
            self.ob.update_from_editmode()
        proxy = get_proxy(self.ob)
        self.face_count = len(proxy.polygons)
        self.tridex = get_tridex_2(self.ob, proxy)
        self.pco = np.empty((len(proxy.vertices), 3), dtype=np.float32)
        proxy.vertices.foreach_get('co', self.pco.ravel())
        self.ob.to_mesh_clear()

    def soft_refresh(self):
        # consider doing this for cloth and sorting
        # out like vertex groups from changes in geometry.
        print('for changes that do not require full refresh')


# cloth instance ---------------
class Cloth(object):
    # The cloth object
    def __init__(self):
        pass

    def refresh(self):
        print('plan to move here for sorting')

    def soft_refresh():
        # like vertex weights without remaking all the springs
        print('does not require recalculating springs')


# cloth instance ---------------
def create_instance(ob=None):
    """Run this when turning on modeling cloth."""
    cloth = Cloth()
    cloth.dg = bpy.context.evaluated_depsgraph_get()
    if ob is None:
        ob = bpy.context.object
    cloth.ob = ob

    cloth.obm = get_bmesh(cloth.ob)

    cloth.p1_trigger = False

    if ob.MC_props.p1_bend:

        cloth.sco = get_co_shape(ob, 'flat') # will be overwritten if there is a flat shape key
        cloth.co = get_proxy_co(ob) # will be overwritten if there is a flat shape key

    if ob.MC_props.cache_only:
        cloth.target = None
        cloth.obm = get_bmesh(ob)
        cloth.update_lookup = True
        cloth.mode = 1
        if ob.data.is_editmode:
            cloth.mode = None
        cloth.undo = False
        cloth.selected = np.array([True, True], dtype=np.bool)
        return cloth # don't calculate all this crap if we're just caching

    #if not ob.MC_props.p1_bend:
    manage_vertex_groups(cloth)

    cloth.update_lookup = False
    cloth.selected = np.zeros(len(ob.data.vertices), dtype=np.bool)
    cloth.grab = np.zeros(len(ob.data.vertices), dtype=np.bool)
    cloth.current_cache_frame = 1 # for the cache continuous playback
    cloth.shape_update = False

    # surface follow data ------------------------------------------
    cloth.surface_tridex = None         # (index of tris in surface object)
    cloth.surface_bary_weights = None   # (barycentric weights)
    cloth.surface_object = None         # (object we are following)
    cloth.surface = False               # (bool value indicating we should run surface code)
    cloth.bind_idx = None               # (verts that are bound to the surface)

    cloth.undo = False

    # the target for stretch and bend springs
    cloth.proxy = None
    cloth.target = None # (gets overwritten by def cb_target)
    cloth.target_obm = None
    cloth.target_undo = False
    cloth.target_geometry = None # (gets overwritten by cb_target)
    cloth.target_mode = 1
    if cloth.target is not None:
        if cloth.target.data.is_editmode:
            cloth.target_mode = None

    # for detecting changes in geometry
    cloth.geometry = get_mesh_counts(ob)

    # for detecting mode changes
    cloth.mode = 1
    if ob.data.is_editmode:
        cloth.mode = None

    #t = T()
    #get_springs_2(cloth)
    #T(t, "time it took to get the springs")

    # coordinates and arrays ------------
    if not ob.MC_props.p1_bend:
        if ob.data.is_editmode:
            cloth.co = get_co_mode(ob)
        else:
            cloth.co = get_co_shape(ob, 'MC_current')

    cloth.wco = np.copy(cloth.co)
    apply_in_place(cloth.ob, cloth.wco)

    # abstract bend setup:
    #if cloth.ob.name == 'mega_tri_mesh':
    if False:
    #if True:

        # Bend spring data ()------------------
        cloth.bend_data = True            # gets overwritten as none if there no bend sets
        cloth.tridex = None               # index the whole cloth mesh as triangles
        cloth.bend_ed_idx = None          # index the edges that are bend axes
        cloth.bend_edges = None           # index the verts in the bend axes
        cloth.bend_tri_tips = None        # tips on tris opposite the bend axes
        cloth.bend_tris = None            # tris that are evaluated as bend tris
        # Dynamic bend data:
        cloth.bend_weights = None         # barycentric weights for placing tri tips
        cloth.bend_surface_offset = None  # distance along non-unit cross product from tris
        #t = T()
        # overwrite all the None values
        get_bend_sets(cloth)
        #T(t, "time it took to get the bend springs")

        # cpoe method for plot
        cloth.axis_div = None
        cloth.cross_div = None
        cloth.tri_div = None
        cpoe_bend_plot_values(cloth)

    # Allocate arrays

    cloth.select_start = np.copy(cloth.co)
    cloth.pin_arr = np.copy(cloth.co)
    cloth.vel = np.zeros_like(cloth.co)
    cloth.vel_zero = np.zeros_like(cloth.co)
    cloth.feedback = np.zeros_like(cloth.co)
    cloth.stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.bend_stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32) # for calculating the weights of the mean

    cloth.target_co = get_co_mode(cloth.target) # (will be None if target is None)
    cloth.velocity = Nx3(ob)

    if cloth.do_stretch:
        cloth.measure_length = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32)
        cloth.measure_dot = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32)

        # target co
        same = False
        if cloth.target is not None: # if we are doing a two way update we will need to put run the updater here
            same = compare_geometry(ob, cloth.target, cloth.obm, cloth.target_obm)
            if same:
                cloth.source, cloth.dots = stretch_springs_basic(cloth, cloth.target)

        if not same: # there is no matching target or target is None

            cloth.v, cloth.source, cloth.dots = stretch_springs_basic(cloth)

        cloth.vdl = stretch_springs_basic(cloth, cloth.target)

    return cloth

# ^                                                          ^ #
# ^                   END cloth instance                     ^ #
# ============================================================ #



# ============================================================ #
#                     update the cloth                         #
#                                                              #

# update the cloth ---------------
def update_groups(cloth, obm=None, geometry=False):
    """Create update data in the cloth instance
    related to the vertex groups.
    geometry is run when there are changes in the geomtry"""
    #ob = cloth.ob
    #current_index = ob.vertex_groups.active_index

    # vertex groups
    current = np.copy(cloth.pin)

    cloth.obm = get_bmesh(cloth.ob)
    manage_vertex_groups(cloth)

    if geometry:
        old = current.shape[0]
        new = cloth.pin.shape[0]
        dif = new - old
        if dif > 0:
            fix = np.zeros_like(cloth.pin)
            fix[:old] += current
            current = fix
            cloth.pin_arr = np.append(cloth.pin_arr, cloth.co[old:], axis=0)
            zeros = np.zeros_like(cloth.co[old:])
            cloth.velocity = np.append(cloth.velocity, zeros, axis=0)
            cloth.vel_zero = np.copy(cloth.co)
            cloth.feedback = np.copy(cloth.co)
        else:
            cloth.pin_arr = np.copy(cloth.co)
            cloth.vel_zero = np.copy(cloth.co)
            cloth.feedback = np.copy(cloth.co)
            cloth.velocity = np.zeros_like(cloth.co)
            return
    # !!! Need to store cloth.pin_arr in the save file !!!
    changed = (cloth.pin - current) != 0 # Have to manage the location of the pin verts with weights less than one
    if hasattr(cloth, 'co'):
        cloth.pin_arr[changed.ravel()] = cloth.co[changed.ravel()]

    # update surface weight
    #ob.vertex_groups.active_index = current_index






# update the cloth ---------------
def measure_edges(co, idx, cloth):
    """Takes a set of coords and an edge idx and measures segments"""
    l = idx[:,0]
    r = idx[:,1]
    v = co[r] - co[l]
    #d = np.einsum("ij ,ij->i", v, v, out=cloth.measure_length)
    np.einsum("ij ,ij->i", v, v, out=cloth.measure_dot)
    #return v, d, np.sqrt(d)
    np.sqrt(cloth.measure_dot, out=cloth.measure_length)
    return v


def stretch_springs_basic(cloth, target=None): # !!! need to finish this
    """Measure the springs"""
    if target is not None:
        dg = cloth.dg
        #proxy = col.ob.to_mesh(bpy.context.evaluated_depsgraph_get(), True, calc_undeformed=False)
        #proxy = col.ob.to_mesh() # shouldn't need to use mesh proxy because I'm using bmesh
        if cloth.proxy is None:
            cloth.proxy = target.evaluated_get(dg)
        co = get_co_mode(cloth.proxy) # needs to be depsgraph eval
        # need to get co with modifiers that don't affect the vertex count
        # so I could create a list of mods to turn off then use that fancy
        # thing I created for turning off modifiers in the list.
        return measure_edges(co, cloth.basic_set, cloth)

    if cloth.ob.MC_props.p1_bend:
        co = cloth.sco
    else:
        co = get_co_shape(cloth.ob, 'MC_source')
    # can't figure out how to update new verts to source shape key when
    #   in edit mode. Here we pull from source shape and add verts from
    #   current bmesh where there are new verts. Need to think about
    #   how to fix this so that it blendes correctly with the source
    #   shape or target... Confusing....  Will also need to update
    #   the source shape key with this data once we switch out
    #   of edit mode. If someone is working in edit mode and saves
    #   their file without switching out of edit mode I can't fix
    #   that short of writing these coords to a file.
    if cloth.ob.data.is_editmode:
        co = np.append(co, cloth.co[co.shape[0]:], axis=0)
    v = measure_edges(co, cloth.basic_set, cloth)
    return v, np.copy(cloth.measure_dot), np.copy(cloth.measure_length)


def surface_forces(cloth):

    # surface follow data ------------------------------------------
    # cloth.surface_follow    (weights on the cloth object)
    # cloth.bind_idx = idx            (verts that are bound to the surface)
    tridex = cloth.surface_tridex          # (index of tris in surface object)
    bary = cloth.surface_bary_weights    # (barycentric weights)
    so = cloth.surface_object      # (object we are following)

    shape = cloth.surface_tridex.shape
    tri_co = np.array([so.data.vertices[i].co for i in tridex.ravel()], dtype=np.float32)
    tri_co.shape = (shape[0] * 3, 3)
    apply_in_place(cloth.surface_object, tri_co)

    tri_co.shape = (shape[0], 3, 3)
    plot = np.sum(tri_co * bary[:, :, nax], axis=1)

    # update the normals -----------------
    cloth.surface_norms = get_normals_from_tris(tri_co)
    norms = cloth.surface_norms * cloth.surface_norm_vals
    #print(cloth.surface_norm_vals[0], 'what is this norm val???????')
    plot += norms
    #plot += apply_rotation(cloth.ob, norms)

    world_co = apply_in_place(cloth.ob, cloth.co[cloth.bind_idx])
    dif = (plot - world_co) * cloth.surface_follow[cloth.bind_idx]
    cloth.co[cloth.bind_idx] += revert_rotation(cloth.ob, dif)



# ============================================================ #
#                  seam wrangler functions                     #
#                                                              #

# seam wrangler -------------
def weight_plot(data, cloth):
    """Use trianlges to plot the slices from bary weights"""

    tri_ob = data['tri_mesh_ob']
    tri_co = np.copy(cloth.co)

    t_shape = tri_co.shape
    tri_co.shape = (t_shape[0]//3, 3, 3)
    tiled = tri_co[data['tri_tiler']]

    # apply xy scale
    xy_s = data['xy_scale']

    if ((xy_s[0] != 1.0) | (xy_s[1] != 1.0)):
        tri_co_start = data['tri_co_start'] # shaped N,3,3
        tiled_start = tri_co_start[data['tri_tiler']]
        scaler = np.array([xy_s[0], xy_s[1], 1], dtype=np.float32)
        points = data['with_z'] * scaler
        new_weights = get_bary_weights(tiled_start, points)
        plot = np.sum(tiled * new_weights[:,:,None], axis=1)
        return plot

    plot = np.sum(tiled * data['weights'][:,:,None], axis=1)
    return plot


# seam wrangler -------------
def pure_linear(cloth, data):
    """linear relationships between
    tris and vps"""
    #print('pure_linear disabled')
    #return
    co = data['cloth_co']
    sp = data['springs']
    ls = sp[:,0]
    rs = sp[:,1]
    l = data['dists']
    stretch = data['seam_influence']  * 0.5 # only moving left side

    s_iters = data['linear_iters']
    for i in range(s_iters):

        # current vec, dot, length
        cv = co[rs] - cloth.co[ls]
        cd = np.einsum("ij ,ij->i", cv, cv)
        cl = np.sqrt(cd)

        move_l = (cl - l) * stretch

        # mean method -------------------
        data['stretch_array'][:] = 0.0
        rock_hard_abs = np.abs(move_l)
        np.add.at(data['stretch_array'], ls, rock_hard_abs)
        weights = rock_hard_abs / data['stretch_array'][ls]
        # mean method -------------------

        # apply forces ------------------
        move = cv * (move_l / cl)[:,None]
        move *= weights[:,None]
        np.add.at(cloth.co, ls, np.nan_to_num(move))
        #print(np.any(np.isnan(weights)), "nanny here")


# seam wrangler -------------
def update_vp_means(data):

    pass


# seam wrangler -------------
def move_tris(cloth, data):
    """Move tris with complete vps to
    location on garment"""

    tri_mesh = data['tri_mesh_ob']

    # move to means
    vecs = data['vp_means'] - data['tri_means']
    #vecs[~data['complete']] *= 0 # Don't move where points are missing

    tri_shape = False
    if tri_shape:
        shape = data['tris'].shape
        data['tris'].shape = (shape[0]//3 ,3 ,3)

    moved = data['tris'] + vecs[:, None]
    return moved


# seam wrangler -------------
def seam_position(cloth, data):
    """Use a mix of means and linear springs
    to position triangles near seams"""

    # setup initial positioning
    co = move_tris(cloth, data)
    #cloth.co.ravel()[:] = co.ravel()[:]
    #return

    if MC_data['count'] == 0:
        cloth.co.ravel()[:] = co.ravel()[:]
        b_log(['returned with: MC_data["count"] == 0'])
        return

    if MC_data['count'] in data['partial_set']:
        mean_idx = data['good_ravel']
        cloth.co.ravel()[mean_idx] = co.ravel()[mean_idx]
        b_log("returned with: [MC_data['count'] in data['partial_set']")
        return

    b_log('returned with: mean_idx = data["complete_ravel"]')
    mean_idx = data['complete_ravel']
    cloth.co.ravel()[mean_idx] = co.ravel()[mean_idx]


# seam wrangler -------------
def seam_updater(cloth, data):
    # overwrite co
    ob = data['ob']

    cloth_key = 'sw_view'
    if ob.data.shape_keys is not None:
        keys = ob.data.shape_keys.key_blocks
        names = [i.name for i in keys if ((i.name != 'sw_view') & (i.value == 1))]
        if len(names) > 0:
            cloth_key = names[-1]
            data['cloth_key'] = cloth_key
            if 'sw_view' in keys:
                keys['sw_view'].value = 0
            use_key = True
        if len(names) > 1:
            msg = ["!!! Warning. More than one shape key with value of 1" for i in range(3)]
            msg += names
            b_log(msg)

    get_proxy_co(ob, data['cloth_co'])

    #print('=== seam updater ===')
    # manage_seams

    #data['velocity'] =         0.77 # movement between run_frams
    #data['run_frames'] =         20 # calculate every force this many times
    #data['iterations'] =          2 # solves the trianlges this many times every run frame (higher means stiffer but slower to solve)
    #data['tri_force'] =           1 # forces the trinalgs towards their original shape (1 or less to avoid exploding)
    #data['seam_influence'] =    1.0 # seams pulling on triangles
    #data['triangle_influenc'] = 0.9 # triangles forcing seams to targets
    #data['xy_scale'] =    [1.0, 1.0]# multiply the xy values by this amount

    if cloth.current_iter != cloth.last_iter - 1:
        return True

    plot = weight_plot(data, cloth) # scales xy
    if data['vis_mesh']:
        link_mesh(plot.tolist(), [], [], 'plot test. Frame: ' + str(bpy.context.scene.frame_current))

    # apply slice force:
    #if False:
    b_log(['tri_influence when actually called', data['tri_influence']])
    vecs = (plot - data['cloth_co'][data['vps']]) * data['tri_influence']
    #print('tri influence on seams', data['tri_influence'])
    #print("MC_tools.py line 1543 testing influence")
    #vecs = (plot - data['cloth_co'][data['vps']]) * .01# data['tri_influence']

    data['cloth_co'][data['vps']] += np.nan_to_num(vecs)

    ob.data.shape_keys.key_blocks[cloth_key].data.foreach_set('co', data['cloth_co'].ravel())
    ob.data.update()

    return False

# ^               END seam wrangler functions                ^ #
# ============================================================ #

def mean_method(move_l, indexer, cv):
    stretch = cloth.ob.MC_props.self_collide_force
    #move_l = (cl - l) * stretch
    move_l *= stretch

    if False: # for the vertex group
        if not cloth.skip_stretch_wieght:
            move_l *= cloth.stretch_group_mult.ravel()

    # separate push springs
    if False:
        if push != 1:
            push_springs = move_l < 0
            move_l[push_springs] *= push

    # !!! here we could square move_l to accentuate bigger stretch
    # !!! see if it solves better.

    # mean method -------------------
    # cloth.stretch_array[:] = 0.0 # if it can be precalculated
    cloth.stretch_array = np.zeros(move_l.shape[0], dtype=np.float32)
    # rock_hard_abs = np.abs(move_l) # for target length. Not used when length is zero like in self collide.
    rock_hard_abs = move_l
    np.add.at(cloth.stretch_array, indexer, rock_hard_abs)
    weights = rock_hard_abs / cloth.stretch_array[indexer]
    # mean method -------------------

    # apply forces ------------------
    #if False:
    #move = cv * (move_l / cl)[:,None]

    #move *= weights[:,None]

    np.add.at(cloth.co, cloth.basic_v_fancy, np.nan_to_num(cv * weights[:, None]))


def bend_spring_force_linear(cloth):

    tris = cloth.co[cloth.bend_tris]
    surface_offset = cross_from_tris(tris) * cloth.bend_surface_offset
    plot = np.sum(tris * cloth.bend_weights, axis=1) + surface_offset
    # -----------------------------------------
    bend_stiff = cloth.ob.MC_props.bend * 0.2

    cv = (plot - cloth.co[cloth.bend_tri_tips]) * bend_stiff

    # swap the mags so that the larger triangles move a shorter distance
    d = np.einsum('ij,ij->i', cv, cv)
    l = np.sqrt(d)[:, None]
    m1 = np.nan_to_num(l[::2] / l[1::2])
    m2 = np.nan_to_num(l[1::2] / l[::2])
    cv[1::2] *= m1
    cv[::2] *= m2

    np.add.at(cloth.co, cloth.bend_tri_tips, cv)

    # now push the hinge the opposite direction
    sh = cv.shape
    cv.shape = (sh[0]//2,2,3)
    mix = np.mean(cv, axis=1)
    np.subtract.at(cloth.co, cloth.bend_edges, mix[:, None])


def bend_spring_force_mixed(cloth):

    tris = cloth.co[cloth.bend_tris]

    cpoe = True
    if cpoe:
        plot = cpoe_bend_plot(cloth)

    else:
        unit = True # for testing unit normalized surface offset
        if unit:
            cross = cross_from_tris(tris)
            U_cross = cross / np.sqrt(np.einsum('ij,ij->i', cross, cross))[:, None]
            surface_offset = U_cross * cloth.bend_U_d[:, None]

        else:
            surface_offset = cross_from_tris(tris) * cloth.bend_surface_offset
        plot = np.sum(tris * cloth.bend_weights, axis=1) + surface_offset
    # -----------------------------------------
    bend_stiff = cloth.ob.MC_props.bend * 0.2

    cv = plot - cloth.co[cloth.bend_tri_tips]
    d = np.einsum('ij,ij->i', cv, cv)
    l = np.sqrt(d)

    m1 = l[::2] / l[1::2]
    m2 = l[1::2] / l[::2]
    cv[1::2] *= m1[:, None]
    cv[::2] *= m2[:, None]

    # mean method ----------------------
    cloth.bend_tri_tip_array[:] = 0
    np.add.at(cloth.bend_tri_tip_array, cloth.bend_tri_tips, l)
    weights = l / cloth.bend_tri_tip_array[cloth.bend_tri_tips]

    cv *= weights[:, None]
    cv *= bend_stiff
    np.add.at(cloth.co, cloth.bend_tri_tips, np.nan_to_num(cv))
    sh = cv.shape
    cv.shape = (sh[0]//2,2,3)
    mix = np.mean(cv, axis=1)

    np.subtract.at(cloth.co, cloth.bend_edges, np.nan_to_num(mix)[:, None])


def bend_spring_force_U_cross(cloth):

    tris = cloth.co[cloth.bend_tris]
    #surface_offset = cross_from_tris(tris) * cloth.bend_surface_offset
    cross = cross_from_tris(tris)
    U_cross = cross / np.sqrt(np.einsum('ij,ij->i', cross, cross))[:, None]
    surface_offset = U_cross * cloth.bend_U_d[:, None]
    plot = np.sum(tris * cloth.bend_weights, axis=1) + surface_offset
    # -----------------------------------------
    bend_stiff = cloth.ob.MC_props.bend

    cv = plot - cloth.co[cloth.bend_tri_tips]
    d = np.einsum('ij,ij->i', cv, cv)
    l = np.sqrt(d) * bend_stiff
    move_l = l * bend_stiff

    # mean method ----------------------
    cloth.bend_tri_tip_array[:] = 0
    np.add.at(cloth.bend_tri_tip_array, cloth.bend_tri_tips, l)
    weights = np.nan_to_num(move_l / cloth.bend_tri_tip_array[cloth.bend_tri_tips])
    cv *= weights[:, None]
    cv *= bend_stiff

    np.add.at(cloth.co, cloth.bend_tri_tips, cv)
    np.subtract.at(cloth.co, cloth.bend_edges.ravel(), cv)


def spring_basic_no_sw(cloth):

    # get properties
    grav = cloth.ob.MC_props.gravity * 0.001
    vel = cloth.ob.MC_props.velocity
    feedback_val = cloth.ob.MC_props.feedback
    stretch = cloth.ob.MC_props.stretch * 0.5
    push = cloth.ob.MC_props.push

    # !!! Optimize here ============================================
    # measure source

    v, d, l = cloth.vdl
    #dynamic = False
    #if dynamic:
        # !!! don't need to run this all the time. Can get a speed improvement here
        #   by caching these values and running them when other updates run
        #v, d, l = stretch_springs_basic(cloth, cloth.target) # from target or source key

    cloth.select_start[:] = cloth.co

    cloth.co += cloth.velocity
    cloth.vel_zero[:] = cloth.co
    cloth.feedback[:] = cloth.co
    
    if cloth.do_bend:
        if cloth.ob.MC_props.bend > 0:
            for i in range(cloth.ob.MC_props.bend_iters):
                abstract_bend_(cloth)
                if cloth.ob.data.is_editmode:
                    if bpy.context.scene.MC_props.pause_selected:
                        cloth.co[cloth.selected] = cloth.select_start[cloth.selected]
                        cloth.pin_arr[cloth.selected] = cloth.select_start[cloth.selected]
                pin_vecs = (cloth.pin_arr - cloth.co)
                cloth.co += (pin_vecs * cloth.pin)


    if cloth.ob.MC_props.stretch > 0:
        for i in range(cloth.ob.MC_props.stretch_iters):

            # (current vec, dot, length)
            cv = measure_edges(cloth.co, cloth.basic_set, cloth) # from current cloth state
            cd = cloth.measure_dot
            cl = cloth.measure_length

            move_l = (cl - l) * stretch

            if not cloth.skip_stretch_wieght:
                move_l *= cloth.stretch_group_mult.ravel()

            # separate push springs
            if push != 1:
                push_springs = move_l < 0
                move_l[push_springs] *= push

            # !!! here we could square move_l to accentuate bigger stretch
            # !!! see if it solves better.

            # mean method -------------------
            cloth.stretch_array[:] = 0.0

            rock_hard_abs = np.abs(move_l)
            np.add.at(cloth.stretch_array, cloth.basic_v_fancy, rock_hard_abs)
            weights = rock_hard_abs / cloth.stretch_array[cloth.basic_v_fancy]
            # mean method -------------------

            # apply forces ------------------
            #if False:
            move = cv * (move_l / cl)[:,None]

            move *= weights[:,None]

            np.add.at(cloth.co, cloth.basic_v_fancy, np.nan_to_num(move))

            # apply surface sew for each iteration:
            if cloth.surface:
                surface_forces(cloth)

            # add pin vecs ------------------

            if cloth.ob.data.is_editmode:
                if bpy.context.scene.MC_props.pause_selected:
                    cloth.co[cloth.selected] = cloth.select_start[cloth.selected]
                    cloth.pin_arr[cloth.selected] = cloth.select_start[cloth.selected]
            
            pin_vecs = (cloth.pin_arr - cloth.co)
            cloth.co += (pin_vecs * cloth.pin)


    spring_move = cloth.co - cloth.feedback



    v_move = cloth.co - cloth.vel_zero

    cloth.velocity += v_move
    cloth.velocity += spring_move * feedback_val
    cloth.velocity *= vel

    cloth.velocity[:,2] += grav
    #cloth.velocity[:,2] += (grav * (-cloth.pin + 1).ravel())
    
    #t = time.time()
    if cloth.ob.MC_props.self_collide:
        sc = self_collide.detect_collisions(cloth)
        if cloth.ob.data.is_editmode:
            # add pin vecs ------------------
            pin_vecs = (cloth.pin_arr - cloth.co)
            cloth.co += (pin_vecs * cloth.pin)

            #if cloth.ob.MC_props.pause_selected:
            cloth.co[cloth.selected] = cloth.select_start[cloth.selected]
            cloth.pin_arr[cloth.selected] = cloth.select_start[cloth.selected]
    #print(time.time()-t, "total self collide time")
    

def spring_basic(cloth):
    #t = time.time()

    seam_wrangler = cloth.ob.name == 'mega_tri_mesh'
    if False:
        if seam_wrangler:
            data = bpy.context.scene.seam_wrangler_data
            type = data['run_type']
            if type == 1:
                if cloth.current_iter == 0:
                    b_log(['running seam position. cloth.current_iter:', cloth.current_iter])
                    seam_position(cloth, data)
                if cloth.current_iter < 3:
                    pure_linear(cloth, data)

        #MC_data['count'] += 1

    # get properties
    grav = cloth.ob.MC_props.gravity * 0.001
    vel = cloth.ob.MC_props.velocity
    feedback_val = cloth.ob.MC_props.feedback
    stretch = cloth.ob.MC_props.stretch * 0.5
    push = cloth.ob.MC_props.push

    # !!! Optimize here ============================================
    # measure source

    v, d, l = cloth.vdl
    #dynamic = False
    #if dynamic:
        # !!! don't need to run this all the time. Can get a speed improvement here
        #   by caching these values and running them when other updates run
        #v, d, l = stretch_springs_basic(cloth, cloth.target) # from target or source key

    cloth.select_start[:] = cloth.co

    cloth.co += cloth.velocity
    cloth.vel_zero[:] = cloth.co
    cloth.feedback[:] = cloth.co

    #if not seam_wrangler:
    if True:
        if cloth.do_bend:
            if cloth.ob.MC_props.bend > 0:
                for i in range(cloth.ob.MC_props.bend_iters):
                    abstract_bend_(cloth)
                    if cloth.ob.data.is_editmode:
                        if bpy.context.scene.MC_props.pause_selected:
                            cloth.co[cloth.selected] = cloth.select_start[cloth.selected]
                            cloth.pin_arr[cloth.selected] = cloth.select_start[cloth.selected]

    if cloth.ob.MC_props.stretch > 0:
        for i in range(cloth.ob.MC_props.stretch_iters):

            if seam_wrangler:
                data = bpy.context.scene.seam_wrangler_data
                type = data['run_type']
                if type == 1:
                    pure_linear(cloth, data)

            # (current vec, dot, length)
            cv = measure_edges(cloth.co, cloth.basic_set, cloth) # from current cloth state
            cd = cloth.measure_dot
            cl = cloth.measure_length

            move_l = (cl - l) * stretch

            if not cloth.skip_stretch_wieght:
                move_l *= cloth.stretch_group_mult.ravel()

            # separate push springs
            if push != 1:
                push_springs = move_l < 0
                move_l[push_springs] *= push

            # !!! here we could square move_l to accentuate bigger stretch
            # !!! see if it solves better.

            # mean method -------------------
            cloth.stretch_array[:] = 0.0

            rock_hard_abs = np.abs(move_l)
            np.add.at(cloth.stretch_array, cloth.basic_v_fancy, rock_hard_abs)
            weights = rock_hard_abs / cloth.stretch_array[cloth.basic_v_fancy]
            # mean method -------------------

            # apply forces ------------------
            #if False:
            move = cv * (move_l / cl)[:,None]

            move *= weights[:,None]

            np.add.at(cloth.co, cloth.basic_v_fancy, np.nan_to_num(move))

            # test ====================== bend springs
            # test ====================== bend springs
            if False:
                if seam_wrangler:
                    for i in range(cloth.ob.MC_props.bend_iters):
                        bend_spring_force_mixed(cloth)
            # test ====================== bend springs
            # test ====================== bend springs

            # apply surface sew for each iteration:
            if cloth.surface:
                surface_forces(cloth)

            # add pin vecs ------------------
            pin_vecs = (cloth.pin_arr - cloth.co)
            cloth.co += (pin_vecs * cloth.pin)
            if cloth.ob.data.is_editmode:
                #if cloth.ob.MC_props.pause_selected:
                cloth.co[cloth.selected] = cloth.select_start[cloth.selected]
                cloth.pin_arr[cloth.selected] = cloth.select_start[cloth.selected]

    spring_move = cloth.co - cloth.feedback
    v_move = cloth.co - cloth.vel_zero

    cloth.velocity += v_move
    cloth.velocity += spring_move * feedback_val
    cloth.velocity *= vel

    cloth.velocity[:,2] += grav
    #cloth.velocity[:,2] += (grav * (-cloth.pin + 1).ravel())

    seam_wrangler = cloth.ob.name == 'mega_tri_mesh'
    if seam_wrangler:
        #data = bpy.context.scene.seam_wrangler_data
        fix_twist(cloth)
        cloth.ob.MC_props.bend = 1.5
        for i in range(10):
            spring_basic_no_sw(cloth)
        cloth.ob.MC_props.bend = data['straightness']

        type = data['run_type']
        if type == 2:

            su = seam_updater(cloth, data)
            if cloth.current_iter < 3:
                if su:
                    pure_linear(cloth, data)
            #seam_position(cloth, data) # cant do this without caclulating vps
    #print(time.time() - t)


# update the cloth ---------------
def cloth_physics(ob, cloth, colliders):

    if ob.MC_props.cache_only:
        if ob.MC_props.cache:
            cloth.co = get_proxy_co(ob)
            cache(cloth)
            return

    if ob.MC_props.animated:
        ob.MC_props['current_cache_frame'] = bpy.context.scene.frame_current

    # If there is a proxy object will need to check that they match and issue
    #   warnings if there is a mismatch. Might want the option to regen the proxy object
    #   or adapt the cloth object to match so I can sync a patter change
    #   given you can change both now I have to also include logic to
    #   decide who is boss if both are changed in different ways.
    if cloth.target is not None:

        # if target is deleted while still referenced by pointer property
        if len(cloth.target.users_scene) == 0:
            ob.MC_props.target = None
            cloth.target = None # (gets overwritten by cb_target)
            cloth.target_undo = False
            cloth.target_geometry = None # (gets overwritten by cb_target)
            cloth.target_mode = 1
            return

        if cloth.target.data.is_editmode:
            if cloth.target_mode == 1 or cloth.target_undo:
                cloth.target_obm = get_bmesh(cloth.target)
                cloth.target_obm.verts.ensure_lookup_table()
                cloth.target_undo = False
            cloth.target_mode = None

        # target in object mode:
        else: # using else so it won't also run in edit mode
            pass

        dynamic_source = True # can map this to a prop or turn it on and off automatically if use is in a mode that makes it relevant.
        # If there is a target dynamic should prolly be on or if switching from active shape MC_source when in edit mode
        if dynamic_source:
            if not cloth.data.is_editmode: # can use bmesh prolly if not OBJECT mode.
                dg = cloth.dg
                if cloth.proxy is None:
                    cloth.proxy = cloth.target.evaluated_get(dg)
                #proxy = col.ob.to_mesh(bpy.context.evaluated_depsgraph_get(), True, calc_undeformed=False)
                #proxy = col.ob.to_mesh() # shouldn't need to use mesh proxy because I'm using bmesh

                co_overwrite(cloth.proxy, cloth.target_co)

    if cloth.shape_update:

        index = ob.data.shape_keys.key_blocks.find('MC_current')
        if cloth.ob.active_shape_key_index == index:
            refresh(cloth)
            cloth.shape_update = False

        if not cloth.ob.data.is_editmode:
            if cloth.shape_update:
                refresh(cloth)
                cloth.shape_update = False

    if ob.data.is_editmode:
        # prop to go into user preferences. (make it so it won't run in edit mode)
        if not bpy.context.scene.MC_props.run_editmode:
            return

        # bmesh gets removed when someone clicks on MC_current shape"
        try:
            cloth.obm.verts
        except:
            cloth.obm = get_bmesh(ob)

        if cloth.update_lookup:
            cloth.obm.verts.ensure_lookup_table()
            cloth.update_lookup = False

        # If we switched to edit mode or started in edit mode:
        if cloth.mode == 1 or cloth.undo:
            cloth.obm = get_bmesh(ob)
            cloth.obm.verts.ensure_lookup_table()
            cloth.undo = False
        cloth.mode = None
        # -----------------------------------

        # detect changes in geometry and update
        if cloth.obm is None:
            cloth.obm = get_bmesh(ob)

        if not cloth.ob.MC_props.cache_only:
            same, faces = detect_changes(cloth.geometry, cloth.obm)
            if faces: # zero faces in mesh do nothing
                return
            if not same:
                # for pinning
                print("DANGER!!!!!!!!!!!!!!!!!!")

                refresh(cloth)
                return

                np.copyto(cloth.pin_arr, cloth.co)
                #cloth.springs, cloth.v_fancy, cloth.e_fancy, cloth.flip = get_springs(cloth, cloth.obm)
                get_springs_2(cloth)
                get_bend_sets(cloth)
                cpoe_bend_plot_values(cloth)
                cloth.geometry = get_mesh_counts(ob, cloth.obm)
                # cloth.sew_springs = get_sew_springs() # build
                cloth.obm.verts.ensure_lookup_table()
                cloth.co = get_co_edit(ob)
                cloth.select_start = np.copy(cloth.co)
                cloth.stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32)
                cloth.selected = np.array([v.select for v in cloth.obm.verts])

                update_groups(cloth, cloth.obm, True)
                cloth.ob.update_from_editmode()
                cloth.obm.verts.ensure_lookup_table()
                cloth.vdl = stretch_springs_basic(cloth, cloth.target)
            # updating the mesh coords -----------------@@
            # detects user changes to the mesh like grabbing verts
            #t = T()

            #T(t, "list comp")

        # if we switch to a different shape key in edit mode:

        if not cloth.ob.MC_props.cache_only:
            index = ob.data.shape_keys.key_blocks.find('MC_current')
            if ob.active_shape_key_index != index:
                cloth.shape_update = True
                return

        cloth.ob.update_from_editmode()

        #cloth.co = np.array([v.co for v in cloth.obm.verts])

        if not cloth.ob.MC_props.cache_only:
            cloth.ob.data.shape_keys.key_blocks['MC_current'].data.foreach_get('co', cloth.co.ravel())
            cloth.selected[:] = False
            if bpy.context.scene.MC_props.pause_selected:
                cloth.ob.data.vertices.foreach_get('select', cloth.selected)

        if False: # detects all modal operators. Not very useful
            M = False
            for w in bpy.context.window_manager.windows:
                screen = w.screen
                for a in screen.areas:
                    if a.type == "VIEW_3D":
                        M = MODAL(w)# = bpy.data.texts['detect_modal.py'].as_module()
                        #print(MODAL.m)
            #print(M , '=========')
            if M:
                spring_basic(cloth)
                for i in range(cloth.co.shape[0]):
                    cloth.obm.verts[i].co = cloth.co[i]

        """ =============== FORCES EDIT MODE ================ """
        # FORCES FORCES FORCES FORCES FORCES
        if cloth.ob.MC_props.play_cache:
            play_cache(cloth)
            return

        for i in range(cloth.ob.MC_props.sub_frames):
            spring_basic_no_sw(cloth)

        if cloth.ob.MC_props.detect_collisions:
            collide(cloth, colliders)

        if False:
            if cloth.pbm:
                cloth.pb.test(cloth)

        # FORCES FORCES FORCES FORCES FORCES
        """ =============== FORCES EDIT MODE ================ """

        # set coords to current edit mode bmesh
        for i, j in enumerate(cloth.co):
            cloth.obm.verts.ensure_lookup_table()
            cloth.obm.verts[i].co = j

        if cloth.ob.MC_props.cache:
            if cloth.ob.MC_props.internal_cache:    
                np_co_to_text(cloth.ob, cloth.co, rw='w')
            else:
                cache(cloth)

        return

    # switched out of edit mode
    if cloth.mode is None:
        cloth.mode = 1
        if not cloth.ob.MC_props.cache_only:
            update_groups(cloth, cloth.obm)

    # OBJECT MODE ====== :
    """ =============== FORCES OBJECT MODE ================ """
    # FORCES FORCES FORCES FORCES
    if cloth.ob.MC_props.play_cache:
        play_cache(cloth)
        return

    for i in range(cloth.ob.MC_props.sub_frames):
        spring_basic_no_sw(cloth)

    if cloth.ob.MC_props.detect_collisions:
        collide(cloth, colliders)

    # FORCES FORCES FORCES FORCES
    """ =============== FORCES OBJECT MODE ================ """

    # updating the mesh coords -----------------@@
    ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set("co", cloth.co.ravel())
    cloth.ob.data.update()

    if cloth.ob.MC_props.cache:
        if cloth.ob.MC_props.internal_cache:    
            np_co_to_text(cloth.ob, cloth.co, rw='w')
        else:
            cache(cloth)

# update the cloth ---------------
def update_cloth(type=0):

    # run from either the frame handler or the timer
    if type == 0:
        cloths = [i[1] for i in MC_data['cloths'].items() if i[1].ob.MC_props.continuous]
        if len(cloths) == 0:
            bpy.app.timers.unregister(cloth_main)

    if type == 1:
        cloths = [i[1] for i in MC_data['cloths'].items() if i[1].ob.MC_props.animated]
        if len(cloths) == 0:
            install_handler(continuous=True, clear=False, clear_anim=True)

    # check collision objects
    colliders = [i[1] for i in MC_data['colliders'].items() if i[1].ob.MC_props.collider]

    for cloth in cloths:
        cloth_physics(cloth.ob, cloth, colliders)

    #print([i.ob.name for i in cloths], "list of cloths")
    #print([i.ob.name for i in colliders], "list of colliders")
    return


# ^                                                          ^ #
# ^                 END update the cloth                     ^ #
# ============================================================ #


# ============================================================ #
#                      Manage handlers                         #
#                                                              #
# handler ------------------
@persistent
def undo_frustration(scene):
    # someone might edit a mesh then undo it.
    # in this case I need to recalc the springs and such.
    # using id props because object memory adress changes with undo

    # find all the cloth objects in the scene and put them into a list
    cloths = [i for i in bpy.data.objects if i.MC_props.cloth]
    # throw an id prop on there.
    for i in cloths:
        cloth = MC_data['cloths'][i['MC_cloth_id']]
        cloth.ob = i
        # update for meshes in edit mode
        cloth.undo = True
        cloth.target_undo = True
        try:
            cloth.obm.verts
        except:
            cloth.obm = get_bmesh(cloth.ob)


        if not detect_changes(cloth.geometry, cloth.obm)[0]:
            cloth.springs, cloth.v_fancy, cloth.e_fancy, cloth.flip = get_springs(cloth)

    # find all the collider objects in the scene and put them into a list
    colliders = [i for i in bpy.data.objects if i.MC_props.collider]
    for i in colliders:
        col = MC_data['colliders'][i['MC_collider_id']]
        col.ob = i
        # update for meshes in edit mode
        col.undo = True

    fun = ["your shoe laces", "something you will wonder about but never notice", "something bad because it loves you", "two of the three things you accomplished at work today", "knots in the threads that hold the fabric of the universe", "a poor financial decision you made", "changes to your online dating profile", "your math homework", "everything you've ever accomplished in life", "something you'll discover one year from today", "the surgery on your cat", "your taxes", "all the mistakes you made as a child", "the mess you made in the bathroom", "the updates to your playstation 3", "nothing! Modeling Cloth makes no mistakes!", "your last three thoughts and you'll never know what they were", "the damage done to the economy by overreaction to covid"]
    print("Modeling Cloth undid " + fun[MC_data["iterator"]])
    MC_data['iterator'] += 1
    if MC_data['iterator'] == len(fun):
        MC_data['iterator'] = 0


# handler ------------------
@persistent
def cloth_main(scene=None):
    """Runs the realtime updates"""

    total_time = T()

    kill_me = []
    # check for deleted cloth objects
    for id, val in MC_data['cloths'].items():
        try:
            val.ob.data
        except:
            # remove from dict
            kill_me.append(id)

    for i in kill_me:
        del(MC_data['cloths'][i])
        print('killed wandering cloths')

    kill_me = []
    # check for deleted collider objects
    for id, val in MC_data['colliders'].items():
        try:
            val.ob.data
        except:
            # remove from dict
            kill_me.append(id)

    for i in kill_me:
        del(MC_data['colliders'][i])
        print('killed wandering colliders')

    # run the update -------------
    type = 1 # frame handler or timer continuous
    if scene is None:
        type=0

    delay = bpy.context.scene.MC_props.delay

    update_cloth(type) # type 0 continuous, type 1 animated

    # auto-kill
    auto_kill = True
    auto_kill = False
    if auto_kill:
        if MC_data['count'] == 20:
            print()
            print('--------------')
            print('died')
            return


    #print('------------------------------------')
    #print()
    #T(total_time, "Total time in handler")
    return delay


# handler ------------------
def install_handler(continuous=True, clear=False, clear_anim=False):
    """Run this when hitting continuous update or animated"""
    # clean dead versions of the animated handler
    handler_names = np.array([i.__name__ for i in bpy.app.handlers.frame_change_post])
    booly = [i == 'cloth_main' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.frame_change_post[i])
        print("deleted handler ", i)
    if clear_anim:
        print('ran clear anim handler')
        return

    # clean dead versions of the timer
    if bpy.app.timers.is_registered(cloth_main):
        bpy.app.timers.unregister(cloth_main)

    # for removing all handlers and timers
    if clear:
        print('ran clear handler')
        return

    MC_data['count'] = 0

    if np.any([i.MC_props.continuous for i in bpy.data.objects]):
        # continuous handler
        bpy.app.timers.register(cloth_main, persistent=True)
        #bpy.app.timers.register(funky.partial(cloth_main, delay, kill, T), first_interval=delay_start, persistent=True)
        return

    # animated handler
    if np.any([i.MC_props.animated for i in bpy.data.objects]):
        bpy.app.handlers.frame_change_post.append(cloth_main)


# ^                                                          ^ #
# ^                      END handler                         ^ #
# ============================================================ #


# ============================================================ #
#                    callback functions                        #
#                                                              #

def fix_twist(cloth=None):
    b_log(['fortinbrass 8:13:2020'])
    ob = bpy.data.objects['mega_tri_mesh']
    keys = ob.data.shape_keys.key_blocks
    k = keys['MC_current']
    #co = np.empty((len(ob.data.vertices), 3), dtype=np.float32)
    #k.data.foreach_get('co', co.ravel())
    co = cloth.co
    tridex = np.arange(co.shape[0])
    tridex.shape = (co.shape[0] // 3, 3)

    start = True

    last_nor = np.cross(co[1]- co[0], co[2] - co[0])
    ed = np.empty((len(ob.data.edges), 2), dtype=np.int32)
    ob.data.edges.foreach_get('vertices', ed.ravel())

    for t in tridex:
        bounds = np.sum(t[0] == ed) == 3
        if bounds & start:
            nor = np.cross(co[t[1]] - co[t[0]], co[t[2]] - co[t[0]])
            last_nor = nor
            start = False
            continue

        if bounds:
            start = True

        nor = np.cross(co[t[1]] - co[t[0]], co[t[2]] - co[t[0]])
        compare = nor @ last_nor

        if compare < 0:
            edge = co[t[1]] - co[t[0]]
            uedge = edge / (np.sqrt(edge @ edge))
            pvec = co[t[2]] - co[t[0]]
            spit = uedge * (pvec @ uedge)
            span = (spit - pvec) * 2
            co[t[2]] += span

            # after we overwrite:
            nor = np.cross(co[t[1]] - co[t[0]], co[t[2]] - co[t[0]])
        last_nor = nor

    #k.data.foreach_set('co', co.ravel())
    #ob.data.update()

# seam wrangler callback ----------
def cb_seam_wrangler(self, context):

    # in simulation 2889 headwrap:
    # Migt be better to run vp means each frame
    # To do this I need to update the coords of vp means.
    # They are currently built on setup when iterating through all the slie target data.
    # There is something causing the seams to be pulled to one side. Coupld be nans are zeroing forces on one sides
    #   could be an index error. Could be there is a problem with the linear springs in "pure_linear"
    b_log(['cb_seam_wrangler called', 'checking self.seam_wrangler', self.seam_wrangler])

    if not self.seam_wrangler:
        return

    data = bpy.context.scene.seam_wrangler_data # this value is different the second time it runs

    MC_data['sw_count'] = data['sw_count']
    MC_data['count'] = data['sw_count']

    ob = data['tri_mesh_ob']
    ob.MC_props.velocity = data['velocity']

    cloth = get_cloth(ob)

    straightness = data['straightness']
    ob.MC_props.bend = data['straightness']

    iters = data['total_iterations']
    #print(data['total_iterations'], "data total iterations")
    #iters = 7

    cloth.last_iter = iters

    type = data['run_type']
    if type == 1:
        if False:
            for i in range(iters):
                np_co_to_text(ob, cloth.co, rw='w')
                cloth.current_iter = i
                if cloth.current_iter == 0:
                    b_log(['running seam position. cloth.current_iter:', cloth.current_iter])
                    seam_position(cloth, data)
                if cloth.current_iter < 3:
                    pure_linear(cloth, data)

        np_co_to_text(ob, cloth.co, rw='w')
        ob.MC_props.bend = 1.5
        fix_twist(cloth)
        for i in range(500):
            pure_linear(cloth, data)
            spring_basic_no_sw(cloth)
        ob.MC_props.bend = data['straightness']
        np_co_to_text(ob, cloth.co, rw='w')
        # try to get rid of the twist
        #ob.MC_props.bend = 1.5
        #for i in range(2000):
            #spring_basic_no_sw(cloth)
        #ob.MC_props.bend = data['straightness']


    for i in range(iters):
        cloth.current_iter = i
        spring_basic(cloth)
        np_co_to_text(cloth.ob, cloth.co, rw='w')

    #cache(cloth, keying=False)
    ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set("co", cloth.co.ravel())
    #ob.data.vertices.foreach_set('co', cloth.co.ravel())
    ob.data.update()

    try:
        from garments_blender.utils.rich_blender_utils import fix_all_shape_key_nans
        fix_all_shape_key_nans(ob)
    except:
        rbu = bpy.data.texts['rich_blender_utils.py'].as_module()
        rbu.fix_all_shape_key_nans(ob)


# calback functions ---------------
def oops(self, context):
    # placeholder for reporting errors or other messages
    return



def open_folder(path):
    import subprocess
    import sys

    if sys.platform == 'darwin':
        subprocess.check_call(['open', '--', path])
    elif sys.platform == 'linux2':
        subprocess.check_call(['gnome-open', '--', path])
    elif sys.platform == 'win32':
        subprocess.check_call(['explorer', path])


# calback functions ---------------
# object:
def cb_cache_only(self, context):

    ob = self.id_data

    if self.cache_only:
        self.cloth = True
        #self.cache = True
        self.animated = True
        self.cache_name = ob.name
        return

    self.cache = False
    self.cloth = False
    self.animated = False

def check_file_path(ob):
    """Check if the current filepath is legit"""
    self = ob.MC_props
    custom = pathlib.Path(self.cache_folder)
    mc_path = custom.joinpath('MC_cache_files')
    name = self.cache_name
    final_path = mc_path.joinpath(name)
    valid = final_path.exists()
    return valid, final_path


def cb_cache(self, context):
    """Manage files and paths for saving cache."""
    if self.cache:
        self['play_cache'] = False
        # Might want to overwrite while playing
        #   back with partial influence and running cloth sim

    ob = self.id_data

    cloth = get_cloth(ob)

    # set path to blender path by default
    path = pathlib.Path(bpy.data.filepath).parent #.parent removes .blend file
    if path == '':
        path = os.path.expanduser("~/Desktop")
        self['cache_folder'] = path

    if ob.MC_props.cache_desktop:
        path = os.path.expanduser("~/Desktop")
        self['cache_folder'] = path
    else:
        self['cache_folder'] = str(path) # so it switches back when turning of desktop

    mc_path = pathlib.Path(path).joinpath('MC_cache_files')

    # overwrite if user path is valid:
    custom = pathlib.Path(self.cache_folder)
    if not custom.exists():
        # Report Error
        msg = '"' + self.cache_folder + '"' + " is not a valid filepath. Switching to .blend location Desktop if blend file is not saved"
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        self['cache_folder'] = str(path)
    else:
        mc_path = custom.joinpath('MC_cache_files')

    # create dir if it doesn't exist
    if not mc_path.exists():
        mc_path.mkdir()

    name = str(ob['MC_cloth_id'])

    custom_name = self.cache_name
    if custom_name != "Mr. Purrkins The Deadly Cyborg Kitten":
        if custom_name != '':
            name = custom_name

    self['cache_name'] = name

    final_path = mc_path.joinpath(name)

    # create dir if it doesn't exist
    if not final_path.exists():
        final_path.mkdir()

    cloth.cache_dir = final_path
    return


def cb_cache_playback(self, context):
    ob = self.id_data
    cloth = get_cloth(ob)

    self['cache'] = False

    if self.cache_only:
        if self.play_cache:
            self['cache'] = False

            if ob.data.shape_keys == None:
                ob.shape_key_add(name='Basis')

            keys = ob.data.shape_keys.key_blocks
            index = ob.data.shape_keys.key_blocks.find('MC_current')
            active = ob.active_shape_key_index
            cloth.key_values = {'MC_active_idx_pre_cache': active}
            for k in keys:
                cloth.key_values[k.name] = k.value

            for k in keys:
                k.value = 0

            keys = ob.data.shape_keys.key_blocks
            if 'cache_key' not in keys:
                ob.shape_key_add(name='cache_key')

            keys['cache_key'].value = 1.0
            index = ob.data.shape_keys.key_blocks.find('cache_key')
            ob.active_shape_key_index = index
            return

        if ob.data.shape_keys is not None:
            keys = ob.data.shape_keys.key_blocks
            if len(keys) >= cloth.key_values['MC_active_idx_pre_cache']:
                ob.active_shape_key_index = cloth.key_values['MC_active_idx_pre_cache']

            if 'cache_key' in keys:
                keys['cache_key'].value = 0

            for k, v in cloth.key_values.items():
                if k in keys:
                    keys[k].value = v


def cb_current_cache_frame(self, context):
    """Keep track of the current saved cache frame."""
    ob = self.id_data

    cloth = get_cloth(ob)
    cloth.cache_dir = check_file_path(ob)[1]
    cloth.current_cache_frame = self.current_cache_frame
    play_cache(cloth, cb=True)


def cb_detect_collisions(self, context):
    ob = self.id_data
    print(ob.name, ' cloth object set to check for collisions')


def cb_self_collision(self, context):
    print('ran cb_self_collision')


def cb_collider(self, context):
    """Set up object as collider"""

    ob = self.id_data
    if ob.type != "MESH":
        self['collider'] = False

        # Report Error
        msg = "Must be a mesh. Collisions with non-mesh objects can create black holes potentially destroying the universe."
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        return

    # The object is the key to the cloth instance in MC_data
    if self.collider:
        collider = Collider()
        collider.refresh()

        d_keys = [i for i in MC_data['colliders'].keys()]
        id_number = 0

        if len(d_keys) > 0:
            id_number = max(d_keys) + 1
            print("created new collider id", id_number)

        MC_data['colliders'][id_number] = collider
        print('created collider instance')

        ob['MC_collider_id'] = id_number
        print(MC_data['colliders'])
        return

    # when setting collider to False
    if ob['MC_collider_id'] in MC_data['colliders']:
        del(MC_data['colliders'][ob['MC_collider_id']])
        del(ob['MC_collider_id'])

# calback functions ---------------
# object:
def cb_cloth(self, context):
    """Set up object as cloth"""

    ob = self.id_data

    if self.cache_only:
        cloth = create_instance(ob=ob)
        id_number = ob.name
        MC_data['cloths'][id_number] = cloth
        ob['MC_cloth_id'] = id_number
        return

    # set the recent object for keeping settings active when selecting empties
    recent = MC_data['recent_object']

    if ob.type != "MESH":
        if recent is not None:
            ob = recent
        else:
            self['cloth'] = False

            # Report Error
            msg = "Must be a mesh. Non-mesh objects make terrible shirts."
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return

    if len(ob.data.polygons) < 1:
        self['cloth'] = False

        # Report Error
        msg = "Must have at least one face. Faceless meshes are creepy."
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        return

    # The custom object id is the key to the cloth instance in MC_data
    if self.cloth:
        # creat shape keys and set current to active
        if not ob.MC_props.p1_bend:
            reset_shapes(ob)
            index = ob.data.shape_keys.key_blocks.find('MC_current')
            ob.active_shape_key_index = index

        cloth = create_instance(ob=ob)

        # recent_object allows cloth object in ui
        #   when selecting empties such as for pinning.
        MC_data['recent_object'] = bpy.context.object

        # use an id prop so we can find the object after undo
        if False: # using time.time() for id
            d_keys = [i for i in MC_data['cloths'].keys()]
            id_number = 0
            if len(d_keys) > 0:
                id_number = max(d_keys) + 1
                print("created new cloth id", id_number)
            id_number = ob.name # Could create problems

        id_number = time.time()

        MC_data['cloths'][id_number] = cloth
        ob['MC_cloth_id'] = id_number
        #cb_cache(self, context) # if a cache exists at the specified file location. So the usere doesn't have to toggle the property if the file is a valid cache.
        print('created instance')
        return

    # when setting cloth to False
    if ob['MC_cloth_id'] in MC_data['cloths']:
        del(MC_data['cloths'][ob['MC_cloth_id']])
        del(ob['MC_cloth_id'])
        # recent_object allows cloth object in ui
        #   when selecting empties such as for pinning
        MC_data['recent_object'] = None
        ob.MC_props['continuous'] = False
        ob.MC_props['animated'] = False


# calback functions ----------------
# object:
def cb_continuous(self, context):
    """Turn continuous update on or off"""
    install_handler(continuous=True)


# calback functions ----------------
# object:
def cb_dense(self, context):
    ob = self.id_data
    print('Ran cb_dense. Did nothing.')


# extra bend function for p1 --------
def manage_pin_group(cloth):

    ob = cloth.ob

    g = 'MC_bend_pin'
    if "SW_seam_pin" in ob.vertex_groups:
        g = "SW_seam_pin"

    pidx = ob.vertex_groups[g].index

    # suspend keyed cloth mod settings:
    action = ob.animation_data.action
    air = [i for i in action.fcurves if i.data_path.endswith('air_damping')][0]
    # ---------------------------------

    # for testing outside p1 ------------------------------------
    if hasattr(bpy.context.scene, 'bend_stiffness_weights'):
        weights = bpy.context.scene.bend_stiffness_weights
        frames = bpy.context.scene.bend_stiffness_frames
    else:
        weights = np.ones(len(ob.data.vertices), dtype=np.float32)
        frames = np.arange(250)
    # ------------------------------------------------------------

    f = bpy.context.scene.frame_current

    if f % 2 == 0:
        if f in frames:
            for i, w in enumerate(weights):
                val = 0.0
                if w > 0:
                    val = 1
                ob.vertex_groups[g].add([pidx, i], val, 'REPLACE')

            for mod in ob.modifiers:
                if mod.type == "CLOTH":
                    mod.settings.vertex_group_mass = g
                    mod.settings.pin_stiffness = 1.0
                    mod.settings.air_damping = 10
                    air.mute = True

            return

    # on frames where we don't run -----------------------
    for i in range(len(ob.data.vertices)):
        ob.vertex_groups[g].add([pidx, i], 0.0, 'REPLACE')

    for mod in ob.modifiers:
        if mod.type == "CLOTH":
            mod.settings.pin_stiffness = 0.0
            air.mute = False
    #use verts with a bend wieght more than zero
    #try not to interfere with seam manager.
    #every other frame need to either be zero or one
    #has to work on any cloth modifier as the mods change
    #needs to detect: mod.settings.vertex_group_mass = 'SW_seam_pin'
    #needs to run after seam wrangler and take into
    #    account changes to mesh made by seam wrangler
    #    so like it will stiffen after seam wrangler has done
    #    it's thing. I could maybe run even frames when sw is odd...

    #sw only pins along the seams. p1b only pins where stiffened
    #maybe they wont affect each other much??


# calback functions ----------------
# object:
def cb_p1_bend(self, context):
    """Used by p1 to stiffen fabric"""

    if self.p1_bend:
        t = time.time()
        ob = self.id_data
        # make sure to overwrite vert groups !!!

        ob.MC_props.stretch_iters = 1
        ob.MC_props.stretch = 0.7
        ob.MC_props.feedback = 0.0
        ob.MC_props.bend_iters = 1
        ob.MC_props.bend =0.85
        ob.MC_props.velocity = 0.777
        ob.MC_props.quad_bend = True
        ob.MC_props.dense = True

        #return

        obm = get_bmesh(ob)
        obm.faces.ensure_lookup_table()
        obm.verts.ensure_lookup_table()

        # setup the vertext groups. (has to come from p1)
        # for testing outside p1 ------------------------------------
        if hasattr(bpy.context.scene, 'bend_stiffness_weights'):
            weights = bpy.context.scene.bend_stiffness_weights
            verts = np.arange(len(ob.data.vertices))
        else:
            verts = np.array([v.index for v in ob.data.vertices if v.select])
            weights = np.ones(len(verts))

        get_weights(ob, 'MC_bend_stiff', obm=obm, default=0, verts=verts, weights=weights)
        get_weights(ob, 'MC_stretch', obm=obm, default=0, verts=verts, weights=weights)
        get_weights(ob, 'MC_bend_pin', obm=obm, default=0, verts=verts, weights=weights)
        # ---------------------------------

        if ob.data.is_editmode:
            bmesh.update_edit_mesh(ob.data)
        else:
            obm.to_mesh(ob.data)
        ob.data.update()

        ob.MC_props.cloth = True
        print(time.time() - t, "setup time for bend solver")


def cb_p1_bend_trigger(self, context):
    """To be triggered by a frame handler
    for stiffening fabric in p1"""
    t = time.time()
    ob = self.id_data
    cloth = get_cloth(ob)
    cloth.co = get_proxy_co(ob)

    # --------------------------
    manage_pin_group(cloth) # sets the pin group in the cloth modifier
    # --------------------------
    f = bpy.context.scene.frame_current

    if hasattr(bpy.context.scene, 'bend_stiffness_frames'): # for running outside of p1

        if f % 2 != 0:
            print(time.time() - t, "ran without solve")
            return

        if f not in bpy.context.scene.bend_stiffness_frames:
            return

    print("running all the way")

    keys = ob.data.shape_keys
    active = []
    cloth.active_key = None
    if keys is None:
        cloth.active_key = None
    else:
        active = [k for k in keys.key_blocks if ((k.value == 1) & (k.name != 'bend_temp'))]
    if len(active) > 0:
        cloth.active_key = active[-1]

    cloth.sco = get_co(ob)

    if keys is not None:
        if 'flat' in keys.key_blocks:
            cloth.sco = get_co_shape(ob, 'flat')

    cloth.p1_trigger = True

    iters = 1

    for i in range(iters):
        spring_basic_no_sw(cloth)

    if cloth.active_key is None:
        print('active key is None. Using temp key')
        if 'bend_temp' not in keys.key_blocks:
            k = ob.shape_key_add(name='bend_temp')
            k.value = 1
        cloth.active_key = keys.key_blocks['bend_temp']
    else:
        if 'bend_temp' in keys.key_blocks:
            keys.key_blocks['bend_temp'].value = 0

    cloth.active_key.data.foreach_set('co', cloth.co.ravel())
    ob.data.update()
    print(time.time() - t, "bend solver time")


# calback functions ----------------
# object:
def cb_quad_bend(self, context):
    ob = self.id_data
    print('Ran cb_quad_bend. Did nothing.')


# calback functions ----------------
# object:
def cb_animated(self, context):
    """Turn animated update on or off"""
    install_handler(continuous=False) # deletes handler when false.

    if not self.animated:
        return

    # updates groups when we toggle "Animated"
    ob = self.id_data
    cloth = get_cloth(ob)
    if self.cache_only:
        cloth.co = get_proxy_co(ob)
        return

    if ob.data.is_editmode:
        index = ob.data.shape_keys.key_blocks.find('MC_current')
        if ob.active_shape_key_index == index:
            cloth.co = np.array([v.co for v in cloth.obm.verts], dtype=np.float32)
            update_groups(cloth, cloth.obm)
            return
        ob.update_from_editmode()

    cloth.co = get_co_shape(ob, key='MC_current')
    update_groups(cloth, None)


# calback functions ----------------
# object:
def cb_target(self, context):
    """Use this object as the source target"""

    # if the target object is deleted while an object is using it:
    if bpy.context.object is None:
        return

    # setting the property normally
    ob = bpy.context.object

    cloth = MC_data["cloths"][ob['MC_cloth_id']]

    # kill target data
    if self.target is None:
        cloth.target = None
        return

    # kill target data
    cloth.proxy = bpy.context.evaluated_get(self.target)
    same = compare_geometry(ob, proxy, obm1=None, obm2=None, all=False)
    if not same:
        msg = "Vertex and Face counts must match. Sew edges don't have to match."
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        self.target = None
        cloth.target = None
        return

    # Ahh don't kill me I'm a valid target!
    cloth.target = self.target
    cloth.target_geometry = get_mesh_counts(cloth.target)
    cloth.target_co = get_co_mode(cloth.target)


# calback functions ----------------
# object:
def cb_reset(self, context):
    """RESET button"""
    ob = MC_data['recent_object']
    if ob is None:
        ob = bpy.context.object
    cloth = MC_data["cloths"][ob['MC_cloth_id']]
    #RESET(cloth)
    self['reset'] = False


# calback functions ----------------
# scene:
def cb_pause_all(self, context):
    print('paused all')


# calback functions ----------------
# scene:
def cb_play_all(self, context):
    print('play all')


# calback functions ----------------
# scene:
def cb_duplicator(self, context):
    # DEBUG !!!
    # kills or restarts the duplicator/loader for debugging purposes
    if not bpy.app.timers.is_registered(duplication_and_load):
        bpy.app.timers.register(duplication_and_load)
        return

    bpy.app.timers.unregister(duplication_and_load)
    print("unloaded dup/load timer")

# ^                                                          ^ #
# ^                 END callback functions                   ^ #
# ============================================================ #


# ============================================================ #
#                     create properties                        #
#                                                              #

# create properties ----------------
# object:
class McPropsObject(bpy.types.PropertyGroup):

    self_collide:\
    bpy.props.BoolProperty(name="Self Collision", description="Self collisions (Hopefully preventing self intersections. Fingers crossed.)", default=False, update=cb_self_collision)

    self_collide_margin:\
    bpy.props.FloatProperty(name="Self Collision Margin", description="Self collision margin", default=0.02, min=0, precision=3)

    self_collide_force:\
    bpy.props.FloatProperty(name="Self Collision Force", description="Self collision force", default=0.5, precision=3)

    sc_vel_damping:\
    bpy.props.FloatProperty(name="Self Collision Velocity Damping", description="Self self collisions reduces velocity", default=1, precision=3)

    sc_friction:\
    bpy.props.FloatProperty(name="Self Collision Friction", description="Self self collisions friction", default=0.0, precision=3)

    sc_box_max:\
    bpy.props.IntProperty(name="SC Box Max", description="Max number of sets in an octree box", default=150, min=10)

    collider:\
    bpy.props.BoolProperty(name="Collider", description="Cloth objects collide with this object", default=False, update=cb_collider)

    outer_margin:\
    bpy.props.FloatProperty(name="Outer Margin", description="Distance from surface of collisions on positive normal side", default=0.1, precision=3)

    inner_margin:\
    bpy.props.FloatProperty(name="Inner Margin", description="Points within this distance on the negative side of the normal will be pushed out", default=0.2, precision=3)

    detect_collisions:\
    bpy.props.BoolProperty(name="Detect Collisions", description="This cloth object checks for collisions", default=False, update=cb_detect_collisions)

    cloth:\
    bpy.props.BoolProperty(name="Cloth", description="Set this as a cloth object", default=False, update=cb_cloth)

    seam_wrangler:\
    bpy.props.BoolProperty(name="Seam Wrangler", description="Special structure for managing seams", default=False, update=cb_seam_wrangler)

    # handler props for each object
    continuous:\
    bpy.props.BoolProperty(name="Continuous", description="Update cloth continuously", default=False, update=cb_continuous)

    animated:\
    bpy.props.BoolProperty(name="Animated", description="Update cloth only when animation is running", default=False, update=cb_animated)

    target:\
    bpy.props.PointerProperty(type=bpy.types.Object, description="Use this object as the target for stretch and bend springs", update=cb_target)


    # Forces
    gravity:\
    bpy.props.FloatProperty(name="Gravity", description="Strength of the gravity", default=0.0, min=-1000, max=1000, soft_min= -10, soft_max=10, precision=3)

    velocity:\
    bpy.props.FloatProperty(name="Velocity", description="Maintains Velocity", default=0.777, min= -1000, max=1000, soft_min= 0.0, soft_max=1, precision=3)

    wind:\
    bpy.props.FloatProperty(name="Wind", description="This Really Blows", default=0.0, min= -1000, max=1000, soft_min= -10.0, soft_max=10.0, precision=3)

    # stiffness
    feedback:\
    bpy.props.FloatProperty(name="Feedback", description="Extrapolate for faster solve", default=1, min= -1000, max=1000, soft_min= 0.0, soft_max=1, precision=3)

    stretch_iters:\
    bpy.props.IntProperty(name="Iters", description="Number of iterations of cloth solver", default=2, min=0, max=1000)#, precision=1)

    sub_frames:\
    bpy.props.IntProperty(name="Sub Frames", description="Number of sub frames between display", default=1, min=0, max=1000)#, precision=1)

    stretch:\
    bpy.props.FloatProperty(name="Stretch", description="Strength of the stretch springs", default=1, min=0, max=10, soft_min= 0, soft_max=1, precision=3)

    push:\
    bpy.props.FloatProperty(name="Push", description="Strength of the push springs", default=1, min=0, max=1, soft_min= -2, soft_max=2, precision=3)

    bend_iters:\
    bpy.props.IntProperty(name="Bend Iters", description="Number of iterations of bend springs", default=2, min=0, max=1000)#, precision=1)

    bend:\
    bpy.props.FloatProperty(name="Bend", description="Strength of the bend springs", default=1, min=0, max=10, soft_min= 0, soft_max=1, precision=3)

    dense:\
    bpy.props.BoolProperty(name="Dense", description="Default vertex weights are set to 0. For dense meshes so we can reduce setup calculations", default=False, update=cb_dense)

    p1_bend:\
    bpy.props.BoolProperty(name="P1 Bend", description="For bend stiffness in p1", default=False, update=cb_p1_bend)

    p1_bend_trigger:\
    bpy.props.BoolProperty(name="P1 Bend Trigger", description="Set it to true to trigger the bend to run", default=False, update=cb_p1_bend_trigger)

    quad_bend:\
    bpy.props.BoolProperty(name="Quad Bend Springs", description="Calculate bend springs on a bmesh with joined triangles", default=False, update=cb_quad_bend)

    # Sewing
    sew_force:\
    bpy.props.FloatProperty(name="Sew Force", description="Shrink Sew Edges", default=0.1, min=0, max=1, soft_min= -100, soft_max=100, precision=3)

    surface_follow_selection_only:\
    bpy.props.BoolProperty(name="Use Selected Faces", description="Bind only to selected faces", default=False)

    # Vertex Groups
    vg_pin:\
    bpy.props.FloatProperty(name="Pin", description="Pin Vertex Group", default=0, min=0, max=1, soft_min= -2, soft_max=2, precision=3)

    vg_drag:\
    bpy.props.FloatProperty(name="Drag", description="Drag Vertex Group", default=0, min=0, max=1, soft_min= -2, soft_max=2, precision=3)

    vg_surface_follow:\
    bpy.props.FloatProperty(name="Surface Follow", description="Surface Follow Vertex Group", default=0, min=0, max=1, soft_min= -2, soft_max=2, precision=3)

    # Edit Mode
    cloth_grab:\
    bpy.props.BoolProperty(name="Cloth Grab", description="Only move cloth during modal grab", default=False)

    # Cache
    cache:\
    bpy.props.BoolProperty(name="Cloth Cache", description='Cache animation when running "Animated" or "Continuous"', default=False, update=cb_cache)

    cache_only:\
    bpy.props.BoolProperty(name="Cache Only", description='Cache vertex coordinates for an evaluated mesh without generating cloth data', default=False, update=cb_cache_only)

    overwrite_cache:\
    bpy.props.BoolProperty(name="Overwrite Cache", description="Save over existing frames", default=False)

    internal_cache:\
    bpy.props.BoolProperty(name="Internal Cache", description="Save cache in blend file", default=False)

    cache_desktop:\
    bpy.props.BoolProperty(name="Cache To Desktop", description='Save the cache files to desktop to they are easy to find', update=cb_cache)

    cache_interpolation:\
    bpy.props.BoolProperty(name="Cache Interpolate", description='Interpolate mesh shape between cached frames.', default=True, update=cb_cache)


    # set the default path
    path = bpy.data.filepath
    if path == '':
        path = os.path.expanduser("~/Desktop")

    mc_path = str(pathlib.Path(path).parent)#.joinpath('MC_cache_files')

    cache_folder:\
    bpy.props.StringProperty(name="Cache Folder", description="Directory for saving cache files", default=mc_path, update=cb_cache)

    cache_name:\
    bpy.props.StringProperty(name="Custom Name", description="Custom name for saving multiple cache files", default="Mr. Purrkins The Deadly Cyborg Kitten", update=cb_cache)

    play_cache:\
    bpy.props.BoolProperty(name="Play Cache", description="Play the cached animation", default=False, update=cb_cache_playback)

    start_frame:\
    bpy.props.IntProperty(name="Start Frame", description="Start cache on this frame", default=1)#, update=cb_cache)

    end_frame:\
    bpy.props.IntProperty(name="End Frame", description="End cache on this frame", default=250)#, update=cb_cache)

    current_cache_frame:\
    bpy.props.IntProperty(name="End Frame", description="End cache on this frame", default=1, update=cb_current_cache_frame)

    cloth_off:\
    bpy.props.BoolProperty(name="Cloth Off", description="Mesh Keyframe Only: No cloth behavior", default=False)#, update=cb_cache)

    cache_force:\
    bpy.props.FloatProperty(name="Cache Force", description="Target the cached value as a force", default=1.0, min=0, max=1, soft_min= -100, soft_max=100, precision=3)

    # record
    record:\
    bpy.props.BoolProperty(name="Cloth Record", description="Record changes when 'Continuous'", default=False)

    max_frames:\
    bpy.props.IntProperty(name="Max Frames", description="Record this many", default=1000)#, update=cb_cache)



# create properties ----------------
# scene:
class McPropsScene(bpy.types.PropertyGroup):

    kill_duplicator:\
    bpy.props.BoolProperty(name="kill duplicator/loader", description="", default=False, update=cb_duplicator)

    pause_all:\
    bpy.props.BoolProperty(name="Pause All", description="", default=False, update=cb_pause_all)

    play_all:\
    bpy.props.BoolProperty(name="Play all", description="", default=False, update=cb_play_all)

    delay:\
    bpy.props.FloatProperty(name="Delay", description="Slow down the continuous update", default=0, min=0, max=100)

    play_all:\
    bpy.props.BoolProperty(name="Play all", description="", default=False, update=cb_play_all)

    pause_selected:\
    bpy.props.BoolProperty(name="Cloth Grab", description="Only move cloth during modal grab", default=True)

    run_editmode:\
    bpy.props.BoolProperty(name="Run Editmode", description="Run cloth sim when in edit mode", default=True)

    view_virtual:\
    bpy.props.BoolProperty(name="View Virtual Springs", description="create a mesh to show virtual springs", default=False)
    # make this one a child object that is not selectable.


# ^                                                          ^ #
# ^                     END properties                       ^ #
# ============================================================ #


# ============================================================ #
#                    registered operators                      #
#                                                              #

class MCResetToBasisShape(bpy.types.Operator):
    """Reset the cloth to basis shape"""
    bl_idname = "object.mc_reset_to_basis_shape"
    bl_label = "MC Reset To Basis Shape"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = get_cloth(ob)
        if ob.data.is_editmode:
            cloth.obm = bmesh.from_edit_mesh(ob.data)

            Basis = cloth.obm.verts.layers.shape["Basis"]
            for v in cloth.obm.verts:
                v.co = v[Basis]

        reset_shapes(ob)
        cloth.co = get_co_shape(ob, "Basis")
        cloth.velocity[:] = 0
        cloth.pin_arr[:] = cloth.co
        cloth.feedback[:] = 0
        cloth.select_start[:] = cloth.co

        current_key = ob.data.shape_keys.key_blocks['MC_current']
        current_key.data.foreach_set('co', cloth.co.ravel())
        ob.data.update()
        return {'FINISHED'}


class MCResetSelectedToBasisShape(bpy.types.Operator):
    """Reset the selected verts to basis shape"""
    bl_idname = "object.mc_reset_selected_to_basis_shape"
    bl_label = "MC Reset Selected To Basis Shape"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = get_cloth(ob)
        if ob.data.is_editmode:
            Basis = cloth.obm.verts.layers.shape["Basis"]
            for v in cloth.obm.verts:
                if v.select:
                    v.co = v[Basis]

        reset_shapes(ob)
        bco = get_co_shape(ob, "Basis")
        cloth.co[cloth.selected] = bco[cloth.selected]
        cloth.pin_arr[cloth.selected] = bco[cloth.selected]
        cloth.select_start[cloth.selected] = bco[cloth.selected]
        cloth.feedback[cloth.selected] = 0

        bco[~cloth.selected] = cloth.co[~cloth.selected]
        ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', bco.ravel())
        ob.data.update()

        return {'FINISHED'}


def refresh(cloth):
    ob = cloth.ob
    cloth.obm = get_bmesh(ob)
    if ob.data.is_editmode:
        ob.update_from_editmode()

    manage_vertex_groups(cloth)

    cloth.velocity = Nx3(ob)
    np.copyto(cloth.pin_arr, cloth.co)
    cloth.geometry = get_mesh_counts(ob, cloth.obm)
    # cloth.sew_springs = get_sew_springs() # build
    cloth.obm.verts.ensure_lookup_table()
    cloth.co = get_co_edit(ob)
    cloth.wco = np.copy(cloth.co)
    apply_in_place(cloth.ob, cloth.wco)
    cloth.select_start = np.copy(cloth.co)
    cloth.stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32)
    cloth.selected = np.array([v.select for v in cloth.obm.verts])

    #cloth.ob.update_from_editmode()
    #cloth.obm.verts.ensure_lookup_table()

    cloth.select_start = np.copy(cloth.co)
    cloth.pin_arr = np.copy(cloth.co)
    cloth.vel_zero = np.zeros_like(cloth.co)
    cloth.feedback = np.zeros_like(cloth.co)
    cloth.stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.bend_stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.measure_length = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.measure_dot = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.vdl = stretch_springs_basic(cloth, cloth.target)


class MCRefreshVertexGroups(bpy.types.Operator):
    """Refresh Vertex Group Weights To Cloth Settings"""
    bl_idname = "object.mc_refresh_vertex_groups"
    bl_label = "MC Refresh Vertex Groups"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = get_cloth(ob)
        refresh(cloth)
        return {'FINISHED'}


class MCSurfaceFollow(bpy.types.Operator):
    """Connect points to nearest surface"""
    bl_idname = "object.mc_surface_follow"
    bl_label = "MC Surface Follow"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):

        active = bpy.context.object
        if active is None:
            msg = "Must have an active mesh"
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return {'FINISHED'}

        if active.type != 'MESH':
            msg = "Active object must be a mesh"
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return {'FINISHED'}

        if not active.data.polygons:
            msg = "Must have at least one face in active mesh"
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return {'FINISHED'}

        cloths = [i for i in bpy.data.objects if ((i.MC_props.cloth) & (i is not active) & (i.select_get()))]
        if not cloths:
            msg = "Must select at least one cloth object and an active object to bind to"
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return {'FINISHED'}

        # writes to cloth instance
        create_surface_follow_data(active, cloths)

        return {'FINISHED'}


class MCCreateSewLines(bpy.types.Operator):
    """Create sew lines between sets of points"""
    bl_idname = "object.mc_create_sew_lines"
    bl_label = "MC Create Sew Lines"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        #mode = ob.mode
        #if ob.data.is_editmode:
            #bpy.ops.object.mode_set(mode='OBJECT')

        cloth = MC_data['cloths'][ob['MC_cloth_id']]

        #bco = get_co_shape(ob, "Basis")

        #bpy.ops.object.mode_set(mode=mode)
        #ob.data.update()
        return {'FINISHED'}


class MCSewToSurface(bpy.types.Operator):
    """Draw a line on the surface and sew to it"""
    bl_idname = "object.mc_sew_to_surface"
    bl_label = "MC Create Surface Sew Line"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        #mode = ob.mode
        #if ob.data.is_editmode:
            #bpy.ops.object.mode_set(mode='OBJECT')

        cloth = MC_data['cloths'][ob['MC_cloth_id']]

        #bco = get_co_shape(ob, "Basis")

        #bpy.ops.object.mode_set(mode=mode)
        #ob.data.update()
        return {'FINISHED'}


# !!! can't undo delete cache or keyframe.
class MCDeleteCache(bpy.types.Operator):
    """Delete cache folder for current object"""
    bl_idname = "object.mc_delete_cache"
    bl_label = "Really delete this folder and its contents?"
    bl_options = {'REGISTER', 'UNDO'}


    def __init__(self):
        print("ran this class method thingy")

        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        if ob.MC_props.cloth:
            cloth = MC_data['cloths'][ob['MC_cloth_id']]

            path = pathlib.Path(ob.MC_props.cache_folder).joinpath('MC_cache_files')
            current = path.joinpath(ob.MC_props.cache_name)

        msg = 'Really delete ' + str(current) + ' and its contents?'

    @classmethod
    def poll(cls, context):
        return True

    def invoke(self, context, event):
        self.bl_label = "Object: {0}".format(context.active_object.name)

        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):

        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = MC_data['cloths'][ob['MC_cloth_id']]

        path = pathlib.Path(ob.MC_props.cache_folder).joinpath('MC_cache_files')
        current = path.joinpath(ob.MC_props.cache_name)

        if os.path.exists(current):
            shutil.rmtree(current, ignore_errors=True)
            ob.MC_props['cache'] = False
            ob.MC_props['play_cache'] = False

            msg = 'Deleted ' + str(current) + ' and its contents.'
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')

            return {'FINISHED'}

        msg = 'Could not delete ' + str(current) + '. No such file'
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')

        return {'FINISHED'}







class MCCreateMeshKeyframe(bpy.types.Operator):
    """Create a linear path between cache files"""
    bl_idname = "object.mc_mesh_keyframe"
    bl_label = "MC Create Mesh Keyframe"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = MC_data['cloths'][ob['MC_cloth_id']]
        print("Will create mesh cache keyframe... Some day. Sigh...")
        cloth.co = get_co_mode(ob)
        overwrite = ob.MC_props.overwrite_cache
        ob.MC_props.overwrite_cache = True
        cache(cloth, keying=True)
        ob.MC_props.overwrite_cache = overwrite


        return {'FINISHED'}


class MCRemoveMeshKeyframe(bpy.types.Operator):
    """Remove mesh keyframe at current frame"""
    bl_idname = "object.mc_remove_keyframe"
    bl_label = "MC Remove Mesh Keyframe"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = MC_data['cloths'][ob['MC_cloth_id']]
        print("Will remove mesh cache keyframe... Some day. Sigh...")

        return {'FINISHED'}


class MCCreateVirtualSprings(bpy.types.Operator):
    """Create Virtual Springs Between Selected Verts"""
    bl_idname = "object.mc_create_virtual_springs"
    bl_label = "MC Create Virtual Springs"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        cloth = get_cloth(ob)
        obm = cloth.obm
        verts = np.array([v.index for v in obm.verts if v.select])
        cloth.virtual_spring_verts = verts
        virtual_springs(cloth)

        return {'FINISHED'}


class MCApplyForExport(bpy.types.Operator):
    # !!! Not Finished !!!!!!
    """Apply cloth effects to mesh for export."""
    bl_idname = "object.MC_apply_for_export"
    bl_label = "MC Apply For Export"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = get_last_object()[1]
        v_count = len(ob.data.vertices)
        co = np.zeros(v_count * 3, dtype=np.float32)
        ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_get('co', co)
        ob.data.shape_keys.key_blocks['Basis'].data.foreach_set('co', co)
        ob.data.shape_keys.key_blocks['Basis'].mute = True
        ob.data.shape_keys.key_blocks['Basis'].mute = False
        ob.data.vertices.foreach_set('co', co)
        ob.data.update()

        return {'FINISHED'}


class MCVertexGroupPin(bpy.types.Operator):
    """Add Selected To Pin Vertex Group"""
    bl_idname = "object.mc_vertex_group_pin"
    bl_label = "MC Vertex Group Pin"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        cloth = MC_data['cloths'][ob['MC_cloth_id']]

        return {'FINISHED'}


# ^                                                          ^ #
#                  END registered operators                    #
# ============================================================ #


# ============================================================ #
#                         draw code                            #
#                                                              #
class PANEL_PT_MC_Master(bpy.types.Panel):
    """MC Panel"""
    bl_label = "MC Master Panel"
    bl_idname = "PANEL_PT_mc_master_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    @classmethod
    def poll(cls, context):
        ob = bpy.context.object
        if ob is None:
            return False

        if ob.type != 'MESH':
            if MC_data['recent_object'] is None:
                return False
            ob = MC_data['recent_object']

        try: # if we delete the object then grab an empty
            ob.name
        except:
            print("Cloth object was deleted")
            MC_data['recent_object'] = None
            return False

        return True

    def __init__(self):
        ob = bpy.context.object

        if ob.type != 'MESH':
            ob = MC_data['recent_object']

        self.ob = ob
        self.cloth = ob.MC_props.cloth


# MAIN PANEL
class PANEL_PT_modelingClothMain(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Main"""
    bl_label = "MC Main"
    bl_idname = "PANEL_PT_modeling_cloth_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        sc = bpy.context.scene
        ob = self.ob
        layout = self.layout
        col = layout.column(align=True)
        col.prop(sc.MC_props, "kill_duplicator", text="kill_duplicator", icon='DUPLICATE')
        col.prop(ob.MC_props, "cache_only", text="Cache Only", icon='RENDER_ANIMATION')
        col.prop(ob.MC_props, "dense", text="Dense Mesh", icon='VIEW_ORTHO')
        col.prop(ob.MC_props, "p1_bend_trigger", text="Trigger p1", icon='QUIT')
        col.prop(ob.MC_props, "quad_bend", text="Quad Bend Springs", icon='VIEW_PERSPECTIVE')

        col = layout.column(align=True)
        col.scale_y = 1.5
        col.prop(ob.MC_props, "collider", text="Collide", icon='MOD_PHYSICS')
        if ob.MC_props.collider:
            row = col.row()
            #col = layout.column(align=True)
            row.scale_y = 0.75
            row.label(icon='PROP_CON')
            row.prop(ob.MC_props, "inner_margin", text="Inner Margin", icon='PROP_CON')
            row = col.row()
            row.scale_y = 0.75
            row.label(icon='PROP_OFF')
            row.prop(ob.MC_props, "outer_margin", text="Outer Margin", icon='PROP_OFF')
        col.prop(ob.MC_props, "self_collide", text="Self Collision", icon='FULLSCREEN_EXIT')
        if ob.MC_props.self_collide:
            row = col.row()
            #col = layout.column(align=True)
            row.scale_y = 0.75
            row.label(icon='PROP_CON')
            row.prop(ob.MC_props, "self_collide_margin", text="SC Margin", icon='PROP_CON')
            row = col.row()
            row.label(icon='CON_PIVOT')
            row.scale_y = 0.75            
            row.prop(ob.MC_props, "sc_friction", text="Friction", icon='CON_PIVOT')            
            row = col.row()
            row.label(icon='CON_PIVOT')            
            row.scale_y = 0.75            
            row.prop(ob.MC_props, "sc_vel_damping", text="Damping", icon='CON_PIVOT')
            row = col.row()
            row.label(icon='CON_PIVOT')
            row.scale_y = 0.75            
            row.prop(ob.MC_props, "self_collide_force", text="SC Force", icon='CON_PIVOT')
            row = col.row()            
            row = col.row()
            row.label(icon='CON_PIVOT')
            row.scale_y = 0.75            
            row.prop(ob.MC_props, "sc_box_max", text="Box Size", icon='CON_PIVOT')


        # use current mesh or most recent cloth object if current ob isn't mesh

        # if we select a new mesh object we want it to display

        col = layout.column(align=True)
        col.scale_y = 1.5
        # display the name of the object if "cloth" is True
        #   so we know what object is the recent object
        recent_name = ''
        if ob.MC_props.cloth:
            recent_name = ob.name

        col.prop(ob.MC_props, "cloth", text="Cloth " + recent_name, icon='MOD_CLOTH')
        if ob.MC_props.cloth:
            #if self.mesh:
            col.prop(ob.MC_props, "detect_collisions", text="Detect Collisions", icon='LIGHT_AREA')
            col.label(text='Update Mode')
            col = layout.column(align=True)
            row = col.row()
            row.scale_y = 2
            row.prop(ob.MC_props, "continuous", text="Continuous", icon='FILE_REFRESH')
            row = col.row()
            row.scale_y = 2
            row.prop(ob.MC_props, "animated", text="Animated", icon='PLAY')
            row = col.row()
            row.scale_y = 1
            row.prop(sc.MC_props, "delay", text="Delay", icon='SORTTIME')
            box = col.box()
            #row = col.row()
            box.scale_y = 2
            box.operator('object.mc_reset_to_basis_shape', text="RESET", icon='RECOVER_LAST')
            box.operator('object.mc_reset_selected_to_basis_shape', text="RESET SELECTED", icon='RECOVER_LAST')
            if ob.data.is_editmode:
                box.operator('object.mc_refresh_vertex_groups', text="V-group Refresh", icon='GROUP_VERTEX')
            col = layout.column(align=True)
            col.use_property_decorate = True
            col.label(text='Target Object')
            col.prop(ob.MC_props, "target", text="", icon='DUPLICATE')


# CACHE PANEL
class PANEL_PT_modelingClothCache(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Cache"""
    bl_label = "MC Cache"
    bl_idname = "PANEL_PT_modeling_cloth_cache"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        sc = bpy.context.scene
        layout = self.layout
        col = layout.column(align=True)
        ob = self.ob

        if ob.MC_props.cloth:
            cloth = get_cloth(ob)

            col = layout.column(align=True)
            box = col.box()
            bcol = box.column()

            bcol.prop(ob.MC_props, "cache", text="Record", icon='RENDER_ANIMATION')
            bcol.prop(ob.MC_props, "max_frames")
            bcol.prop(ob.MC_props, "overwrite_cache", text="Overwrite", icon='FILE_REFRESH')
            bcol.prop(ob.MC_props, "internal_cache", text="Pack", icon='NORMALIZE_FCURVES')
            bcol = box.column()
            bcol.scale_y = 1.5
            bcol.operator('object.mc_delete_cache', text="Delete Cache", icon='KEY_HLT')
            bcol = box.column()
            bcol.label(text="Cache Foder")
            bcol.prop(ob.MC_props, "cache_folder", text="")
            bcol.label(text="Custom Name")
            bcol.prop(ob.MC_props, "cache_name", text="")
            bcol.prop(ob.MC_props, "cache_desktop", text="Use Desktop")

            #col.separator()
            bcol = col.box().column()

            bcol.label(text='Animate Mode')

            #bcol.enabled = hasattr(cloth, 'cache_dir')
            bcol.enabled = check_file_path(ob)[0]

            bcol.prop(ob.MC_props, 'play_cache', text='Playback', icon='PLAY')
            bcol.prop(ob.MC_props, 'cache_force', text='Influence', icon='SNAP_ON')
            bcol.prop(ob.MC_props, 'current_cache_frame', text='Frame')#, icon='SNAP_ON')



            #row = bcol.row()
            #row.prop(ob.MC_props, "start_frame", text="")
            #row.prop(ob.MC_props, "end_frame", text="")
            #bcol.label(text='Continuous Mode')
            #bcol.prop(ob.MC_props, "record", text="Record", icon='REC')




            folder = 'Cach=None'
            if hasattr(cloth, 'cache_dir'):
                fo = cloth.cache_dir
                if os.path.exists(fo):
                    folder = fo

            col.separator()
            box = col.box().column()
            box.label(text='Mesh Keyframing')
            box.prop(ob.MC_props, 'cloth_off', text='Cloth Off', icon='CANCEL')

            box.separator()

            frame = 'Key=None'

            if hasattr(cloth, 'cache_dir'):
                folder = cloth.cache_dir
                if folder.exists():
                    f = str(folder) + str(sc.frame_current)
                    if os.path.exists(f):
                        frame = 'Key=' + str(sc.frame_current)


            box.operator('object.mc_mesh_keyframe', text="Keyframe Mesh", icon='KEY_HLT')
            box.label(text=frame)
            box.operator('object.mc_remove_keyframe', text="Del Keyframe", icon='KEY_DEHLT')



            #col.separator()
            #box = col.box().column()
            #box.label(text='Record Continuous')
            #box.prop(ob.MC_props, "record", text="Record", icon='REC')
            #box.prop(ob.MC_props, "max_frames")
            #col.separator()
            #box = col.box().column()
            #box.label(text='Playback Mode')
            #box.prop(ob.MC_props, 'play_cache', text='Playback', icon='PLAY')
            #box.prop(ob.MC_props, 'cache_force', text='Influence', icon='SNAP_ON')


# SEWING PANEL
class PANEL_PT_modelingClothSewing(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Settings"""
    bl_label = "MC Sewing"
    bl_idname = "PANEL_PT_modeling_cloth_sewing"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        ob = self.ob
        cloth = ob.MC_props.cloth
        if cloth:
            sc = bpy.context.scene
            layout = self.layout

            # use current mesh or most recent cloth object if current ob isn't mesh
            MC_data['recent_object'] = ob
            col = layout.column(align=True)
            col.scale_y = 1
            col.label(text='Sewing')
            col.prop(ob.MC_props, "sew_force", text="sew_force")

            box = col.box()
            box.scale_y = 2
            box.operator('object.mc_create_virtual_springs', text="Virtual Springs", icon='AUTOMERGE_ON')
            box.operator('object.mc_create_sew_lines', text="Sew Lines", icon='AUTOMERGE_OFF')
            box.operator('object.mc_sew_to_surface', text="Surface Sewing", icon='MOD_SMOOTH')

            return
        sc = bpy.context.scene
        layout = self.layout
        col = layout.column(align=True)
        box = col.box()
        box.scale_y = 2
        box.operator('object.mc_surface_follow', text="Follow Surface", icon='OUTLINER_DATA_SURFACE')
        box = col.box()
        box.scale_y = 1
        box.prop(ob.MC_props, "surface_follow_selection_only", text="Selected Polys Only")


# SETTINGS PANEL
class PANEL_PT_modelingClothSettings(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Settings"""
    bl_label = "MC Settings"
    bl_idname = "PANEL_PT_modeling_cloth_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        ob = self.ob
        cloth = ob.MC_props.cloth
        if cloth:
            sc = bpy.context.scene
            layout = self.layout

            # use current mesh or most recent cloth object if current ob isn't mesh
            MC_data['recent_object'] = ob
            col = layout.column(align=True)
            col.scale_y = 1.5
            col.label(text='Forces')
            col.prop(ob.MC_props, "gravity", text="gravity")
            col.prop(ob.MC_props, "velocity", text="velocity")

            #col.scale_y = 1
            col = layout.column(align=True)
            col.prop(ob.MC_props, "sub_frames", text="Sub Frames")
            col.prop(ob.MC_props, "stretch_iters", text="stretch iters")
            col.prop(ob.MC_props, "stretch", text="stretch")
            col.prop(ob.MC_props, "push", text="push")
            col.prop(ob.MC_props, "feedback", text="feedback")
            col.prop(ob.MC_props, "bend_iters", text="bend iters")
            col.prop(ob.MC_props, "bend", text="bend")


# VERTEX GROUPS PANEL
class PANEL_PT_modelingClothVertexGroups(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Vertex Groups"""
    bl_label = "MC Vertex Groups"
    bl_idname = "PANEL_PT_modeling_cloth_vertex_groups"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        ob = self.ob
        cloth = ob.MC_props.cloth
        if cloth:
            sc = bpy.context.scene
            layout = self.layout

            # use current mesh or most recent cloth object if current ob isn't mesh
            MC_data['recent_object'] = ob
            col = layout.column(align=True)
            col.scale_y = 1
            col.label(text='Vertex Groups')
            col.operator('object.mc_vertex_group_pin', text="Pin Selected", icon='PINNED')
            col.prop(ob.MC_props, "vg_pin", text="pin")
            col.prop(ob.MC_props, "vg_drag", text="drag")
            col.prop(ob.MC_props, "vg_surface_follow", text="Surface Follow")


# EDIT MODE PANEL
class PANEL_PT_modelingClothPreferences(bpy.types.Panel):
    """Modeling Cloth Preferences"""
    bl_label = "MC Preferences"
    bl_idname = "PANEL_PT_modeling_cloth_preferences"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        sc = bpy.context.scene
        layout = self.layout
        col = layout.column(align=True)
        col.scale_y = 1
        col.label(text='Preferences')
        col.prop(sc.MC_props, "run_editmode", text="Editmode Run")
        col.prop(sc.MC_props, "pause_selected", text="Pause Selected")
        col.prop(sc.MC_props, "view_virtual", text="View Virtual Springs")

# ^                                                          ^ #
# ^                     END draw code                        ^ #
# ============================================================ #


# testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#install_handler(False)


# testing end !!!!!!!!!!!!!!!!!!!!!!!!!



# ============================================================ #
#                         Register                             #
#                                                              #
def duplication_and_load():
    """Runs in it's own handler for updating objects
    that are duplicated while coth properties are true.
    Also checks for cloth, collider, and target objects
    in the file when blender loads."""
    # for loading no need to check for duplicates because we are regenerating data for everyone
    #if load:
    #    return 0


    # for detecting duplicates
    obm = False # because deepcopy doesn't work on bmesh
    print("running duplicator")
    cloths = [i for i in bpy.data.objects if i.MC_props.cloth]
    if len(cloths) > 0:
        id = [i['MC_cloth_id'] for i in cloths]
        idx = max(id) + 1
        u, inv, counts = np.unique(id, return_inverse=True, return_counts=True)
        repeated = counts[inv] > 1
        if np.any(repeated):
            dups = np.array(cloths)[repeated]
            for i in dups:
                cloth_instance = MC_data['cloths'][i['MC_cloth_id']]
                cloth_instance.ob = None
                cloth_instance.target = None # objs don't copy with deepcopy
                if 'obm' in dir(cloth_instance):
                    obm = True # objs don't copy with deepcopy
                    cloth_instance.obm = None # objs don't copy with deepcopy
                MC_data['cloths'][idx] = copy.deepcopy(cloth_instance)
                MC_data['cloths'][idx].ob = i # cloth.ob doesn't copy
                MC_data['cloths'][idx].target = i.MC_props.target # cloth.ob doesn't copy

                # not sure if I need to remake the bmesh since it will be remade anyway... Can't duplicate an object in edit mode. If we switch to edit mode it will remake the bmesh.
                #if obm:
                    #MC_data['cloths'][idx].obm = get_bmesh(i) # bmesh doesn't copy

                i['MC_cloth_id'] = idx
                idx += 1

            print("duplicated an object cloth instance here=============")
            # remove the cloth instances that have been copied
            for i in np.unique(np.array(id)[repeated]):
                MC_data['cloths'].pop(i)

    print("finished duplication =====+++++++++++++++++++++++++")
    colliders = [i for i in bpy.data.objects if i.MC_props.cloth]
    return 1


classes = (
    McPropsObject,
    McPropsScene,
    MCResetToBasisShape,
    MCResetSelectedToBasisShape,
    MCRefreshVertexGroups,
    MCSurfaceFollow,
    MCCreateSewLines,
    MCSewToSurface,
    MCCreateVirtualSprings,
    MCDeleteCache,
    MCCreateMeshKeyframe,
    MCRemoveMeshKeyframe,
    PANEL_PT_modelingClothMain,
    PANEL_PT_modelingClothCache,
    PANEL_PT_modelingClothSettings,
    PANEL_PT_modelingClothSewing,
    PANEL_PT_modelingClothVertexGroups,
    PANEL_PT_modelingClothPreferences,
    MCVertexGroupPin,
)


def register():
    # classes
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    # props
    bpy.types.Object.MC_props = bpy.props.PointerProperty(type=McPropsObject)
    bpy.types.Scene.MC_props = bpy.props.PointerProperty(type=McPropsScene)

    # clean dead versions of the undo handler
    handler_names = np.array([i.__name__ for i in bpy.app.handlers.undo_post])
    booly = [i == 'undo_frustration' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.undo_post[i])
        print("deleted handler ", i)

    # drop in the undo handler
    bpy.app.handlers.undo_post.append(undo_frustration)

    # register the data management timer. Updates duplicated objects and objects with modeling cloth properties
    if False:
        bpy.app.timers.register(duplication_and_load)

    # special forces -------------
    bpy.types.Scene.MC_seam_wrangler = False


def unregister():
    # classes
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    # props
    del(bpy.types.Scene.MC_props)
    del(bpy.types.Object.MC_props)

    # clean dead versions of the undo handler
    handler_names = np.array([i.__name__ for i in bpy.app.handlers.undo_post])
    booly = [i == 'undo_frustration' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.undo_post[i])

if __name__ == "__main__":
    register()
