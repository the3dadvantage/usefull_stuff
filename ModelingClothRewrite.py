


""" New Features: """
# pause button using space bar or something with modal grab
# cache file option
# awesome sew doesn't care how many verts (target a location based on a percentage of the edge)
# run cloth in edit mode????
# Could I pull the normals off a normal map and add them to the bend springs for adding wrinkles?
# For adding wrinkles maybe I can come up with a clever way to put them into the bend springs.
# Could even create a paint modal tool that expands the source where the user paints to create custom wrinkles.
#   Wrinkle brush could take into account stroke direction, or could grow in all directions.
#   Crease brush would be different making wrinkles more like what you would need to iron out.

# Target:
# Could make the cloth always read from the source shape key
#   and just update target changes to the source shape.

# Bugs (not bunny)
# Don't currently have a way to update settings on duplicated objects.
#   !! could load a separate timer that both updates cloth objects when
#       loading a saved file and updates duplicates that have cloth properties
#   I think it would load whenever blender reopens if the module is registered
#   I'll have to see if I need to regen springs or if the global variable is overwritten every time

import bpy
from bpy.ops import op_as_string
from bpy.app.handlers import persistent
import os
import bmesh
import functools as funky
import numpy as np
from numpy import newaxis as nax
import time
import copy # for duplicate cloth objects

# global data
MC_data = {}
MC_data['colliders'] = {}
MC_data['cloths'] = {}
MC_data['iterator'] = 0

# recent_object allows cloth object in ui
#   when selecting empties such as for pinning.
MC_data['recent_object'] = None


def reload():
    """!! for development !! Resets everything"""
    # when this is registered as an addon I will want
    #   to recaluclate these objects not set prop to false
    # set all props to false:
    reload_props = ['continuous', 'collider', 'animated', 'cloth']
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


print("new--------------------------------------")
def cache(ob):
    # for when I cache animation data
    # store the cache in the place where
    # the blend file is saved
    # if the blend file is not saved should
    #   prolly store it in a tmp directors...
    folder = bpy.data.filepath
    file = os.path.join(folder, ob.name)


# debugging
def T(type=1, message=''):
    if type == 1:
        return time.time()
    print(time.time() - type, message)


# ============================================================ #
#                    universal functions                       #
#                                                              #

# universal ---------------
def co_overwrite(ob, ar):
    """Fast way to overwite"""
    shape = ar.shape
    ar.shape = (shape[0] * 3,)
    ob.data.vertices.foreach_get('co', ar)
    ar.shape = shape


# universal ---------------
def get_bmesh(ob=None):
    if ob.mode == 'OBJECT':
        obm = bmesh.new()
        obm.from_mesh(ob.data)
    elif ob.mode == 'EDIT':
        obm = bmesh.from_edit_mesh(ob.data)
    return obm


# universal ---------------
def get_co(ob, ar=None):
    """Get vertex coordinates from an object in object mode"""
    if ar is None:
        v_count = len(ob.data.vertices)
        ar = np.empty(v_count * 3, dtype=np.float32)
    ar.shape = (v_count * 3,)
    ob.data.vertices.foreach_get('co', ar)
    ar.shape = (v_count, 3)
    return ar


# universal ---------------
def get_co_shape(ob, key=None, ar=None):
    """Get vertex coords from a shape key"""
    v_count = len(ob.data.shape_keys.key_blocks[key].data)
    if ar is None:
        ar = np.empty(v_count * 3, dtype=np.float32)
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar)
    ar.shape = (v_count, 3)
    return ar


# universal ---------------
def get_co_edit(ob, obm=None):
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
        obm = bmesh.from_edit_mesh(ob.data)
        obm.verts.ensure_lookup_table()
        count = (len(obm.verts))
    else:
        count = len(ob.data.vertices)
    ar = np.zeros(count*3, dtype=np.float32)
    ar.shape = (count, 3)
    return ar


# universal ---------------
def get_co_mode(ob):
    """Edit or object mode"""
    if ob is None: # cloth.target object might be None
        return None
    if ob.data.is_editmode:
        obm = bmesh.from_edit_mesh(ob.data)
        obm.verts.ensure_lookup_table()
        co = np.array([i.co for i in obm.verts], dtype=np.float32)
        return co

    count = len(ob.data.vertices)
    ar = np.zeros(count*3, dtype=np.float32)
    ob.data.vertices.foreach_get('co', ar)
    ar.shape = (count, 3)
    return ar


# universal ---------------
def compare_geometry(ob1, ob2, obm1=None, obm2=None, all=False):
    """Check for differences in verts, edges, and faces between two objects"""
    # if all is false we're comparing a target objects. verts in faces
    #   and faces must match.
    def get_counts(obm):
        v_count = len([v for v in obm.verts if len(v.link_faces) > 0])
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
def reset_shapes(ob):
    """Create shape keys if they are missing"""

    if ob.data.shape_keys == None:
        ob.shape_key_add(name='Basis')
    if 'MC_source' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add(name='MC_source')
    if 'MC_current' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add(name='MC_current')
        ob.data.shape_keys.key_blocks['MC_current'].value=1


# universal ---------------
def get_weights(ob, name, obm=None):
    """Get vertex weights. If no weight is assigned assign
    and set weight to zero"""
    # might want to look into using map()
    count = len(ob.data.vertices)

    g_idx = ob.vertex_groups[name].index

    # for edit mode:
    if ob.data.is_editmode:
        count = len(obm.verts)
        arr = np.zeros(count, dtype=np.float32)
        dvert_lay = obm.verts.layers.deform.active

        for idx, v in enumerate(obm.verts):
            dvert = v[dvert_lay]

            if g_idx in dvert:
                arr[idx] = dvert[g_idx]
            else:
                dvert[g_idx] = 0
                arr[idx] = 0
        return arr

    arr = np.zeros(count, dtype=np.float32)

    g = ob.vertex_groups[name]
    for idx, v in enumerate(ob.data.vertices):
        try:
            w = g.weight(idx)
            arr[idx] = w
        except RuntimeError:
            # Raised when the vertex is not part of the group
            w = 0
            ob.vertex_groups['MC_pin'].add([g_idx, idx], w, 'REPLACE')
            arr[idx] = w

    return arr
# ^                                                          ^ #
# ^                 END universal functions                  ^ #
# ============================================================ #




# ============================================================ #
#                   precalculated data                         #
#                                                              #

# precalculated ---------------
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


# precalculated ---------------
def get_springs(cloth, obm=None, debug=False):
    """Creates minimum edges and indexing arrays for spring calculations"""
    if obm is None:
        obm = get_bmesh(cloth.ob)

    # get index of verts related by polygons for each vert
    id = [[[v.index for v in f.verts if v != i] for f in i.link_faces] for i in obm.verts]

    # strip extra dimensions
    squeezed = [np.unique(np.hstack(i)) if len(i) > 0 else i for i in id]

    # create list pairing verts with connected springs
    paired = [[i, np.ones(len(i), dtype=np.int32)*r] for i, r in zip(squeezed, range(len(squeezed)))]

    # merge and cull disconnected verts
    merged1 = [np.append(i[0][nax], i[1][nax], axis=0) for i in paired if len(i[0]) > 0]
    a = np.hstack([i[0] for i in merged1])
    b = np.hstack([i[1] for i in merged1])
    final = np.append(a[nax],b[nax], axis=0).T
    springs = eliminate_duplicate_pairs(final)

    # generate indexers for add.at
    v_fancy, e_fancy, flip = b, [], []
    nums = np.arange(len(obm.verts)) # need verts in case there are disconnected verts in mesh
    idxer = np.arange(springs.shape[0] * 2)
    flat_springs = springs.T.ravel()
    mid = springs.shape[0]

    # would be nice to speed this up a bit...
    for i in nums:
        hit = idxer[i==flat_springs]
        if len(hit) > 0: # nums and idxer can be different because of disconnected verts
            for j in hit:
                if j < mid:
                    e_fancy.append(j)
                    flip.append(1)
                else:
                    e_fancy.append(j - mid)
                    flip.append(-1)

    #debug=True
    if debug:
        print('---------------------------------------------')
        for i, j, k, in zip(v_fancy, springs[e_fancy], flip):
            print(i,j,k)

    print("created springs --------------------------------------")
    return springs, v_fancy, e_fancy, np.array(flip)[:, nax]



# ^                                                          ^ #
# ^                 END precalculated data                   ^ #
# ============================================================ #

# ============================================================ #
#                      cloth instance                          #
#                                                              #

# collider instance -----------------
class Collider(object):
    # The collider object
    pass


# collider instance -----------------
def create_collider():
    collider = Collider()
    collider.name = "Henry"
    collider.ob = bpy.context.object
    return collider


# cloth instance ---------------
class Cloth(object):
    # The cloth object
    pass


# cloth instance ---------------
def create_instance():
    """Run this when turning on modeling cloth."""
    cloth = Cloth()
    ob = bpy.context.object
    cloth.ob = ob

    # check for groups:
    if 'MC_pin' not in ob.vertex_groups:
        ob.vertex_groups.new(name='MC_pin')

    # drag can be something like air friction
    if 'MC_drag' not in ob.vertex_groups:
        ob.vertex_groups.new(name='MC_drag')


    # stuff ---------------
    cloth.obm = get_bmesh(ob)

    if ob.data.is_editmode:
        cloth.obm.verts.layers.deform.verify()
        #cloth.obm.verts.layers.deform.new()
        update_groups(cloth, cloth.obm)
    else:
        update_groups(cloth)

    cloth.undo = False

    # the target for stretch and bend springs
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

    t = T()
    cloth.springs, cloth.v_fancy, cloth.e_fancy, cloth.flip = get_springs(cloth)
    T(t, "time it took to get the springs")

    # coordinates and arrays
    print(ob.name, '!!!!!!!!!!!!!!!! name!!')
    if ob.data.is_editmode:
        cloth.co = get_co_mode(ob)
    else:
        cloth.co = get_co_shape(ob, 'MC_current')

    # for pinning
    cloth.pin_arr = np.copy(cloth.co)

    # velocity array
    cloth.vel_start = np.copy(cloth.co)
    cloth.vel = np.copy(cloth.co)

    cloth.target_co = get_co_mode(cloth.target) # (will be None if target is None)
    cloth.velocity = Nx3(ob)

    # target co
    same = False
    if cloth.target is not None: # if we are doing a two way update we will need to put run the updater here
        same = compare_geometry(ob, cloth.target, cloth.obm, cloth.target_obm)
        if same:
            cloth.source, cloth.dots = stretch_springs(cloth, cloth.target)

    if not same: # there is no matching target or target is None
        print("we are somehow here !!!!!!!!!!!!!!!!")
        print(cloth.co)
        cloth.v, cloth.source, cloth.dots = stretch_springs(cloth)



    return cloth

# ^                                                          ^ #
# ^                   END cloth instance                     ^ #
# ============================================================ #



# ============================================================ #
#                     update the cloth                         #
#                                                              #

# update the cloth ---------------
def update_groups(cloth, obm=None):
    # needs to work in edit mode!!!
    ob = cloth.ob
    # vertex groups
    cloth.pin = get_weights(ob, 'MC_pin', obm=obm)[:, nax]
    #pin_arr = np.zeros(pin.shape[0]*3, dtype=np.float32)
    #pin_arr.shape = (pin.shape[0], 3)
    cloth.pin_arr = get_co_mode(ob)


# update the cloth ---------------
def collide(colliders=None):
    print('checked collisions')


# update the cloth ---------------
def obm_co(a, b, i):
    """Runs inside a list comp attempting to improve speed"""
    a[i].co = b[i]


# update the cloth ---------------
def changed_geometry(ob1, ob2):
    """Copy the geometry of ob1 to ob2"""
    # when extruding geometry there is a face
    # or faces the extrued verts come from
    # get barycentric mapping from that face
    # or faces. Remap that to the current
    # state of the cloth object.

    """  OR!!!  """

    # FOR ADDING (so detect if we're adding verts)
    # (If we're just filling in faces it should be more simple)
    # the new verts will always be selected and old ones
    #   will always be deselected.
    # take a shapshot of the current cloth state's
    #   vertex locations
    # overwrite the mesh
    # update the coords from previous verts to the
    # shapshot.


    # FOR REMOVING (should only be for removing verts)
    # no way to know what verts will get deleted
    #



# update the cloth ---------------
def measure_edges(co, idx):
    """Takes a set of coords and an edge idx and measures segments"""
    l = idx[:,0]
    r = idx[:,1]
    v = co[r] - co[l]
    d = np.einsum("ij ,ij->i", v, v)
    return v, d, np.sqrt(d)


# update the cloth ---------------
def stretch_springs(cloth, target=None): # !!! need to finish this
    """Measure the springs"""
    if target is not None:
        dg = bpy.context.evaluated_depsgraph_get()
        #proxy = col.ob.to_mesh(bpy.context.evaluated_depsgraph_get(), True, calc_undeformed=False)
        #proxy = col.ob.to_mesh() # shouldn't need to use mesh proxy because I'm using bmesh
        proxy = target.evaluated_get(dg)
        co = get_co_mode(proxy) # needs to be depsgraph eval
        return measure_edges(co, cloth.springs)

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

    return measure_edges(co, cloth.springs)


# update the cloth ---------------
def spring_force(cloth):


    dynamic = True
    if dynamic:
        # springs (vec, dot, length) # could put: if dynamic
        v, d, l = stretch_springs(cloth, cloth.target) # from target or source key


    # for pinning
    np.copyto(cloth.vel_start, cloth.co)

    # add gravity before measuring stretch
    grav = np.array([0,0,-0.01])
    cloth.co += grav

    iters = 2
    for i in range(iters):
    # apply pin force
        pin_vecs = (cloth.pin_arr - cloth.co) * cloth.pin
        cloth.co += pin_vecs


        # (current vec, dot, length)
        cv, cd, cl = measure_edges(cloth.co, cloth.springs) # from current cloth state

        spring_dif = cl - l

        #push = spring_dif < 0

        #spring_dif[~push] *= 0
        '''Applys the magnitudes of v1 to v2 assuming N x 3 arrays'''
        # get the move vecs with mag swap
        div = np.nan_to_num(d/cd)
        #div = d/cd

        if False: # this might be a good way to create heat
            div[~push] *= 0

        #div[push] *= .5

        swap = cv * np.sqrt(div)[:, nax]
        move = cv - swap
        #if i == 0:
            #move[~push] *= 4
            #move *= 4
        #if i == 1:
            #move[~push] *= 2

        #if i == 1:
            #move[push] *= 0


        if True: # experimental damping
            move *= np.sqrt(div)[:, nax]

        stretch = cloth.ob.MC_props.stretch
        if i == 0:
            stretch *= 2
        move *= stretch
        # now get the length of move so we can square it


        # post indexing ------------------------
        flipped = move[cloth.e_fancy]
        flipped *= cloth.flip




        # for pinning
        #np.copyto(cloth.pin_arr, cloth.co)

        np.add.at(cloth.co, cloth.v_fancy, flipped)

        # apply pin force
        pin_vecs = (cloth.pin_arr - cloth.co)
        cloth.co += (pin_vecs * cloth.pin)



        #if cloth.ob.MC_props.pause_selected:
        pause_selected = True
        if cloth.ob.data.is_editmode:
            if pause_selected:

                # need to check if copyto is faster than:
                #cloth.co[cloth.selected] = cloth.pin_arr[cloth.selected]

                np.copyto(cloth.co, cloth.vel_start, where=cloth.selected[:,nax])
                np.copyto(cloth.pin_arr, cloth.vel_start, where=cloth.selected[:,nax])

        #spring_move = cloth.co - cloth.vel_start
        #cloth.co += spring_move * .5






# update the cloth ---------------
def cloth_physics(ob, cloth, collider):
    # can run in edit mode if I deal with looping through bmesh verts
    # will also have to check for changes in edge count, vert count, and face count.
    # if any of the geometry counts change I will need to update springs and other data.
    # If there is a proxy object will need to check that they match and issue
    #   warnings if there is a mismatch. Might want the option to regen the proxy object
    #   or adapt the cloth object to match so I can sync a patter change
    #   given you can change both now I have to also include logic to
    #   decide who is boss if both are changed in different ways.
    grav = np.array([0,0,-0.1])
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
        if cloth.target is not None:
            if not cloth.data.is_editmode: # can use bmesh prolly if not OBJECT mode.
                dg = bpy.context.evaluated_depsgraph_get()
                #proxy = col.ob.to_mesh(bpy.context.evaluated_depsgraph_get(), True, calc_undeformed=False)
                #proxy = col.ob.to_mesh() # shouldn't need to use mesh proxy because I'm using bmesh
                proxy = cloth.target.evaluated_get(dg)
                co_overwrite(proxy, cloth.target_co)


    if ob.data.is_editmode:
        # prop to go into user preferences. (make it so it won't run in edit mode)
        pause_in_edit = False
        if pause_in_edit:
            return

        # if we switch to a different shape key in edit mode:
        index = ob.data.shape_keys.key_blocks.find('MC_current')
        if ob.active_shape_key_index != index:
            return

        # bmesh gets removed when someone clicks on MC_current shape"
        try:
            cloth.obm.verts
        except:
            cloth.obm = get_bmesh(ob)

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
        same, faces = detect_changes(cloth.geometry, cloth.obm)
        if faces: # zero faces in mesh do nothing
            return
        if not same:
            # for pinning
            np.copyto(cloth.pin_arr, cloth.co)
            cloth.springs, cloth.v_fancy, cloth.e_fancy, cloth.flip = get_springs(cloth, cloth.obm)
            cloth.geometry = get_mesh_counts(ob, cloth.obm)
            # cloth.sew_springs = get_sew_springs() # build
            cloth.obm.verts.ensure_lookup_table()
            update_groups(cloth, cloth.obm)
            cloth.vel_start = get_co_edit(None, cloth.obm)


        # updating the mesh coords -----------------@@
        # detects user changes to the mesh like grabbing verts
        cloth.co = np.array([v.co for v in cloth.obm.verts])
        pause_selected = True
        if pause_selected: # create this property for the user (consider global settings for all meshes)
            cloth.selected = np.array([v.select for v in cloth.obm.verts])
        """======================================"""
        """======================================"""
        # FORCES FORCES FORCES FORCES FORCES

        # add some forces
        #cloth.co += grav
        spring_force(cloth)

        # FORCES FORCES FORCES FORCES FORCES
        """======================================"""
        """======================================"""

        # write vertex coordinates back to bmesh
        #for i in range(co.shape[0]):
            #cloth.obm.verts[i].co = co[i]
        #vco_list = [v for v in cloth.obm.verts]

        cloth.obm.verts.ensure_lookup_table()
        [obm_co(cloth.obm.verts, cloth.co, i) for i in range(cloth.co.shape[0])]
        # not sure if calling the function in list comp is faster
        #   than loop. Need to check this.

        bmesh.update_edit_mesh(ob.data)
        if False: # (there might be a much faster way by writing to a copy mesh then loading copy to obm)
            bmesh.ops.object_load_bmesh(bm, scene, object)
        # -----------------------------------------@@
        return


    if cloth.mode is None:
        cloth.mode = 1
        # switched out of edit mode

    # OBJECT MODE ====== :
    #cloth.co = get_co_shape(cloth.ob, "MC_current")


    """ =============== FORCES OBJECT MODE ================ """
    # FORCES FORCES FORCES FORCES

    #cloth.co += grav
    spring_force(cloth)

    # FORCES FORCES FORCES FORCES
    """ =============== FORCES OBJECT MODE ================ """

    # updating the mesh coords -----------------@@
    shape = cloth.co.shape
    cloth.co.shape = (shape[0] * 3,)
    ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set("co", cloth.co)
    cloth.co.shape = shape
    cloth.ob.data.update()



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

        if not detect_changes(cloth.geometry, cloth.obm)[0]:
            cloth.springs, cloth.v_fancy, cloth.e_fancy, cloth.flip = get_springs(cloth)

    # find all the collider objects in the scene and put them into a list
    colliders = [i for i in bpy.data.objects if i.MC_props.collider]
    for i in colliders:
        col = MC_data['colliders'][i['MC_collider_id']]
        col.ob = i
        # update for meshes in edit mode
        col.undo = True

    fun = ["your shoe laces", "something you will wonder about but never notice", "something bad because it loves you", "two of the three things you accomplished at work today", "knots in the threads that hold the fabric of the universe", "a poor financial decision you made", "changes to your online dating profile", "your math homework", "everything you've ever accomplished in life", "something you'll discover one year from today", "the surgery on your cat", "your taxes", "all the mistakes you made as a child", "the mess you made in the bathroom", "the updates to your playstation 3", "nothing! Modeling Cloth makes no mistakes!", "your last three thoughts and you'll never know what the"]
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
    update_cloth(type) # type 0 continuous, type 1 animated
    # ----------------------------

    MC_data['count'] += 1
    print('frame: ', MC_data['count'])

    # auto-kill
    auto_kill = True
    auto_kill = False
    if auto_kill:
        if MC_data['count'] == 20:
            print()
            print('--------------')
            print('died')
            return

    delay = bpy.context.scene.MC_props.delay# !! put this into a user property !!
    T(total_time, "Total time in handler")
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


# calback functions ---------------
def oops(self, context):
    # placeholder for reporting errors or other messages
    return

# calback functions ---------------
# object:

def cb_collider(self, context):
    """Set up object as collider"""

    ob = bpy.context.object
    if ob.type != "MESH":
        self['collider'] = False

        # Report Error
        msg = "Must be a mesh. Collisions with non-mesh objects can create black holes potentially destroying the universe."
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        return

    # The object is the key to the cloth instance in MC_data
    if self.collider:
        collider = create_collider()

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

    # set the recent object for keeping settings active when selecting empties
    ob = MC_data['recent_object']
    if ob is None:
        ob = bpy.context.object

    if ob.type != "MESH":
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
        reset_shapes(ob)
        index = ob.data.shape_keys.key_blocks.find('MC_current')
        ob.active_shape_key_index = index

#        # vertex groups
#        if 'MC_pin' not in ob.vertex_groups:
#            ob.vertex_groups.new(name='MC_pin')
#        pin = get_weights(ob, 'MC_pin', obm=None)
#        pin_arr = np.zeros(pin.shape[0]*3, dtype=np.float32)
#        pin_arr.shape = (pin.shape[0], 3)

        cloth = create_instance()
#        #cloth.pin = pin[:, nax]
#        #cloth.pin_arr = pin_arr
        #update_groups(cloth)


        # recent_object allows cloth object in ui
        #   when selecting empties such as for pinning.
        MC_data['recent_object'] = bpy.context.object

        # use an id prop so we can find the object after undo
        d_keys = [i for i in MC_data['cloths'].keys()]
        id_number = 0
        if len(d_keys) > 0:
            id_number = max(d_keys) + 1
            print("created new cloth id", id_number)

        MC_data['cloths'][id_number] = cloth
        ob['MC_cloth_id'] = id_number
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

    # updates groups when we toggle "Continuous"
    ob = bpy.context.object
    cloth = MC_data["cloths"][ob['MC_cloth_id']]
    if ob.data.is_editmode:
        cloth.co = np.array([v.co for v in cloth.obm.verts])
        update_groups(cloth, cloth.obm)
        return
    cloth.co = get_co_shape(ob, key='MC_current')
    update_groups(cloth, None)


# calback functions ----------------
# object:
def cb_animated(self, context):
    """Turn animated update on or off"""
    install_handler(continuous=False)

    # updates groups when we toggle "Animated"
    ob = bpy.context.object
    cloth = MC_data["cloths"][ob['MC_cloth_id']]
    if ob.data.is_editmode:
        cloth.co = np.array([v.co for v in cloth.obm.verts])
        update_groups(cloth, cloth.obm)
        return
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
    same = compare_geometry(ob, self.target, obm1=None, obm2=None, all=False)
    if not same:
        msg = "Vertex and Face counts must match. Sew edges don't have to match."
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        self.target = None
        cloth.target = None
        return

    # Ahh don't kill me I'm a valid target!
    print(self.target, "this should be target dot namey poo")
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

    collider:\
    bpy.props.BoolProperty(name="Collider", description="Cloth objects collide with this object", default=False, update=cb_collider)

    cloth:\
    bpy.props.BoolProperty(name="Cloth", description="Set this as a cloth object", default=False, update=cb_cloth)

    # handler props for each object
    continuous:\
    bpy.props.BoolProperty(name="Continuous", description="Update cloth continuously", default=False, update=cb_continuous)

    animated:\
    bpy.props.BoolProperty(name="Animated", description="Update cloth only when animation is running", default=False, update=cb_animated)

    target:\
    bpy.props.PointerProperty(type=bpy.types.Object, description="Use this object as the target for stretch and bend springs", update=cb_target)


    stretch:\
    bpy.props.FloatProperty(name="Stretch", description="Strength of the stretch springs", default=.2, min=0, max=1, soft_min= -2, soft_max=2)



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

        mode = ob.mode
        if ob.data.is_editmode:
            bpy.ops.object.mode_set(mode='OBJECT')

        reset_shapes(ob)
        bco = get_co_shape(ob, "Basis")
        cloth = MC_data['cloths'][ob['MC_cloth_id']]
        cloth.co = bco
        ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', bco.ravel())

        bpy.ops.object.mode_set(mode=mode)
        ob.data.update()
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

# ^                                                          ^ #
#                  END registered operators                    #
# ============================================================ #


# ============================================================ #
#                         draw code                            #
#                                                              #
class PANEL_PT_modelingCloth(bpy.types.Panel):
    """Modeling Cloth Panel"""
    bl_label = "Modeling Cloth Panel 2"
    bl_idname = "PANEL_PT_modelingCloth_2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        sc = bpy.context.scene
        layout = self.layout
        col = layout.column(align=True)
        col.prop(sc.MC_props, "kill_duplicator", text="kill_duplicator", icon='DUPLICATE')
        # use current mesh or most recent cloth object if current ob isn't mesh
        ob = bpy.context.object
        if ob is not None:
            mesh = ob.type == "MESH"
            recent = MC_data['recent_object']
            if recent is not None:
                if not mesh:
                    ob = recent
                    mesh = True
                else:
                    MC_data['recent_object'] = ob

            cloth = ob.MC_props.cloth
            if mesh:
                # if we select a new mesh object we want it to display

                col = layout.column(align=True)
                # display the name of the object if "cloth" is True
                #   so we know what object is the recent object
                recent_name = ''
                if ob.MC_props.cloth:
                    recent_name = ob.name
                col.prop(ob.MC_props, "cloth", text="Cloth " + recent_name, icon='MOD_CLOTH')
                col.prop(ob.MC_props, "collider", text="Collide", icon='MOD_PHYSICS')

                if cloth:
                    col.label(text='Update Mode')
                    col = layout.column(align=True)
                    #col.scale_y = 1
                    row = col.row()
                    row.scale_y = 2
                    row.prop(ob.MC_props, "animated", text="Animated", icon='PLAY')
                    row = col.row()
                    row.scale_y = 2
                    row.prop(ob.MC_props, "continuous", text="Continuous", icon='FILE_REFRESH')
                    row = col.row()
                    row.scale_y = 1
                    row.prop(sc.MC_props, "delay", text="Delay", icon='SORTTIME')
                    box = col.box()
                    #row = col.row()
                    box.scale_y = 2
                    box.operator('object.mc_reset_to_basis_shape', text="RESET", icon='RECOVER_LAST')
                    col = layout.column(align=True)
                    col.use_property_decorate = True
                    col.label(text='Target Object')
                    col.prop(ob.MC_props, "target", text="", icon='DUPLICATE')

                    col.label(text='Forces')
                    col.prop(ob.MC_props, "stretch", text="stretch", icon='PLAY')



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


    #print("running")
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
    PANEL_PT_modelingCloth,
    MCResetToBasisShape,
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


def unregister():
    # classes
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    # props
    del(bpy.types.Scene.MC_props)

    # clean dead versions of the undo handler
    handler_names = np.array([i.__name__ for i in bpy.app.handlers.undo_post])
    booly = [i == 'undo_frustration' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.undo_post[i])


if __name__ == '__main__':
    register()
