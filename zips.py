import bpy
import bmesh
import numpy as np
import json


def verts_in_group(ob, name='Group'):
    """Returns np array of indices for vertices in the group"""
    ob.update_from_editmode() # in case someone has assigned verts in editmode
    idx = ob.vertex_groups[name].index
    idxer = np.arange(len(ob.data.vertices))
    this = [[j.group for j in v.groups if j.group == idx] for v in ob.data.vertices]
    idxs = [i for i in idxer if len(this[i]) > 0]
    return np.array(idxs)


def total_curve_length(curve):
    """Put in a curve object. Get a single float."""
    p_count = len(curve.data.splines[0].points)
    c_co = np.zeros(p_count * 4, dtype=np.float32)
    curve.data.splines[0].points.foreach_get('co', c_co)
    c_co.shape = (p_count, 4)
    c_vco = c_co[:,:3]
    vecs = c_vco[1:] - c_vco[:-1]
    lengths = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
    return np.sum(lengths)


def copy_transforms(ob1, ob2):
    """Set the transforms of the first object
    to the second object."""

    re = ob1.rotation_euler
    rq = ob1.rotation_quaternion
    s = ob1.scale
    l = ob1.location

    ob2.rotation_euler = re
    ob2.rotation_quaternion = rq
    ob2.scale = s
    ob2.location = l
    return re, rq, s, l


def get_co(ob):
    """Returns Nx3 numpy array of vertex coords as float32"""
    v_count = len(ob.data.vertices)
    co = np.empty(v_count * 3, dtype=np.float32)
    ob.data.vertices.foreach_get('co', co)
    co.shape = (v_count, 3)
    return co

def co_to_pco(co):
    """Add extra value to co
    for use as point co"""
    pco = np.ones(co.shape[0]*4, dtype=np.float32)
    pco.shape = (co.shape[0],4)
    pco[:,:3] = co
    return pco


def get_sub_co(ob, idx):
    """Use list comp to generate a small set of coords.
    Works in edit or object mode."""
    if ob.data.is_editmode:
        obm = bmesh.from_edit_mesh(ob.data)
        obm.verts.ensure_lookup_table()
        co = np.array([obm.verts[i].co for i in idx], dtype=np.float32)
    else:
        co = np.array([ob.data.vertices[i].co for i in idx], dtype=np.float32)
    return co


def get_co_with_modifiers(ob, types=[], names=[], include_mesh=False, dg=None, att='co', proxy_only=False):
    """Get the coordinates of modifiers with
    specific modifiers turned on or off.
    List mods by type or name.
    This lets you turn off all mods of a type
    or just turn off by name.
    Can also be used to get vertex normals. set 'att' to 'normals'"""

    emo = False
    if ob.data.is_editmode:
        emo = True
        bpy.ops.object.mode_set(mode='OBJECT')

    debug = False
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

    # Can't evaluate depsgraph more than once or there are random errors
    return_dg = False
    if dg is None:
        return_dg = True
        dg = bpy.context.evaluated_depsgraph_get()
    proxy = ob.evaluated_get(dg)

    if emo:
        bpy.ops.object.mode_set(mode='EDIT')

    if proxy_only: # Sometimes you just want the dg object
        return proxy
    # get the coordinates with the current modifier state
    # can get normals or co
    if att == 'co':
        co = get_co(proxy)
    if att == 'normals':
        co = get_vertex_normals(proxy)

    for i, j in zip(mod_states, ob.modifiers):
        j.show_viewport = i

    if include_mesh:
        return co, proxy.data

    if return_dg:
        return co, dg

    return co


def delete_by_names(names=[]):
    """Deletes objects and meshes by name"""
    meshes = [bpy.data.objects[i].data.name for i in names]
    for i in bpy.data.objects:
        i.select_set(i.name in names)
    bpy.ops.object.delete()
    for i in meshes:
        bpy.data.meshes.remove(bpy.data.meshes[i])


def tilt_curves(idx, proxy, curve):
    """idx is the points on the garment
    that define the path for the zipper"""

    deletables = [] # dummy curve objects to be deleted.

    # set active object and select it
    bpy.context.view_layer.objects.active = curve
    for i in bpy.data.objects:
        i.select_set(i.name==curve.name)

    g_co = np.array([proxy.data.vertices[i].co for i in idx])

    p_count = len(curve.data.splines[0].points)
    c_co = np.zeros(p_count * 4, dtype=np.float32)
    curve.data.splines[0].points.foreach_get('co', c_co)
    c_co.shape = (p_count, 4)
    c_vco = c_co[:,:3]

    vecs = c_vco[1:] - c_vco[:-1]
    lengths = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
    # ----------------------------------------

    co = np.zeros(p_count * 3, dtype=np.float32)
    co.shape = (p_count, 3)
    co[1::,0] = np.cumsum(lengths)

    b = curve.shape_key_add(name='Basis')
    f = curve.shape_key_add(name='flat')
    f.data.foreach_set('co', co.ravel())
    f.value = 1

    # create duplicate
    bpy.ops.object.duplicate_move()

    bpy.ops.object.convert()
    co[:,2] = 0.1
    c2 = bpy.context.object
    c2.data.vertices.foreach_set('co', co.ravel())
    c2.data.update()
    c2.name = 'please kill me'
    deletables.append(c2.name)

    mod = c2.modifiers.new(name='C', type='CURVE')
    mod.object = curve
    f.value = 0
    nor_marker = bpy.context.object

    g_normals = np.array([proxy.data.vertices[i].normal for i in idx]) * -1
    nor_ob_co = get_co_with_modifiers(bpy.context.object)[0]
    tilt_nor = nor_ob_co - c_vco
    Utd = tilt_nor / np.sqrt(np.einsum('ij,ij->i', tilt_nor, tilt_nor))[:,None]
    angle = np.arccos(np.einsum('ij,ij->i', Utd, g_normals))
    vec_add_1 = np.append(vecs[0][None], vecs, axis=0)
    cross = np.cross(vec_add_1, Utd)
    signs = np.sign(np.einsum('ij,ij->i', cross, g_normals))
    curve.shape_key_remove(f)
    curve.shape_key_remove(b)

    curve.data.splines[0].points.foreach_set('tilt', angle * signs)
    bpy.data.objects[curve.name].parent = bpy.data.objects[proxy.name]
    delete_by_names(deletables)


def generate_curve(co, name='my_curve'):
    # create the Curve Datablock
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 2

    # map coords to spline
    polyline = curveData.splines.new('POLY')
    polyline.points.add(co.shape[0] -1)

    polyline.points.foreach_set('co', co.ravel())
    # create Object
    curveOB = bpy.data.objects.new(name, curveData)
    curveOB.data.splines.update()

    # attach to scene and validate context
    scn = bpy.context.scene
    bpy.context.collection.objects.link(curveOB)
    return curveOB


def oops(self, context):
    """placeholder for reporting errors or other messages"""
    return

    # example: Put the below code in the operator's execute function
    # Report Error
    msg = "Must be a mesh. Collisions with non-mesh objects can create black holes potentially destroying the universe."
    bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')


def auto_flip(co):
    """Checks the direction of the zipper
    assuming -z is down and reverses curve."""
    if co[-1][2] < co[0][2]:
        print("Flipped the zipper curve direction.")
        return True

    return False


def save_data(name='saved_data.py', var='some_variable', data={'key': [1,2,3]}):
    """Saves a dictionary as a variable in a python file
    as a blender internal text file. Can later import
    module and call all data as global variables.
    Overwrites if variable is already there"""
    if name not in bpy.data.texts:
        bpy.data.texts.new(name)

    data_text = bpy.data.texts[name]
    mod = data_text.as_module()
    atts =[item for item in dir(mod) if not item.startswith("__")]

    if var in atts:
        # copy existing and overwrite data at key
        data_text.clear()
        dicts = [getattr(mod, a) for a in atts]
        for i in range(len(dicts)):
            if atts[i] == var:
                dicts[i] = data

        # now rewrite from copy
        for name, di in zip(atts, dicts):
            # can also just add to the text
            data_text.cursor_set(-1, character=-1) # in case someone moves the cursor
            m = json.dumps(di, sort_keys=True, indent=2)

            data_text.write(name + ' = ' + m)

            data_text.cursor_set(-1, character=-1) # add the new line or we can't read it as a python module
            data_text.write('\n')
        return

    # can also just add to the text
    m = json.dumps(data, sort_keys=True, indent=2)

    data_text.cursor_set(-1, character=-1) # in case someone moves the cursor
    data_text.write(var + ' = ' + m)

    data_text.cursor_set(-1, character=-1) # add the new line or we can't read it as a python module
    data_text.write('\n')


def get_saved_data(name='saved_data.py'):
    saved = bpy.data.texts[name].as_module()
    print(saved.left_idx)


def get_order_of_selection(garment_Bobj):
    """Takes selection and looks for
    an order to generate a curve.
    Returns ordered idx from selection"""

    ob = garment_Bobj
    if ob.data.is_editmode:
        obm = bmesh.from_edit_mesh(ob.data)
        v_sel = np.array([v.select for v in obm.verts])
        edges = np.array([[e.verts[0].index, e.verts[1].index] for e in obm.edges if e.select])
        uni, counts = np.unique(edges, return_counts=True)

    else:
        v_sel = np.zeros(len(ob.data.vertices), dtype=np.bool)
        e_sel = np.zeros(len(ob.data.edges), dtype=np.bool)
        edges_1 = np.zeros(len(ob.data.edges)*2, dtype=np.int32)
        ob.data.edges.foreach_get('vertices', edges_1)
        edges_1.shape = (edges_1.shape[0]//2, 2)
        ob.data.edges.foreach_get('select', e_sel)
        edges = edges_1[e_sel]
        uni, counts = np.unique(edges, return_counts=True)

    # if we don't have clear start and end return None
    # returning None triggers the error message
    if counts.ravel().shape[0] < 2:
        return
    if np.max(counts.ravel()) > 2:
        return
    if np.sum(counts[counts==1]) != 2:
        return

    ends = uni[counts==1]
    e1_bool = np.any(edges==ends[0], axis=1)
    e2_bool = np.any(edges==ends[1], axis=1)
    e1 = edges[e1_bool]
    e2 = edges[e2_bool]

    next_vert = e1[e1 != ends[0]]
    ordered = [ends[0], next_vert[0]]

    mid_edges = edges[~(e1_bool + e2_bool)]

    for e in range(mid_edges.shape[0]):
        b = np.any(mid_edges==next_vert, axis=1)
        ed = mid_edges[b]
        next_vert = ed[ed!=next_vert]
        mid_edges = mid_edges[~b] # so the search shrinks
        ordered.append(next_vert[0])

    ordered.append(ends[1])

    return np.array(ordered)


def set_path(idx=None, key="my variable"):
    if idx is None:
        idx = get_order_of_selection(bpy.context.object)
    if idx is None:
        return
    return idx


def modifier_setup(ob, curve, bottom, top, name='zips_curve'):
    # array mod
    arr = ob.modifiers.new(name, type='ARRAY')
    arr.fit_type = 'FIT_CURVE'
    arr.curve = curve
    arr.start_cap = bottom
    arr.end_cap = top
    # curve mod
    cur = ob.modifiers.new(name, type='CURVE')
    cur.object = curve


def path_setup(name, path=None):
    """Run in execute function to setup each path"""
    ob = bpy.context.object
    proxy = get_co_with_modifiers(ob, proxy_only=True)
    if path is None:
        path = get_order_of_selection(bpy.context.object)

    if path is None:
        msg = "Select a continuous path of verts with a start and end."
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        return {'FINISHED'}

    co = get_sub_co(proxy, path)
    flip = auto_flip(co)
    #flip = True
    if flip:
        path[:] = path[::-1]
        co[:] = co[::-1]

    pco = co_to_pco(co)

    curve = generate_curve(pco, name=name)
    #ob.zips_props.left_curve = curve # set property for later
    tilt_curves(path, proxy, curve)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)

    # for saving the path to file (might not need path here...)
    di = {'path': path.tolist(), 'curve_name': curve.name}
    save_data(name=ob.name + '_zips_data.py', var=name, data=di)
    return {'FINISHED'}


class ZipsSetLeftPath(bpy.types.Operator):
    """Save the left zipper path"""
    bl_idname = "object.zips_set_left_path"
    bl_label = "Define Left Path"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return path_setup('left_path')


class ZipsSetRightPath(bpy.types.Operator):
    """Save the right zipper path"""
    bl_idname = "object.zips_set_right_path"
    bl_label = "Define Right Path"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return path_setup('right_path')


class ZipsSetMiddlePath(bpy.types.Operator):
    """Save the middle zipper path"""
    bl_idname = "object.zips_set_middle_path"
    bl_label = "Define Middle Path"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return path_setup('middle_path')


class ZipsApplyToGarment(bpy.types.Operator):
    """Apply The Zipper To The Garment"""
    bl_idname = "object.apply_zipper_to_garment"
    bl_label = "Zips Apply Zipper To Garment"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        # run all the functions here
        # !!! make this overwrite the data file !!!
        return {'FINISHED'}


class ZipsWriteSettingsToFile(bpy.types.Operator):
    """Save Settings To File"""
    bl_idname = "object.zips_write_settings_to_file"
    bl_label = "Zips Write Settings To File"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        """Save the properties as a default"""
        # run all the functions here

        return {'FINISHED'}


class ZipsApplySettingsFromFile(bpy.types.Operator):
    """Apply Settings From File"""
    bl_idname = "object.zips_apply_settings_from_file"
    bl_label = "Zips Apply Settings From File"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        # run all the functions here

        return {'FINISHED'}


def armature_setup(zob, ar, curve):
    G = bpy.context.object
    # add the curve constraint to the bone
    if 'zips_path' not in ar.constraints:
        con = ar.constraints.new('FOLLOW_PATH')
        con.name = 'zips_path'

    con = ar.constraints['zips_path']
    con.use_curve_follow = True
    con.forward_axis = 'FORWARD_X'
    con.use_fixed_location = True

    bpy.context.view_layer.objects.active = ar
    #ar.select_set(True)
    override = {'constraint':ar.constraints['zips_path']}
    bpy.ops.constraint.followpath_path_animate(override, constraint='zips_path')
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update
    bpy.context.view_layer.objects.active = G
    # have to animate path
    con.target = curve # !!! if you do this
    #   it breaks the other objects on the path

    length = total_curve_length(curve)
    bottom_offset = ar.pose.bones['offset'].head.x
    co = get_co(zob)
    top_offset = np.max(co[::3]) - ar.pose.bones['base'].head.x

    di = {'total_length': str(length),
    'bottom_offset': str(bottom_offset),
    'top_offset': str(top_offset)}
    ob=bpy.context.object
    save_data(name=ob.name + '_zips_data.py', var='offset_data', data=di)

    # calculate offset
    prop_offset = ob.zips_props.zipper_pull_offset
    con_length = 1 - ((bottom_offset/length) + (top_offset/length))
    prop_offset + (bottom_offset/length)
    con.offset_factor = (prop_offset * con_length) + (bottom_offset/length)


def object_callback_setup(zob, order, var, ar=False):
    # for some reason this is running twice when an object property is None
    gob = bpy.context.object
    # get text module
    name = gob.name + '_zips_data.py'
    if name not in bpy.data.texts:
        return None, None

    module = bpy.data.texts[name].as_module()
    atts = [item for item in dir(module) if not item.startswith("__")]

    if zob is None:
        if var in atts:
            zob = bpy.data.objects[getattr(module, var)['ob']]
            if 'zips_array' in zob.modifiers:
                zob.modifiers.remove(zob.modifiers['zips_array'])
            if 'zips_curve' in zob.modifiers:
                zob.modifiers.remove(zob.modifiers['zips_curve'])
            if 'zips_displace' in zob.modifiers:
                zob.modifiers.remove(zob.modifiers['zips_displace'])
            if zob.parent is not None:
                if zob.parent.type == 'ARMATURE':
                    ar = zob.parent
                    try:
                        cons = ar.constraints
                        cons.remove(cons['zips_path'])
                    except:
                        print('tried to remove constraint. Already removed.')
                if zob.parent.type != 'ARMATURE':
                    zob.parent = None
        return None, None

    # save the object name so the prop can find it on remove
    di = {'ob': zob.name}
    save_data(name=name, var=var, data=di)

    curve = None

    if order[0] in atts:
        curve = bpy.data.objects[getattr(module, order[0])['curve_name']]

    if order[1] in atts:
        curve = bpy.data.objects[getattr(module, order[1])['curve_name']]

    if gob.zips_props.flip_lr:
        if order[2] in atts:
            curve = bpy.data.objects[getattr(module, order[2])['curve_name']]

    if curve is None:
        msg = "No path found for this object"
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        print('!! no suitable path !!')
        return None, None

    if ar:
        armature_setup(zob, ar, curve)

        return None, None

    if 'zips_displace' in zob.modifiers:
        displace = zob.modifiers['zips_displace']
    else:
        displace = zob.modifiers.new('zips_displace', type='DISPLACE')
    displace.direction = 'X'
    displace.mid_level = 0.0
    displace.strength = 0.0

    # curve mod
    if 'zips_curve' in zob.modifiers:
        curve_mod = zob.modifiers['zips_curve']
    else:
        curve_mod = zob.modifiers.new('zips_curve', type='CURVE')

    curve_mod.object = curve
    bpy.data.objects[zob.name].parent = bpy.data.objects[curve.name]

    return curve, displace


# Objects: (Left and Right from the perspective of the wearer of the garment)
def cb_garment(self, context):
    """This object is the garment"""
    print(self.name, "this is the garment object")


def cb_zipper_pull(self, context):
    """Set this object as the zipper pull"""
    zob = self.zipper_pull
    order = ['middle_path', 'right_path', 'left_path']
    var = 'zipper_pull'

    ar = False # switches to armature if we are using an armature
    if zob is not None:
        if zob.parent is not None:
            if zob.parent.type == 'ARMATURE':
                ar = zob.parent
                bones = [b.name for b in ar.pose.bones]
                names = ['tab', 'base', 'offset']
                if not np.all(np.in1d(names, bones)):
                    # make sure we have all the bones
                    ar = False

    curve, displace = object_callback_setup(zob, order, var, ar)
    if ar:
        return

    # if we're not using an armature
    # add displace for pull offset
    if curve is not None:
        # so the zipper pull won't stick past the end
        co = np.zeros(len(zob.data.vertices) * 3)
        zob.data.vertices.foreach_get('co', co)
        x_dif = np.max(co[::3])
        length = total_curve_length(curve)
        offset = (self.zipper_pull_offset * length) - x_dif
        displace.direction = 'X'
        displace.strength = offset


def cb_right_top(self, context):
    """This is the top part of the
    zipper on the right"""
    zob = self.right_top
    order = ['middle_path', 'right_path', 'left_path']
    var = 'right_top'

    curve, displace = object_callback_setup(zob, order, var)

    # add displace for pull offset
    if curve is not None:
        # so the zipper pull won't stick past the end
        co = np.zeros(len(zob.data.vertices) * 3)
        zob.data.vertices.foreach_get('co', co)
        x_dif = np.max(co[::3])
        length = total_curve_length(curve)
        offset = length - x_dif
        displace.direction = 'X'
        displace.strength = offset


def cb_right_bottom(self, context):
    """This is the bottom part of the
    zipper on the right"""
    zob = self.right_bottom
    order = ['middle_path', 'right_path', 'left_path']
    var = 'right_bottom'
    object_callback_setup(zob, order, var)


def cb_left_top(self, context):
    """This is the top part of the
    zipper on the left"""
    zob = self.left_top
    order = ['middle_path', 'left_path', 'right_path']
    var = 'left_top'

    curve, displace = object_callback_setup(zob, order, var)

    # add displace for pull offset
    if curve is not None:
        # so the zipper pull won't stick past the end
        co = np.zeros(len(zob.data.vertices) * 3)
        zob.data.vertices.foreach_get('co', co)
        x_dif = np.max(co[::3])
        length = total_curve_length(curve)
        offset = length - x_dif
        displace.direction = 'X'
        displace.strength = offset


def cb_left_bottom(self, context):
    """This is the bottom part of
    the zipper on the left"""
    zob = self.left_bottom
    order = ['middle_path', 'left_path', 'right_path']
    var = 'left_bottom'
    curve, displace = object_callback_setup(zob, order, var)


def cb_left_tooth(self, context):
    """Zipper tooth Object to be
    repeated on left side"""
    zob = self.left_tooth
    order = ['middle_path', 'left_path', 'right_path']
    var = 'left_tooth'

    if zob is not None:
        if 'zips_arr' not in zob.modifiers:
            zob.modifiers.new('zips_array', type='ARRAY')
        arr = zob.modifiers['zips_array']
    curve, displace = object_callback_setup(zob, order, var)
    # set the count manually
    if curve is None:
        return

    co = np.zeros(len(zob.data.vertices) * 3)
    zob.data.vertices.foreach_get('co', co)
    x_min = np.min(co[::3])
    x_max = np.max(co[::3])
    x_dist = x_max - x_min
    length = total_curve_length(curve)

    # so the zipper teeth stop before the top object
    top = 0
    if self.left_top is not None:
        lt = self.left_top
        ltco = np.zeros(len(lt.data.vertices) * 3, dtype=np.float32)
        lt.data.vertices.foreach_get('co', ltco)
        tx_min = np.min(ltco[::3])
        tx_max = np.max(ltco[::3])
        top = tx_max - tx_min

    arr_count = (length - (x_min + top)) // x_dist
    arr.count = arr_count


def cb_right_tooth(self, context):
    """Zipper tooth Object to be
    repeated on right side"""
    zob = self.right_tooth
    order = ['middle_path', 'right_path', 'left_path']
    var = 'right_tooth'

    if zob is not None:
        if 'zips_arr' not in zob.modifiers:
            zob.modifiers.new('zips_array', type='ARRAY')
        arr = zob.modifiers['zips_array']
    curve, displace = object_callback_setup(zob, order, var)
    # set the count manually

    if curve is None:
        return
    co = np.zeros(len(zob.data.vertices) * 3)
    zob.data.vertices.foreach_get('co', co)
    x_min = np.min(co[::3])
    x_max = np.max(co[::3])
    x_dist = x_max - x_min
    length = total_curve_length(curve)

    # so the zipper teeth stop before the top object
    top = 0
    if self.right_top is not None:
        rt = self.right_top
        rtco = np.zeros(len(rt.data.vertices) * 3, dtype=np.float32)
        rt.data.vertices.foreach_get('co', rtco)
        tx_min = np.min(rtco[::3])
        tx_max = np.max(rtco[::3])
        top = tx_max - tx_min

    arr_count = (length - (x_min + top)) // x_dist
    arr.count = arr_count


def auto_rotate(ob):
    # !!! only runs if there is no armature !!!
    """Rotate the tab so it hangs down.
    Will not rotate past zipper pull body"""
    zob = ob.zips_props.zipper_pull

    if zob is None:
        return
    ar = zob.parent
    if ar is None:
        return
    if ar.type != 'ARMATURE':
        return
    if 'tab' not in ar.pose.bones:
        return

    tab = ar.pose.bones['tab']
    tab.rotation_mode = 'XYZ'
    v_count = len(zob.data.vertices)
    zco = np.zeros(v_count * 3, dtype=np.float32)
    zob.data.vertices.foreach_get('co', zco)

    # get an edge that points close to x axis
    ed = np.empty(len(zob.data.edges) * 2, dtype=np.int32)
    zob.data.edges.foreach_get('vertices', ed)
    vecs = zco[ed[1::2]] - zco[ed[::2]]

    low_vert = np.argmin(zco[::3])
    high_vert = np.argmax(zco[::3])

    zco.shape = (v_count, 3)
    tab.rotation_euler.z = 0
    current = 0

    dco, dg = get_co_with_modifiers(zob)
    rots = [dco[low_vert][2]]
    angles = [0]
    for i in range(18):
        current -= 10 * (np.pi/180)
        angles.append(current)
        tab.rotation_euler.z = current
        zob.data.update()
        pco = get_co_with_modifiers(zob, dg=dg)
        proxy = get_co_with_modifiers(zob, dg=dg, proxy_only=True)
        rots.append(pco[low_vert][2])

    low_angle = np.argmin(rots)
    tab.rotation_euler.z = angles[low_angle]

    # prevent from rotating tab into zipper body
    if tab.rotation_euler.z < np.pi:
        tab.rotation_euler.z = np.pi
    if tab.rotation_euler.z > 0:
        tab.rotation_euler.z = 0


# Settings:
def cb_flip_lr(self, context):
    """Flip the zipper pull to the other side"""
    print('flipped the zipper pull to the other side')


def cb_flip_inside_out(self, context):
    """For flipping the zipper inside out"""
    print('flipped the zipper inside out')


def cb_reverse_left_direction(self, context):
    """Reverse direction on the left side"""
    print('Reversed left side direction')


def cb_reverse_right_direction(self, context):
    """Reverse direction on the right side"""
    print('Reversed right side direction')


def cb_zipper_pull_offset(self, context):
    """Slide the zipper pull up and down"""
    if self.zipper_pull is None:
        return
    zob = self.zipper_pull
    ar = zob.parent
    if ar is not None:
        if ar.type == 'ARMATURE':
            ob=bpy.context.object
            module = ob.name + '_zips_data.py'
            data = bpy.data.texts[module].as_module().offset_data

            bottom_offset = float(data['bottom_offset'])
            top_offset = float(data['top_offset'])
            length = float(data['total_length'])

            # calculate offset
            prop_offset = ob.zips_props.zipper_pull_offset
            con_length = 1 - ((bottom_offset/length) + (top_offset/length))
            prop_offset + (bottom_offset/length)

            con = ar.constraints['zips_path']
            con.offset_factor = (prop_offset * con_length) + (bottom_offset/length)
            dg = context.evaluated_depsgraph_get()

            # auto rotate
            tab = ar.pose.bones['tab']
            tab.rotation_mode = 'XYZ'
            tab.rotation_euler.z = 0
            dg.update()
            h_co = ar.matrix_world @ tab.head.xyz
            t_co = ar.matrix_world @ tab.tail.xyz

            down = np.array([0,0,-1], dtype=np.float32)
            # get base vec for sign so it doesn't rotate backwards when pointing up
            base = ar.pose.bones['base']
            b_co = ar.matrix_world @ base.head.xyz
            bt_co = ar.matrix_world @ base.tail.xyz
            b_vec = np.array(bt_co - b_co)
            stop = (b_vec @ down) <= 0.0

            vec = np.array(t_co - h_co)
            U_vec = vec / np.sqrt(vec @ vec)
            dot = U_vec @ down
            rot = np.arccos(dot)
            if not stop:
                tab.rotation_euler.z = -rot

            # prevent from rotating tab into zipper body
            else:
                offset_bone = ar.pose.bones['offset']
                ob_co = ar.matrix_world @ offset_bone.head.xyz
                orient_vec = np.array(ob_co - b_co)
                orient = orient_vec @ down
                if orient > 0:
                    tab.rotation_euler.z = np.pi
                else:
                    tab.rotation_euler.z = 0
            return

    # if there is no armature for the zipper!!!
    auto_rotate(bpy.context.object)

    # so the zipper pull won't stick past the end
    co = np.zeros(len(zob.data.vertices) * 3)
    zob.data.vertices.foreach_get('co', co)
    x_dif = np.max(co[::3])
    if 'zips_curve' in zob.modifiers:
        if 'zips_displace' in zob.modifiers:
            displace = zob.modifiers['zips_displace']
            curve = zob.modifiers['zips_curve'].object
            length = total_curve_length(curve) - x_dif
            offset = self.zipper_pull_offset * length
            displace.strength = offset


def cb_zipper_tab(self, context):
    zob = self.zipper_tab
    if zob is None:
        return


def cb_rotate_tab(self, context):
    zob = self.zipper_tab
    if zob is None:
        return
    # !!! can't rotate into zipper body!!!


def cb_auto_rotate_tab(self, context):
    zob = self.zipper_tab
    if zob is None:
        return


def cb_left_side_offset(self, context):
    """Offset the left side of the zipper
    left or right"""
    print("Offset the left side left or right")


def cb_right_side_offset(self, context):
    """Offset the right side of the zipper
    left or right"""
    print("Offset the right side left or right")


def cb_left_side_normal_offset(self, context):
    """Offset the left side of the zipper along normals"""
    print("Offset the left side normals")


def cb_right_side_normal_offset(self, context):
    """Offset the right side of the zipper along normals"""
    print("Offset the right side normals")


def cb_left_side_rotate(self, context):
    """Rotate the tilt of the left side"""
    ob_props = [self.left_top, self.left_bottom, self.left_tooth]
    if self.flip_lr:
        ob_props.append(self.zipper_pull)
    obs = [ob for ob in ob_props if ob is not None]
    for ob in obs:
        ob.rotation_euler.x = self.left_side_rotate


def cb_right_side_rotate(self, context):
    """Rotate the tilt of the right side"""
    ob_props = [self.right_top, self.right_bottom, self.right_tooth]
    if not self.flip_lr:
        ob_props.append(self.zipper_pull)
    obs = [ob for ob in ob_props if ob is not None]
    for ob in obs:
        ob.rotation_euler.x = self.right_side_rotate


class ZipsPropsObject(bpy.types.PropertyGroup):

    # The Garment Object ----------------------- possibly unused
    garment:\
    bpy.props.BoolProperty(name="Garment",
    description="The garment object",
    default=False, update=cb_garment)

    # Objects ----------------------------------
    zipper_pull:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="The object for the zipper pull",
    update=cb_zipper_pull)

    #zipper_tab:\
    #bpy.props.PointerProperty(type=bpy.types.Object,
    #description="The object for the tab on the zipper pull",
    #update=cb_zipper_tab)

    right_top:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="The object for the zipper top on the right",
    update=cb_right_top)

    right_bottom:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="The object for the zipper bottom on the right",
    update=cb_right_bottom)

    right_tooth:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="The right repeated tooth object",
    update=cb_right_tooth)

    left_top:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="The object for the zipper top on the left",
    update=cb_left_top)

    left_bottom:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="The object for the zipper bottom on the left",
    update=cb_left_bottom)

    left_tooth:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="The left repeated tooth object",
    update=cb_left_tooth)

    # Generated Curves (invisible to ui)--------
    left_curve:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="Curve on Left Side")

    middle_curve:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="Curve in the Middle")

    right_curve:\
    bpy.props.PointerProperty(type=bpy.types.Object,
    description="Curve on Right Side")

    # Settings ---------------------------------
    flip_lr:\
    bpy.props.BoolProperty(name="Flip L/R",
    description="flipped the zipper pull to the other side",
    default=False, update=cb_flip_lr) # Some garments have the zipper pull on the left...

    flip_inside_out:\
    bpy.props.BoolProperty(name="Flip Inside Out",
    description="Flipped the zipper inside out",
    default=False, update=cb_flip_inside_out)

    reverse_left_direction:\
    bpy.props.BoolProperty(name="Reverse Left Direction",
    description="Reverse the curve direction on the left side",
    default=False, update=cb_reverse_left_direction)

    reverse_right_direction:\
    bpy.props.BoolProperty(name="Reverse Right Direction",
    description="Reverse the curve direction on the right side",
    default=False, update=cb_reverse_right_direction)

    zipper_pull_offset:\
    bpy.props.FloatProperty(name="Zipper Pull Offset",
    description="Move the zipper pull along the curve",
    default=0.0, precision=3, soft_min=0.0, soft_max=1.0, update=cb_zipper_pull_offset)

    rotate_tab:\
    bpy.props.FloatProperty(name="Rotate Tab",
    description="Rotate the tab on the zipper pull",
    default=0.0, precision=3, update=cb_rotate_tab)

    auto_rotate_tab:\
    bpy.props.BoolProperty(name="Auto-Rotate Tab",
    description="Rotate tab to point down",
    default=True, update=cb_auto_rotate_tab)

    left_side_offset:\
    bpy.props.FloatProperty(name="Left Side Offset",
    description="Offset the left side of the zipper left and right",
    default=0.0, precision=3, update=cb_left_side_offset)

    right_side_offset:\
    bpy.props.FloatProperty(name="Right Side Offset",
    description="Offset the right side of the zipper left and right",
    default=0.0, precision=3, update=cb_right_side_offset)

    left_side_normal_offset:\
    bpy.props.FloatProperty(name="Left Side Normal Offset",
    description="Offset the left side of the zipper along normals",
    default=0.0, precision=3, update=cb_left_side_normal_offset)

    right_side_normal_offset:\
    bpy.props.FloatProperty(name="Right Side Normal Offset",
    description="Offset the right side of the zipper along normals",
    default=0.0, precision=3, update=cb_right_side_normal_offset)

    left_side_rotate:\
    bpy.props.FloatProperty(name="Left Side Rotate",
    description="Rotate the tilt of the left curve",
    default=0.0, precision=3, update=cb_left_side_rotate)

    right_side_rotate:\
    bpy.props.FloatProperty(name="Right Side Rotate",
    description="Rotate the tilt of the right curve",
    default=0.0, precision=3, update=cb_right_side_rotate)


# Ui for debugging
class PANEL_PT_zipsPanel(bpy.types.Panel):
    """Zips Panel"""
    bl_label = "Zips Panel"
    bl_idname = "PANEL_PT_zips_tool_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    @classmethod
    def poll(cls, context):
        ob = context.object
        if ob is None:
            return False
        if ob.type=='MESH':
            return True

    def __init__(self):
        # strings for objects ui
        self.ob_props = [
        'zipper_pull',
        #'zipper_tab',
        'rotate_tab',
        'auto_rotate_tab',
        'right_top',
        'right_bottom',
        'right_tooth',
        'left_top',
        'left_bottom',
        'left_tooth',
        ]

        # text for objects ui
        self.ob_labels = [
        'Zipper Pull',
        #'Zipper Tab',
        'Rotate Tab',
        'Auto-Rotate Tab',
        'Right Top',
        'Right Bottom',
        'Right Tooth',
        'Left Top',
        'Left Bottom',
        'Left Tooth',
        ]

        # strings for settings ui
        self.settings = [
        'flip_lr',
        'flip_inside_out',
        'reverse_left_direction',
        'reverse_right_direction',
        'zipper_pull_offset',
        'left_side_offset',
        'right_side_offset',
        'left_side_normal_offset',
        'right_side_normal_offset',
        'left_side_rotate',
        'right_side_rotate'
        ]

        # divide bool and float props because the ui draws them different
        self.setting_bool_props = self.settings[:4]
        self.setting_props = self.settings[4:]
        self.setting_labels = [
        'Flip L/R',
        'Flip Inside Out',
        'Reverse Left Direction',
        'Reverse Right Direction',
        'Zipper Pull Offset',
        'Left Side Offset',
        'Right Side Offset',
        'Left Side In/Out',
        'Right Side In/Out',
        'Left Side Rotate',
        'Right Side Rotate'
        ]

        self.setting_bool_labels = self.setting_labels[:4]
        self.setting_prop_labels = self.setting_labels[4:]



class PANEL_PT_zipsPanelOperators(PANEL_PT_zipsPanel, bpy.types.Panel):
    """Zips Panel Operators"""
    bl_label = "Zipper Operators"
    bl_idname = "PANEL_PT_zips_panel_operators"

    def draw(self, context):
        ob = bpy.context.object
        module = ob.name + '_zips_data.py'
        vars=[]
        if module in bpy.data.texts:
            data = bpy.data.texts[module].as_module()
            vars = dir(data)
        layout = self.layout

        col = layout.column(align=True)
        col.scale_y = 2
        col.alert = False
        text="Left Path From Selection"
        if 'left_path' in vars:
            col.alert = True
            text="Overwrite Left"
        col.operator('object.zips_set_left_path', text=text, icon='MOUSE_LMB')

        col = layout.column(align=True)
        col.scale_y = 2
        col.alert = False
        text="Middle Path From Selection"
        if 'middle_path' in vars:
            col.alert = True
            text="Overwrite Middle"
        col.operator('object.zips_set_middle_path', text=text, icon='MOUSE_MMB')

        col = layout.column(align=True)
        col.scale_y = 2
        col.alert = False
        text="Right Path From Selection"
        if 'right_path' in vars:
            col.alert = True
            text="Overwrite Right"
        col.operator('object.zips_set_right_path', text=text, icon='MOUSE_RMB')

        col = layout.column(align=True)
        col.scale_y = 2
        col.operator('object.apply_zipper_to_garment', text="Apply Zipper", icon='RECOVER_LAST')
        col.operator('object.zips_write_settings_to_file', text="Write to File", icon='RECOVER_LAST')
        col.operator('object.zips_apply_settings_from_file', text="Apply From File", icon='RECOVER_LAST')


class PANEL_PT_zipsPanelObjects(PANEL_PT_zipsPanel, bpy.types.Panel):
    """Zips Panel Objects"""
    bl_label = "Zipper Objects"
    bl_idname = "PANEL_PT_zips_panel_objects"

    def draw(self, context):
        ob = bpy.context.object
        layout = self.layout
        col = layout.column(align=True)

        for p, t in zip(self.ob_props, self.ob_labels):
            col.label(text=t)
            col.prop(ob.zips_props, p, text='')


class PANEL_PT_zipsPanelSettings(PANEL_PT_zipsPanel, bpy.types.Panel):
    """Zips Panel Settings"""
    bl_label = "Zipper Settings"
    bl_idname = "PANEL_PT_zips_panel_settings"

    def draw(self, context):
        ob = bpy.context.object
        layout = self.layout
        col = layout.column(align=True)

        for p,t in zip(self.setting_bool_props, self.setting_bool_labels):
            col.prop(ob.zips_props, p, text=t)

        for p, t in zip(self.setting_props, self.setting_prop_labels):
            col.label(text=t)
            col.prop(ob.zips_props, p, text='')


# register functions
classes = (
    ZipsApplyToGarment,
    ZipsPropsObject,
    ZipsSetLeftPath,
    ZipsSetMiddlePath,
    ZipsSetRightPath,
    PANEL_PT_zipsPanelOperators,
    PANEL_PT_zipsPanelObjects,
    PANEL_PT_zipsPanelSettings,
    ZipsWriteSettingsToFile,
    ZipsApplySettingsFromFile,
)


def register():
    # classes
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    # props
    bpy.types.Object.zips_props = bpy.props.PointerProperty(type=ZipsPropsObject)


def unregister():
    # classes
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    # props
    del(bpy.types.Object.zips_props)


#if __name__ == '__main__':
    #register()

# -------------------------
def place_zipper_on_garment(garment, left_path=None, right_path=None, zipper_pull_normal=1):
    """garment is the garment blender obj.
    left path and right path is a list of vert
    indices for each side of the zipper"""
    register()

    # set garment as active object
    bpy.context.view_layer.objects.active = garment
    for i in bpy.data.objects:
        i.select_set(i.name==garment.name)

    # !!! unfinished !!! In case the zipper paths are not
    if (left_path is None) | (right_path is None):
        lg = 'P_L ZIP'
        rg = 'P_R ZIP'

        lidx = verts_in_group(garment, name=lg)
        ridx = verts_in_group(garment, name=rg)
        key = garment.data.shape_keys.key_blocks['pre_wrap']
        lco = np.array([key.data[i].co for i in lidx], dtype=np.float32)

    path_setup('left_path', path=np.array(left_path))
    path_setup('right_path', path=np.array(right_path))

    # check and assign each property.
    # rename objects since files can have multiple zippers
    self = garment.zips_props
    obs = bpy.data.objects
    if 'zipper_top_stop_L' in obs:
        ob = obs['zipper_top_stop_L']
        self.left_top = ob
        ob.name = ob.name + '000'
        
    if 'zipper_top_stop_R' in obs:
        ob = obs['zipper_top_stop_R']
        self.right_top = ob
        ob.name = ob.name + '000'

    if 'zipper_retaining_box' in obs:
        ob = obs['zipper_retaining_box']
        self.right_bottom = ob
        ob.name = ob.name + '000'

    if 'zipper_bottom_stop' in obs:
        ob = obs['zipper_bottom_stop']
        self.right_bottom = ob
        ob.name = ob.name + '000'

    if 'zipper_insert_pin' in obs:
        ob = obs['zipper_insert_pin']
        self.left_bottom = ob
        ob.name = ob.name + '000'

    if 'zipper_tooth_L' in obs:
        ob = obs['zipper_tooth_L']
        self.left_tooth = ob
        ob.name = ob.name + '000'

    if 'zipper_tooth_R' in obs:
        ob = obs['zipper_tooth_R']
        self.right_tooth = ob
        ob.name = ob.name + '000'

    if 'zipper_slider_body' in obs:
        ob = obs['zipper_slider_body']
        self.zipper_pull = ob
        ob.name = ob.name + '000'

    self.zipper_pull_offset = zipper_pull_normal
    return

# -------------------------
def test():
    garment = bpy.data.objects['garment']

    # Run append function to get blender objects

    # test module !!! Just using it to get the points
    data = bpy.data.texts['G_zips_data.py'].as_module()
    left_path = data.left_path['path']
    right_path = data.right_path['path']

    place_zipper_on_garment(garment, left_path=left_path, right_path=right_path, zipper_pull_normal=0.5)

test()
