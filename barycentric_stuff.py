import bpy
import numpy as np
from numpy import newaxis as nax
import bmesh
import time

def cpoe(p, e1, e2):
    ev = e2 - e1
    pv = p - e1
    d = np.dot(ev, pv) / np.dot(ev,ev)
    cp = e1 + ev * d
    return cp, d


def inside_tri_wa(tris, points, norms=None):
    """use weighted average to check inside tri.
    Might be able to modify to work with more than
    three points for surface follow to more than three"""

    # drop points to planes
    origins = tris[:, 0]
    cross_vecs = origins[:, nax] - tris[:, 1:]
    if norms is None:    
        norms = np.cross(cross_vecs[:,0], cross_vecs[:, 1])
    pv = points - origins
    d = np.einsum('ij,ij->i', norms, pv) / np.einsum('ij,ij->i', norms, norms)    
    cpv = norms * d[:, nax]
    on_p = points - cpv

    # get weights
    weights = np.linalg.solve(np.swapaxes(tris, 1,2), on_p)
    inside = np.any(weights < 0, axis=1)


def inside_triangles(tris, points, check=True):
    
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
    
    weights = np.array([1 - (u+v), u, v, ])
    if not check:
        return weights.T
    
    check = np.all(weights > 0, axis=0)
    # check if bitwise is faster when using lots of tris
    if False:    
        check = (u > 0) & (v > 0) & (u + v < 1)

    return weights.T, check


def closest_point_mesh(obm, edit_obj, target):
    """Uses bmesh method to get CPM"""
    context = bpy.context
    scene = context.scene

    me = edit_obj.data
    mesh_objects = [target]
    bm = bmesh.new()

    smwi = target.matrix_world.inverted()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    vert = bm.verts[0]
    for vert in bm.verts:    
        v1 = edit_obj.matrix_world @ vert.co # global face median
        local_pos = smwi @ v1  # face cent in target local space

        (hit, loc, norm, face_index) = target.closest_point_on_mesh(local_pos)
        if hit:
            v2 = target.matrix_world @ loc
            bpy.ops.object.empty_add(location=v2)
            #print(target.name, (v2 - v1).length)
            #print(face_index)
    bm.clear()
        
if False:        
    ob = bpy.context.object
    target = bpy.data.objects['target']
    obm = bmesh.new()
    obm.from_mesh(ob.data)
    print(obm.verts)
    closest_point_mesh(obm, ob, target)
    
    
    
tri = bpy.data.objects['tri']
tri2 = bpy.data.objects['tri2']
tri3 = bpy.data.objects['t3']
tri4 = bpy.data.objects['t4']
e = bpy.data.objects['e']
e2 = bpy.data.objects['e2']


def wab(point, tri):
    print()
    print()
    co = np.array([v.co for v in tri.data.vertices])
    co2 = np.array([v.co for v in tri2.data.vertices])
    co3 = np.array([v.co for v in tri3.data.vertices])
    co4 = np.array([v.co for v in tri4.data.vertices])
    # it's going to be each vector * something...

    weightss = np.array([7, 1, 1])

    norm = weightss/np.sum(weightss)

    balance = co * norm[:, nax]
    bpy.data.objects['eee'].location = np.sum(balance, axis=0)
    eco = np.array(e.location)
    e2co = np.array(e2.location)


    tris = np.array([co, co2, co3, co4])
    points = np.array([e2co, e2co, eco, eco])
    
    #inside_tri_wa(tris2, points2, norms=None)
    weights, check = inside_triangles(tris, points)
    
    bpy.data.objects['s'].location = np.sum(tris[0] * weights[0][:, nax], axis=0)
    


def get_weights(ob, group):
    idx = ob.vertex_groups[group].index
    w = np.array([v.groups[idx].weight for v in ob.data.vertices])
    
    


def apply_weights(ob, points, weights, tris, normals):
    """
    ob: the blender object
    points: a bool array of the points we're moving
    weights: barycentric weigts
    tris: the faces we parented to
    normals: the distance from those faces
    """
    
    
    
    
    
    # divide the verts that need to follow into a vertex group
    #   This way we can pull from the blender object.
    
    
    
    
    
    
    
wab(e, tri)

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





so first I get a selection set by doing something like if i.select
