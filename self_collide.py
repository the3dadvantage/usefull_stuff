try:
    import bpy
    import bmesh
except ImportEroor:
    pass

import numpy as np
import time
import json


def timer(t, name='name'):
    ti = bpy.context.scene.timers
    if name not in ti:
        ti[name] = 0.0
    ti[name] += t


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


def bmesh_proxy(ob):
    """Get a bmesh contating modifier effects"""
    dg = bpy.context.evaluated_depsgraph_get()
    prox = ob.evaluated_get(dg)
    proxy = prox.to_mesh()
    obm = bmesh.new()
    obm.from_mesh(proxy)
    return obm
    

def get_co_key(ob, key):
    k = ob.data.shape_keys.key_blocks[key]
    co = np.empty((len(ob.data.vertices), 3), dtype=np.float32)
    k.data.foreach_get('co', co.ravel())
    return co
    

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


def get_edges(ob):
    ed = np.empty((len(ob.data.edges), 2), dtype=np.int32)
    ob.data.edges.foreach_get('vertices', ed.ravel())
    return ed


def get_faces(ob):
    """Only works on triangle mesh."""
    fa = np.empty((len(ob.data.polygons), 3), dtype=np.int32)
    ob.data.polygons.foreach_get('vertices', fa.ravel())
    return fa
    

def generate_bounds(minc, maxc, margin):
    """from a min corner and a max corner
    generate the min and max corner of 8 boxes"""

    diag = (maxc - minc) / 2
    mid = minc + diag
    mins = np.zeros((8,3), dtype=np.float32) 
    maxs = np.zeros((8,3), dtype=np.float32) 

    # blf
    mins[0] = minc
    maxs[0] = mid
    # brf
    mins[1] = minc
    mins[1][0] += diag[0]
    maxs[1] = mid
    maxs[1][0] += diag[0]
    # blb
    mins[2] = minc
    mins[2][1] += diag[1]
    maxs[2] = mid
    maxs[2][1] += diag[1]
    # brb
    mins[3] = mins[2]
    mins[3][0] += diag[0]
    maxs[3] = maxs[2]
    maxs[3][0] += diag[0]
    # tlf
    mins[4] = mins[0]
    mins[4][2] += diag[2]
    maxs[4] = maxs[0]
    maxs[4][2] += diag[2]
    # trf
    mins[5] = mins[1]
    mins[5][2] += diag[2]
    maxs[5] = maxs[1]
    maxs[5][2] += diag[2]
    # tlb
    mins[6] = mins[2]
    mins[6][2] += diag[2]
    maxs[6] = maxs[2]
    maxs[6][2] += diag[2]
    # trb
    mins[7] = mins[3]
    mins[7][2] += diag[2]
    maxs[7] = maxs[3]
    maxs[7][2] += diag[2]
    
    return mid, [mins, maxs]


# universal ---------------------
def octree_et(sc, margin, idx=None, eidx=None, bounds=None):
    """Adaptive octree. Good for finding doubles or broad
    phase collision culling. et does edges and tris.
    Also groups edges in boxes.""" # first box is based on bounds so first box could be any shape rectangle

    T = time.time()
    margin = 0.0 # might be faster than >=, <=
    
    co = sc.co

    if bounds is None:
        b_min = np.min(co, axis=0)
        b_max = np.max(co, axis=0)
    else:
        b_min, b_max = bounds[0], bounds[1]

        #eco = co[sc.ed[eidx].ravel()]
        #b_min = np.min(eco, axis=0)
        #b_max = np.max(eco, axis=0)
        
    # bounds_8 is for use on the next iteration.
    mid, bounds_8 = generate_bounds(b_min, b_max, margin)
    
    #mid = b_min + ((b_max - b_min) / 2)
    mid_ = mid + margin
    _mid = mid - margin

    x_, y_, z_ = mid_[0], mid_[1], mid_[2]
    _x, _y, _z = _mid[0], _mid[1], _mid[2]

    # tris
    xmax = sc.txmax
    xmin = sc.txmin

    ymax = sc.tymax
    ymin = sc.tymin

    zmax = sc.tzmax
    zmin = sc.tzmin

    # edges
    exmin = sc.exmin
    eymin = sc.eymin
    ezmin = sc.ezmin
    
    exmax = sc.exmax
    eymax = sc.eymax
    ezmax = sc.ezmax

    # l = left, r = right, f = front, b = back, u = up, d = down
    if idx is None:
        idx = sc.tridex
    if eidx is None:    
        eidx = sc.eidx
    
    # -------------------------------
    B = xmin[idx] < x_# + margin
    il = idx[B]

    B = xmax[idx] > _x# - margin
    ir = idx[B]
    
    # edges
    eB = exmin[eidx] < x_# + margin
    eil = eidx[eB]

    eB = exmax[eidx] > _x# - margin
    eir = eidx[eB]

    # ------------------------------
    B = ymax[il] > _y# - margin
    ilf = il[B]

    B = ymin[il] < y_# + margin
    ilb = il[B]

    B = ymax[ir] > _y# - margin
    irf = ir[B]

    B = ymin[ir] < y_# + margin
    irb = ir[B]
    
    # edges
    eB = eymax[eil] > _y# - margin
    eilf = eil[eB]

    eB = eymin[eil] < y_# + margin
    eilb = eil[eB]

    eB = eymax[eir] > _y# - margin
    eirf = eir[eB]

    eB = eymin[eir] < y_# + margin
    eirb = eir[eB]

    # ------------------------------
    B = zmax[ilf] > _z# - margin
    ilfu = ilf[B]
    B = zmin[ilf] < z_# + margin
    ilfd = ilf[B]

    B = zmax[ilb] > _z# - margin
    ilbu = ilb[B]
    B = zmin[ilb] < z_# + margin
    ilbd = ilb[B]

    B = zmax[irf] > _z# - margin
    irfu = irf[B]
    B = zmin[irf] < z_# + margin
    irfd = irf[B]

    B = zmax[irb] > _z# - margin
    irbu = irb[B]
    B = zmin[irb] < z_# + margin
    irbd = irb[B]

    # edges
    eB = ezmax[eilf] > _z# - margin
    eilfu = eilf[eB]
    eB = ezmin[eilf] < z_# + margin
    eilfd = eilf[eB]

    eB = ezmax[eilb] > _z# - margin
    eilbu = eilb[eB]
    eB = ezmin[eilb] < z_# + margin
    eilbd = eilb[eB]

    eB = ezmax[eirf] > _z# - margin
    eirfu = eirf[eB]
    eB = ezmin[eirf] < z_# + margin
    eirfd = eirf[eB]

    eB = ezmax[eirb] > _z# - margin
    eirbu = eirb[eB]
    eB = ezmin[eirb] < z_# + margin
    eirbd = eirb[eB]    

    boxes = [ilbd, irbd, ilfd, irfd, ilbu, irbu, ilfu, irfu]
    eboxes = [eilbd, eirbd, eilfd, eirfd, eilbu, eirbu, eilfu, eirfu]
    
    bbool = np.array([i.shape[0] > 0 for i in boxes])
    ebool = np.array([i.shape[0] > 0 for i in eboxes])
    both = bbool & ebool
    
    full = np.array(boxes, dtype=np.object)[both]
    efull = np.array(eboxes, dtype=np.object)[both]
    
    return full, efull, [bounds_8[0][both], bounds_8[1][both]]
    

def get_link_faces(sc, e=None):
    """Create a set of indices for each edge of faces
    that are linked by the edge vert link faces.
    These faces cannot have the edge passing through them.
    !!! This group will not apply when doing live self colliions !!!
    !!! The points will need to be handled differently from edge segments"""

    if e is not None:
        i = sc.obm.edges[e]
        fa = []
        for v in i.verts:
            for f in v.link_faces:    
                if f.index not in fa:    
                    fa.append(f.index)
        return fa

    linked_by_edge = []
    
    for i in sc.obm.edges:
        fa = []
        for v in i.verts:
            for f in v.link_faces:    
                if f.index not in fa:    
                    fa.append(f.index)
        linked_by_edge.append(fa)
    
    #timer(time.time()-T, "link faces")
    
    return linked_by_edge


def point_in_tri(tri, point):
    """Checks if points are inside triangles"""
    origin = tri[0]
    cross_vecs = tri[1:] - origin
    v2 = point - origin

    # ---------
    v0 = cross_vecs[0]
    v1 = cross_vecs[1]

    #d00_d11 = np.einsum('ijk,ijk->ij', cross_vecs, cross_vecs)
    d00 = v0 @ v0
    d11 = v1 @ v1
    d01 = v0 @ v1
    d02 = v0 @ v2
    d12 = v1 @ v2

    div = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * div
    v = (d00 * d12 - d01 * d02) * div

    #weights = np.array([1 - (u+v), u, v, ])
    check = (u > 0) & (v > 0) & (u + v < 1)
    
    return check
        
    
def inside_triangles(tris, points, cross_vecs):
    """Checks if points are inside triangles"""
    origins = tris[:, 0]
    #cross_vecs = tris[:, 1:] - origins[:, None]
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

    #weights = np.array([1 - (u+v), u, v, ])
    check = (u > 0) & (v > 0) & (u + v < 1)
    
    return check
    

def ray_check(sc, ed, trs):

    if True: # doesn't make much speed difference. Might even be slower
        a = np.concatenate((np.array(sc.ees)[:,None], np.array(sc.trs)[:,None]), 1)
        x = np.random.rand(a.shape[1])
        y = a @ x
        unique, index = np.unique(y, return_index=True)
        ed = np.array(ed)[index]
        trs = np.array(trs)[index]

    e = sc.edges[ed]
    tris = sc.tris[trs]

    origins = tris[:, 0]
    
    ev0 = e[:, 0] - origins
    ev1 = e[:, 1] - origins
    
    cross_vecs = tris[:, 1:] - origins[:, None]
    tv0 = cross_vecs[:, 0]
    tv1 = cross_vecs[:, 1]
        
    norms = np.cross(tv0, tv1)
    
    d0d = np.einsum('ij,ij->i', norms, ev0)
    d0 = np.sign(d0d)
    d1 = np.sign(np.einsum('ij,ij->i', norms, ev1))
    
    # check if edge verts are on opposite sides of the face
    in_edge = d0 != d1
    if np.any(in_edge):

        e_vec = sc.vecs[ed]

        e_dot = np.einsum('ij,ij->i', norms, e_vec)
        scale = d0d / e_dot

        #in_edge = (scale < 0) & (scale > -1)
        on_plane = (ev0 - e_vec * scale[:, None]) + origins

        in_tri = inside_triangles(tris[in_edge], on_plane[in_edge], cross_vecs[in_edge])
        
        sc.has_col = False
        if np.any(in_tri):            
            sc.has_col = True
            sc.trc = np.array(trs, dtype=np.int32)[in_edge][in_tri]
            sc.edc = np.array(ed, dtype=np.int32)[in_edge][in_tri]
            sc.on_plane = on_plane[in_edge][in_tri]
            sc.scale = scale[in_edge][in_tri]
            # selecting ---------
            if sc.sel:
                # edit mode
                select_edit_mode(sc, sc.ob, sc.edc, type='e', obm=sc.obm)
                select_edit_mode(sc, sc.ob, sc.trc, type='f', obm=sc.obm)
                # object mode
                for e in sc.edc:    
                    sc.ob.data.edges[e].select = True
                for t in sc.trc:
                    #bpy.context.scene.self_collisions.append([ed, t])                    
                    sc.ob.data.polygons[t].select = True
    

def b2(sc):

    if len(sc.big_boxes) == 0:
        print("ran out")
        return

    boxes = []
    for oct in sc.big_boxes:
        t = oct[0]
        e = oct[1]
        b = oct[2]
                
        tfull, efull, bounds = octree_et(sc, margin=0.0, idx=t, eidx=e, bounds=b)
        
        for i in range(len(tfull)):
            t = tfull[i]
            e = efull[i]
            bmin = bounds[0][i]
            bmax = bounds[1][i]
            
            if (t.shape[0] < sc.box_max) | (e.shape[0] < sc.box_max):
                sc.small_boxes.append([t, e])
            else:
                boxes.append([t, e, [bmin, bmax]])            
    sc.big_boxes = boxes


def self_collisions_6(sc):
    
    T = time.time()

    tx = sc.tris[:, :, 0]
    ty = sc.tris[:, :, 1]
    tz = sc.tris[:, :, 2]
    
    txmax = np.max(tx, axis=1)
    txmin = np.min(tx, axis=1)

    tymax = np.max(ty, axis=1)
    tymin = np.min(ty, axis=1)

    tzmax = np.max(tz, axis=1)
    tzmin = np.min(tz, axis=1)

    sc.txmax = txmax
    sc.txmin = txmin

    sc.tymax = tymax
    sc.tymin = tymin

    sc.tzmax = tzmax
    sc.tzmin = tzmin

    sc.tridex = sc.indexer # will have to use bmesh tris on non-triangular mesh...

    # edge bounds:
    ex = sc.edges[:, :, 0]
    ey = sc.edges[:, :, 1]
    ez = sc.edges[:, :, 2]

    sc.exmin = np.min(ex, axis=1)
    sc.eymin = np.min(ey, axis=1)
    sc.ezmin = np.min(ez, axis=1)
    
    sc.exmax = np.max(ex, axis=1)
    sc.eymax = np.max(ey, axis=1)
    sc.ezmax = np.max(ez, axis=1)
    
    #sc.eidx = np.arange(sc.edges.shape[0])
        
    timer(time.time()-T, "self col 5")
    # !!! can do something like check the octree to make sure the boxes are smaller
    #       to know if we hit a weird case where we're no longer getting fewer in boxes
    
    tfull, efull, bounds = octree_et(sc, margin=0.0)

    T = time.time()
    for i in range(len(tfull)):
        t = tfull[i]
        e = efull[i]
        bmin = bounds[0][i]
        bmax = bounds[1][i]
        
        if (t.shape[0] < sc.box_max) | (e.shape[0] < sc.box_max):
            sc.small_boxes.append([t, e])
        else:
            sc.big_boxes.append([t, e, [bmin, bmax]]) # using a dictionary or class might be faster !!!
            # !!! instead of passing bounds could figure out the min and max in the tree every time
            #       we divide. So divide the left and right for example then get the new bounds for
            #       each side and so on...
    
    timer(time.time()-T, 'sort boxes')
    T = time.time()
    
    limit = 20
    count = 0
    while len(sc.big_boxes) > 0:
        b2(sc)
        if sc.report:    
            print("recursion level:", count)
        if count > limit:
            for b in sc.big_boxes:
                sc.small_boxes.append(b)
            break
        count += 1    
    
    timer(time.time()-T, 'b2')    
    if sc.report:
        print(len(sc.big_boxes), "how many big boxes")
        print(len(sc.small_boxes), "how many small boxes")
        
    for en, b in enumerate(sc.small_boxes):

        trs = b[0]
        ed = b[1]

        if ed.shape[0] == 0:
            continue
        
        tris = sc.tris[trs]
        eds = sc.edges[ed]
        
        # detect link faces and broadcast
        nlf_0 = sc.ed[ed][:, 0] == sc.fa[trs][:, :, None]
        nlf_1 = sc.ed[ed][:, 1] == sc.fa[trs][:, :, None]
        ab = np.any(nlf_0 | nlf_1, axis=1)
        
        rse = np.tile(ed, trs.shape[0])
        rse.shape = (trs.shape[0], ed.shape[0])
        rst = np.repeat(trs, ed.shape[0])
        rst.shape = (trs.shape[0], ed.shape[0])
        
        re = rse[~ab] # repeated edges with link faces removed
        rt = rst[~ab] # repeated triangles to match above edges
                
        in_x = txmax[rt] > sc.exmin[re]
        rt, re = rt[in_x], re[in_x]

        in_x2 = txmin[rt] < sc.exmax[re]
        rt, re = rt[in_x2], re[in_x2]

        in_y = tymax[rt] > sc.eymin[re]
        rt, re = rt[in_y], re[in_y]

        in_y2 = tymin[rt] < sc.eymax[re]
        rt, re = rt[in_y2], re[in_y2]

        in_z = tzmin[rt] < sc.ezmax[re]
        rt, re = rt[in_z], re[in_z]
        
        in_z2 = tzmax[rt] > sc.ezmin[re]
        rt, re = rt[in_z2], re[in_z2]
                                    
        timer(time.time()-T, 'edge bounds')
        
        T = time.time()
        
        if rt.shape[0] > 0:
            sc.ees += re.tolist()
            sc.trs += rt.tolist()

            #sc.ees = np.concatenate((sc.ees, re))
            #sc.trs = np.concatenate((sc.trs, rt))
        
            #print(sc.ees)

    
class self_collide():
    name = "sc"
    
    def __init__(self, precalc, co=None, test=False):
        if test:
            return
        self.has_col = False
        self.ob = precalc['ob']
        self.ed = precalc['ed']
        self.eidx = precalc['eidx']
        self.fa = precalc['fa']
        self.indexer = precalc['indexer']
        self.box_max = precalc['box_max']
        
        self.co = co
        if co is None:  
            self.co = get_proxy_co(self.ob)
        self.vecs = self.co[self.ed[:, 1]] - self.co[self.ed[:, 0]]
        self.tris = self.co[self.fa]
        self.edges = self.co[self.ed]
        self.big_boxes = [] # boxes that still need to be divided
        self.small_boxes = [] # finished boxes less than the maximum box size

        # debug stuff
        self.sel = False
        #self.sel = True
        self.report = False
        self.report = True
        if self.report:
            self.select_counter = np.zeros(self.eidx.shape[0], dtype=np.int32)        
        if self.sel:
            if self.ob.data.is_editmode:
                self.obm = bmesh.from_edit_mesh(self.ob.data)
            else:    
                self.obm = bmesh.new()
                self.obm.from_mesh(self.ob.data)
            self.obm.edges.ensure_lookup_table()
            self.obm.verts.ensure_lookup_table()
            self.obm.faces.ensure_lookup_table()

        # store sets of edge and tris to check
        #self.trs = np.empty((0), dtype=np.int32)
        #self.ees = np.empty((0), dtype=np.int32)
        self.trs = []
        self.ees = []
        
        

def detect_collisions(ob, co, cloth=None):

    bpy.types.Scene.timers = {}
    bpy.types.Scene.self_collisions = []
    
    # precalc:

    precalc = {'ob': ob,
               'ed': get_edges(ob),
               'eidx': np.arange(len(ob.data.edges), dtype=np.int32),
               'fa': get_faces(ob),
               'indexer': np.arange(len(ob.data.polygons), dtype=np.int32),
               'box_max': 150,
               }

    sc = self_collide(precalc, co)
    t = time.time()
    self_collisions_6(sc)
    ray_check(sc, sc.ees, sc.trs)
    
    # collisions:
    #if sc.has_col: # might not be any collisions
        #print(sc.edc.shape)
        #print(sc.trc.shape)
    # -----------
    
    if sc.report:
        print(sc.box_max, "box max")
        print(np.sum(sc.select_counter > 1), ": In too many boxes")
        print(np.max(sc.select_counter), "max times and edge was selected")
        print(time.time() - t)
        
    if sc.sel:
        if ob.data.is_editmode:
            bmesh.update_edit_mesh(ob.data)
            
        ob.data.update()
    return sc
    

def expand_selection(verts, steps):
    """Take a list of verts and find it's
    neighbors using bool indices from verts
    and tris."""


def selected(ob):
    if ob.data.is_editmode:
        obm = bmesh.from_edit_mesh(ob.data)
        obm.verts.ensure_lookup_table()

    selidx = np.array([v.index for v in obm.verts if v.select], dtype=np.int32)
    return selidx



def which_side(v, faces, pr):
    """Find which direction the flap should fold.
    Verts on the wrong side get moved to the other side.
    'v' is a vert in the flap.
    'faces' are triangle indices in the flap.
    'pr is the vertex pointers for the current flap
    as we iterate through the vertex pointers in the json file"""
    
    
    # for each vert:
    
def grow_selection(verts, obm):
    """Stupid way of growing selection
    by taking link faces of everything"""
    grow = []
    for v in verts:
        vert = obm.verts[v]
        lf = vert.link_faces
        for f in lf:
            vvv = [fv.index for fv in f.verts]
            grow += vvv
    return np.unique(grow)


def shrink_selection(verts, obm):
    """Use link faces to find if surrounding
    verts are in the set. If not drop them."""
    keep = []
    for v in verts:
        vert = obm.verts[v]
        lf = vert.link_faces
        vvv = np.hstack([[fv.index for fv in f.verts] for f in lf])
        
        keeper = True
        for vvvv in vvv:
            if vvvv not in verts:
                keeper = False
        if keeper:
            keep.append(v)
    return keep        
            

def sc_response(sc):
    #print(sc.verts)
    #print(sc.ees)
    #print(sc.trs)
    sc.avatar = [ob for ob in bpy.data.objects if ob.name.startswith('body_mannequin')][0]
    if sc.ob.data.is_editmode:    
        sc.obm = bmesh.from_edit_mesh(sc.ob.data)
    else:
        sc.obm = bmesh.new()
        sc.obm.from_mesh(sc.ob.data)
        
    sc.obm.verts.ensure_lookup_table()    
    sc.obm.edges.ensure_lookup_table()    
    file = bpy.data.texts['flap_ptrs.json']
    flaps = json.loads(file.as_string())
    
    ptrs = [np.array(v) for k, v in flaps.items()]
    
    # mask data missing from slice targets
    sc.folded_co = get_co_key(sc.ob, 'folded')
    sc.flat_co = get_co_key(sc.ob, 'flat')

    isolate = []

    for pr in ptrs:
        folded = np.any(sc.folded_co[pr][:, 2] != sc.flat_co[pr][:, 2])
        print(folded)        
        if folded:
            
            for v in pr:
                if np.all(sc.folded_co[v][:2] == sc.flat_co[v][:2]):
                    #pass
                    isolate.append(v)
                    #sc.obm.verts[v].select = True
                    
        
    #verts = grow_selection(isolate, sc.obm)    
    #isolate = grow_selection(verts, sc.obm)    
    isolate = shrink_selection(isolate, sc.obm)
    #isolate = shrink_selection(isolate, sc.obm)
    idx = np.arange(len(ob.data.vertices), dtype=np.int32)
    invert = np.delete(idx, isolate)
    
    for v in invert:
        sc.obm.verts[v].select = True
    
    #for v in invert:
    group = sc.ob.vertex_groups.new(name='mask')
    group.add(invert.tolist(), 1.0, 'REPLACE')
    
    mask = ob.modifiers.new(name='mask', type='MASK')
    mask.vertex_group = 'mask'
    print(np.array(isolate).shape, "isolate shape !!!!!!!!!!")
        
    return
    
    # figure out if flap folds in or out
    for pr in ptrs:
        
        # get boundary verts
        boundary_verts = [v for v in pr if sc.obm.verts[v].is_boundary]

        # get panel group     
        v1 = sc.ob.data.vertices[pr[0]]
        grs = [g.group for g in v1.groups]
        panel_group = [sc.ob.vertex_groups[g].name for g in grs if sc.ob.vertex_groups[g].name.startswith('P_')][0]

        # get panel tris for flap panel
        plf = []
        for v in pr:
            lf = sc.obm.verts[v].link_faces
            for f in lf:
                pf = True
                for fv in f.verts:
                    if fv.index not in pr:
                        pf = False
                if pf:
                    if f.index not in plf:
                        plf.append(f.index)

        # get vecs from boundary verts to avatar
        for v in boundary_verts:    
            vco = sc.co[v]
            b, location, normal, index = sc.avatar.closest_point_on_mesh(vco)        
        
        for v in pr:    
            sc.obm.verts[v].select=True


        print(location, "location vec")

    if ob.data.is_editmode:    
        bmesh.update_edit_mesh(sc.ob.data)            
                
    '''
    Could add to the selection
    to get the rest of the geomotry
    near the flap...
    Prolly only need
    to check the flap
    against the area the flap
    sews to and compare their
    normals to see if we're on the
    wrong side. Then flip flap
    to other side. Flap is almost
    always going to be the part
    that should move.
    
    once we get the direction of the
    fold I can use the normal
    of the fold to see which side
    of the panel Im on. I dont
    need the avatar normal. maybe
    check the the area around the fold
    for panel faces and check the side???        
    '''
    
    
    return    
        
        #for f in plf:
            #sc.obm.faces[f].select = True                
        
        #for v in pr:
            #sc.obm.verts[v].select = True

        # get a vec between the closes point and
        #   the vert. Check if that vec passes
        #   through a face 
        
        # get closest point on mesh for each boundary vert        
        
        # do raycast for each vec to see if it passes
        #   through any panl tris.
        
        # get the total number that pass through tris
        #   and check if the majority are in or out.
        
        # Go through the panel and check all the flap verts
        # to see if they are in or out.
        
        # move points that are on the wrong side.
        
        # might want to only do this for detected
        #   collisions because flaps might
        #   turn over where they are far from the
        #   body like sleeve ends or hood.
        
    #(could just check all the points in the flaps
    #   any that aren't in or out that should be
    #   can be moved to the opposite side.)
    # could use a similar logic on places
    # where panels poke through.
    # See if the vec between point and closest
    # points on avatar passes through a panel
    # if it passes through but it's on the wrong
    # side of the panel move it in. maybe???    
    
    #sc.ob.data.update()

    print("made it past ptrs")
    print("made it past ptrs")
    print("made it past ptrs")
    
    avatar = [ob for ob in bpy.data.objects if ob.name.startswith('body_mannequin')][0]
    

    
    sew = np.array([len(sc.obm.edges[e].link_faces) == 0 for e in sc.edc])
    e = sc.edc[~sew]
    t = sc.trc[~sew]
    
    
    
    select_edit_mode(sc, sc.ob, e[[0]], type='e', obm=sc.obm)

    #pco = get_proxy_co(sc.ob)
    
    # working with just two for now...
    ve = sc.ed[e[:2]]
    vt = sc.fa[t[:2]]
    #ve_co = sc.co[ve[:2]]
    
    locs = []
    for e, t in zip(ve, vt):
        
        ev1 = sc.ob.data.vertices[e[0]]
        ev2 = sc.ob.data.vertices[e[1]]
        
        tv1 = sc.ob.data.vertices[t[0]]
        
        evgs = [g.group for g in ev1.groups]
        tvgs = [g.group for g in tv1.groups]
        
        epanel = [sc.ob.vertex_groups[g].name for g in evgs if sc.ob.vertex_groups[g].name.startswith('P_')]
        tpanel = [sc.ob.vertex_groups[g].name for g in tvgs if sc.ob.vertex_groups[g].name.startswith('P_')]
        
        print(epanel, tpanel, "do these match???????")
        print(evgs)
        print(tvgs)
        
        match = epanel[0] == tpanel[0]
        print(match, 'this is match')
        
        if match:
            print('same panel logic')
            in_flap = [np.any(e[0] == p) for p in ptrs]
            print(np.any(in_flap))
    
            in_flap = [np.any(e[1] == p) for p in ptrs]
            print(np.any(in_flap))
            print()
            print()
            print()
            print()
            
            '''
            maybe check if its in a flap?
            Use the flap json file
            If I can find it in a flap then
            I can go through the verts in that
            flap and see if they should
            be folded under or over based
            on what the majority of verts do.
            Do the closest point on mesh thing.
            get the dot of the vec
            get the do.
            could do like a raycast.
            see if it goes through a face
            if it goes through a face in the
            same panel I can check if the
            face is closer to the body or
            further from the body in most
            cases...
            could simplify by just checking the boundary edges...
            '''
            
        else:
            print('different panel logic')
        
        vco = sc.co[e[0]]
        
        b, location, normal, index = avatar.closest_point_on_mesh(vco)
        locs.append(location)

        # figure out if the edge is colliding with
        # a face in the same panel or in a different one
    print(locs)

    
    bpy.data.objects['e1'].location = locs[0]
    bpy.data.objects['e2'].location = locs[1]
    
    print(np.sum(sew), "sew edge count")
    print(sc.obm.edges[sc.edc[0]])
    print(len(sc.edc), 'total edge count')
    
    
    '''
    I have these two points
    I have an edge that matches them
    I can get closest point on mesh
    first thing I need is to do is make
    sure we''re not dealing with sew edges
    
    
    identify what panels the point
    and the
    '''
    



print()
print("new =====================================")
ob = bpy.data.objects['g8322']
ob = bpy.data.objects['g8424']
ob = bpy.data.objects['g8424.001']
#ob = bpy.data.objects['Cube']
#ob = bpy.data.objects['p']
#ob = bpy.data.objects['a']
#ob = bpy.data.objects['m']
#ob = bpy.data.objects['pp']

# detect edges passing through faces:
if False: # finds collided edges    
    sc = detect_collisions(ob, None)
    sc = self_collide(precalc=None, co=None, test=False)


# working on fold and stuff...
testing = False
if testing:
    sc = self_collide(precalc=None, co=None, test=True)
    sc.ob = ob
    sc.co = get_proxy_co(sc.ob)

    sc_response(sc)


#could use my cloth engine on the collided parts.
#Could evaluate the sew edges...
#If sew edges connect panel to panel treat them as
#virtual springs or faces.

#Maybe expand the selection and run mc cloth with bend stiff
#Will have to check collisions on everything that moves so on
#edges and verts that are part of the expanded selection around
#collisions.
#Could run smooth iters... might be better to just rund mc linear springs
#could also check against avatar to make sure body is not being penetrated.

'''
        I could finish sewing, finish object collisions,
        identify sew edges that are in the field as virtual springs,
        find relationships between those edge to give them a bias spring,
        create the folded shape from the flap slice target,
        find relationships between the fold flap and the surface below
        where topology has been altered and there are sew springs to hold
        the fold (find the point between the fold and the panel where it
        makes sense to create virtual spring faces with diagonals)


        or: I could figure out a way to untangle self collisions. 


        # rules:
        1. if the edge and the face that are collided are part of the same panel
            Check the direction of the normal relative to the avatar.
            If the section should be fulded under the normal will
            be the opposite direction. Move to the other side
            of the face. 
        2. if the edge and the face are different panels:
            Could draw a line from points in the panels towards the closest point
            on the avatar. Find the neares face in the other panel and see if its
            closer to the avatar of further from it. This would gives us a layer
            order for the panels in question.
            In response I could move the inner points to the closest point on mesh
            of the avatar. 
'''

    
def run_cloth_sim(ob, iters):
    """Use high bend stiff to flatten.
    Treat sew edges as sew or given length
    if they connect in the field."""
    # the vert on at least one side of the edge
    # will have no boundary edges. This should
    # identify if it's boundary or field.
    # might be able to treat field edges
    #   as square faces...

    # maybe find a way to treat area around 
    #   selection as if it were pinned...
        
        



"""
If I can get a decent self collide solver
I might be able to use the current state of the mesh
as a target and move towards that state while preventing
self collisions. Flatten the mesh first to untangle. 

The idea of being on one side of a tri, then being on
the other side of the tri. So at the start of the frame
we check what side we're on. 

At the end of the solve check what side in cloth.select_start.
check what side in cloth.co
Can treat the start and end as edges. where the cross it's like
a raycast. 
if it's on the opposite side, try moving it back to on_plane + some fraction of the vec.
The EA games method was to use the barycentric weights to move
to the face instead of on_plane. 


"""






def sc_response_old(sc, co, shape='MC_current', cloth=None):
    
    #print('-------- response --------')
    #print(sc.on_plane.shape, "on_plane shape")
    keys = sc.ob.data.shape_keys.key_blocks
    k = keys[shape]


    # get uniqe pairs of edges and faces
    # !!! need to test that code for duplicate pairs to see if it's faster !!!
    strs = [str(e) + str(t) for e, t in zip(sc.edc, sc.trc)]
    uni = np.unique(strs, return_index=True)[1]
    ecu = sc.edc[uni]
    tcu = sc.trc[uni]
    
    # use scale to know which direction an edge should move
    scale = sc.scale[uni]

    ze = np.zeros(scale.shape[0], dtype=np.int32)
    #sig = -np.ones(scale.shape[0], dtype=np.float32)
    flip = scale > -0.5
    
    ze[flip] = 1
    #sig[flip] = 1

    side = np.take_along_axis(sc.ed[ecu], ze[:, None], 1)
    
    op = sc.on_plane[uni]
    
    tris = sc.tris[tcu]
    
    origins = tris[:, 0]
    #print(tris[0])
    #print(origins[0])
    v1 = tris[:, 1] - tris[:, 0]
    v2 = tris[:, 2] - tris[:, 0]
    
    norms = np.cross(v1, v2)# * sig[:, None]
    unor = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]
    #print(side.shape, op.shape)
    #print(sc.co[side.ravel()].shape, op.shape)
    vec = sc.co[side.ravel()] - origins
    dist = np.einsum('ij,ij->i', unor, vec)
    move = unor * dist[:, None]
    
    co[side.ravel()] += (move * .2)
    cloth.velocity[side.ravel()] *= 0
    #cloth.velocity[side.ravel()] += (move * .2)
    #sc.co[side.ravel()] += (unor * .1)
    
    
    if False:
        # find collided edges that share a point (like a single vert poking through a face)
        # !!! could do a combination of this method and margin...
        op = sc.on_plane[uni]

        erav = sc.ed[ecu].ravel()


        oprav = np.repeat(op, 2, axis=0)

        uni, idx, counts = np.unique(erav, return_index=True, return_counts=True)
        multiverts = uni[counts > 1]

        for v in multiverts:
            ops = oprav[erav==v]
            op_mean = np.mean(ops, axis=0)
            
            
            vec = op_mean - sc.co[v]     
            co[v] += (vec * .2)     

    #k.data.foreach_set('co', sc.co.ravel())
    #sc.ob.data.update()
        

    return side
    
    np.core.defchararray.add(a1, a2)
    
    e0c = sc.ed[sc.edc][:, 0]
    e1c = sc.ed[sc.edc][:, 1]
    
    #print(sc.scale)
    
    #e0 = sc.edc[:, 0]
    #print(e0.shape, "e0 shape")
    if False:    
        co[this] = sc.on_plane
    #k.data.foreach_set('co', sc.co.ravel())
    #sc.ob.data.update()
    #print(sc.ed.shape, "ed shape")
    #print()
    #sc.co[]
    
