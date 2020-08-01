import bpy
import numpy as np
import bmesh
import time


counter = {}
counter['count'] = 0


def timer(t, name='name'):
    ti = bpy.context.scene.timers
    if name not in ti:
        ti[name] = 0.0
    ti[name] += t


def select_edit_mode(ob, idx, type='v', deselect=False, obm=None):
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
    
    full = np.array(boxes)[both]
    efull = np.array(eboxes)[both]
    
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
    

def ray_check(ed, trs):

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
        
        if np.any(in_tri):
            sc.useful_counter_2 += 1
            
            tr = np.array(trs)[in_edge][in_tri]
            ed = np.array(ed)[in_edge][in_tri]

            # selecting ---------
            if sc.sel:
                # edit mode
                select_edit_mode(sc.ob, ed, type='e', obm=sc.obm)
                select_edit_mode(sc.ob, tr, type='f', obm=sc.obm)
                # object mode
                for e in ed:    
                    ob.data.edges[e].select = True
                for t in tr:
                    bpy.context.scene.self_collisions.append([ed, t])                    
                    ob.data.polygons[t].select = True
    

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
        print("recursion level:", count)
        if count > limit:
            for b in sc.big_boxes:
                sc.small_boxes.append(b)
            break
        count += 1    
    
    timer(time.time()-T, 'b2')    
    print(len(sc.big_boxes), "how many big boxes")
    print(len(sc.small_boxes), "how many small boxes")

        
    for en, b in enumerate(sc.small_boxes):

        trs = b[0]
        ed = b[1]

        if ed.shape[0] == 0:
            continue
        
        tris = sc.tris[trs]
        eds = sc.edges[ed]

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

    
class self_collide():
    name = "sc"

    def __init__(self):
        self.ob = ob    
        self.ed = get_edges(ob)
        self.eidx = np.arange(self.ed.shape[0])
        self.fa = get_faces(ob)
        self.co = get_proxy_co(ob)
        self.vecs = self.co[self.ed[:, 1]] - self.co[self.ed[:, 0]]
        self.tris = self.co[self.fa]
        self.edges = self.co[self.ed]
        self.box_max = 200
        self.indexer = np.arange(self.fa.shape[0])
        self.indexer2 = np.copy(self.indexer)
        self.big_boxes = [] # boxes that still need to be divided
        self.big_boxes2 = [] # boxes that still need to be divided
        self.small_boxes = [] # finished boxes less than the maximum box size
        self.useful_counter = 0
        self.useful_counter_2 = 0
        self.singles = 0
        self.sb_counter = 0
        self.select_counter = np.zeros(self.eidx.shape[0], dtype=np.int32)
        self.recursion_count = 0
        # bmesh stuff
        self.sel = False
        self.sel = True
        
        if self.sel:
            if self.ob.data.is_editmode:
                self.obm = bmesh.from_edit_mesh(ob.data)
            else:    
                self.obm = bmesh.new()
                self.obm.from_mesh(ob.data)
            self.obm.edges.ensure_lookup_table()
            self.obm.verts.ensure_lookup_table()
            self.obm.faces.ensure_lookup_table()

        # store sets of edge and tris to check
        self.trs = []
        self.ees = []
        
        
print()
print("new =====================================")
ob = bpy.data.objects['g8322']
#ob = bpy.data.objects['Cube']
#ob = bpy.data.objects['p']
#ob = bpy.data.objects['a']
#ob = bpy.data.objects['m']
#ob = bpy.data.objects['pp']

if False:
    # checking collide without octree
    if ob.data.is_editmode:
        bmesh.update_edit_mesh(ob.data)
    t = time.time()
    sc = self_collide()
    for ed in sc.edexer:
        ray_check(ed, sc.indexer, en=None, i=None)
    print(time.time() - t)
    
if True:

    bpy.types.Scene.timers = {}
    bpy.types.Scene.self_collisions = []

    print()
    t = time.time()
    sc = self_collide()
    timer(time.time()-t, 'build class')
    sc.box_max = 150
    self_collisions_6(sc)
    print(sc.box_max, "box max")
    print(sc.useful_counter_2, "edges flagged")
    print(np.sum(sc.select_counter > 1), ": In too many boxes")
    print(np.max(sc.select_counter), "max times and edge was selected")
    print(sc.singles, "singles")
    ray_check(sc.ees, sc.trs)
    print(time.time() - t)
    
    if sc.sel:
        if ob.data.is_editmode:
            bmesh.update_edit_mesh(ob.data)
            
        ob.data.update()


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



def expand_selection(verts, steps):
    """Take a list of verts and find it's
    neighbors using bool indices from verts
    and tris."""
    
    
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
        
        
def generate_mids(minc, maxc):
    """from a min corner and a max corner
    generate the min and max corner of 8 boxes"""

    diag = (maxc - minc) / 2
    half = diag / 2
    mid = minc + diag
    mids = np.zeros((8,3), dtype=np.float32) 
    # up, down, left, right, front, back
    # dlf
    mids[0] = minc + half
    # drf
    mids[1] = mids[0]
    mids[1][0] += diag[0]
    # dlb
    mids[2] = mids[0]
    mids[2][1] += diag[1]
    # drb
    mids[3] = mids[1]
    mids[3][1] += diag[1]
    # ulf
    mids[4] = mids[0]
    mids[4][2] += diag[2]
    # urf
    mids[5] = mids[1]
    mids[5][2] += diag[2]
    # ulb
    mids[6] = mids[2]
    mids[6][2] += diag[2]
    # urb
    mids[7] = mids[3]
    mids[7][2] += diag[2]
    
    return mid, mids
