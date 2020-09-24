import bpy
import numpy as np
import bmesh
import time

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


def get_edges(ob, fake=False):
    """Edge indexing for self collision"""
    if fake:
        c = len(ob.data.vertices)
        ed = np.empty((c, 2), dtype=np.int32)
        idx = np.arange(c * 2, dtype=np.int32)
        ed[:, 0] = idx[:c]
        ed[:, 1] = idx[c:]
        return ed
    
    ed = np.empty((len(ob.data.edges), 2), dtype=np.int32)
    ob.data.edges.foreach_get('vertices', ed.ravel())
    return ed


def get_faces(ob):
    """Only works on triangle mesh."""
    fa = np.empty((len(ob.data.polygons), 3), dtype=np.int32)
    ob.data.polygons.foreach_get('vertices', fa.ravel())
    return fa


def get_tridex(ob, tobm=None):
    """Return an index for viewing the verts as triangles"""
    free = True
    if ob.data.is_editmode:
        ob.update_from_editmode()
    if tobm is None:
        tobm = bmesh.new()
        tobm.from_mesh(ob.data)
        free = True
    bmesh.ops.triangulate(tobm, faces=tobm.faces[:])
    tridex = np.array([[v.index for v in f.verts] for f in tobm.faces], dtype=np.int32)
    if free:
        tobm.free()
    return tridex


def inside_triangles(tris, points, margin=0.0):#, cross_vecs):
    """Checks if points are inside triangles"""
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

    w = 1 - (u+v)
    # !!!! needs some thought
    margin = -.2
    # !!!! ==================
    weights = np.array([w, u, v]).T
    check = (u > margin) & (v > margin) & (w > margin)
    
    return check, weights


def b2(sc, cloth):

    
    if len(sc.big_boxes) == 0:
        print("ran out")
        return

    boxes = []
    for oct in sc.big_boxes:
        t = oct[0]
        e = oct[1]
        b = oct[2]
                
        tfull, efull, bounds = octree_et(sc, margin=0.0, idx=t, eidx=e, bounds=b, cloth=cloth)
        
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


def octree_et(sc, margin, idx=None, eidx=None, bounds=None, cloth=None):
    """Adaptive octree. Good for finding doubles or broad
    phase collision culling. et does edges and tris.
    Also groups edges in boxes.""" # first box is based on bounds so first box could be any shape rectangle

    T = time.time()
    margin = sc.M # might be faster than >=, <=
    
    co = cloth.sc_co

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
        idx = cloth.sc_indexer
    if eidx is None:    
        eidx = cloth.sc_eidx

    idx = np.array(idx, dtype=np.int32)
    eidx = np.array(eidx, dtype=np.int32)

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
    

def self_collisions_7(sc, margin=0.1, cloth=None):
    
    T = time.time()

    tx = sc.tris[:, :, 0]
    ty = sc.tris[:, :, 1]
    tz = sc.tris[:, :, 2]
    
    txmax = np.max(tx, axis=1) + margin
    txmin = np.min(tx, axis=1) - margin

    tymax = np.max(ty, axis=1) + margin
    tymin = np.min(ty, axis=1) - margin

    tzmax = np.max(tz, axis=1) + margin
    tzmin = np.min(tz, axis=1) - margin

    sc.txmax = txmax
    sc.txmin = txmin

    sc.tymax = tymax
    sc.tymin = tymin

    sc.tzmax = tzmax
    sc.tzmin = tzmin

    # edge bounds:
    ex = sc.edges[:, :, 0]
    ey = sc.edges[:, :, 1]
    ez = sc.edges[:, :, 2]

    sc.exmin = np.min(ex, axis=1) - margin
    sc.eymin = np.min(ey, axis=1) - margin
    sc.ezmin = np.min(ez, axis=1) - margin
    
    sc.exmax = np.max(ex, axis=1) + margin
    sc.eymax = np.max(ey, axis=1) + margin
    sc.ezmax = np.max(ez, axis=1) + margin
        
    timer(time.time()-T, "self col 5")
    # !!! can do something like check the octree to make sure the boxes are smaller
    #       to know if we hit a weird case where we're no longer getting fewer in boxes
    
    tfull, efull, bounds = octree_et(sc, margin=0.0, cloth=cloth)

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
    
    limit = 3
    count = 0
    while len(sc.big_boxes) > 0:
        b2(sc, cloth)
        if sc.report:    
            print("recursion level:", count)
        if count > limit:
            for b in sc.big_boxes:
                sc.small_boxes.append(b)
            break
        count += 1    

    timer(time.time()-T, 'b2')    
    #if sc.report:
    if 0:
        print(len(sc.big_boxes), "how many big boxes")
        print(len(sc.small_boxes), "how many small boxes")
        
    for en, b in enumerate(sc.small_boxes):
        trs = np.array(b[0], dtype=np.int32)
        ed = np.array(b[1], dtype=np.int32) # can't figure out why this becomes an object array sometimes...

        if ed.shape[0] == 0:
            continue
        
        tris = sc.tris[trs]
        eds = sc.edges[ed]
        
        # detect link faces and broadcast
        nlf_0 = cloth.sc_edges[ed][:, 0] == cloth.tridex[trs][:, :, None]
        ab = np.any(nlf_0, axis=1)
        
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
    

# ^^
def ray_check_(sc, ed, trs, cloth):
    
    e = sc.edges[ed] 
    t = sc.tris[trs]
        
    #tv = [[v for v in sc.ob.data.polygons[t].vertices] for t in trs]
    
    ori1 = t[:, 0]
    ori2 = t[:, 3]
    
    t1a = t[:, 1] - ori1
    t2a = t[:, 2] - ori1
    norms_a = np.cross(t1a, t2a)
    una = norms_a / np.sqrt(np.einsum('ij,ij->i', norms_a, norms_a))[:, None]
    
    t1b = t[:, 4] - ori2
    t2b = t[:, 5] - ori2
    norms_b = np.cross(t1b, t2b)    
    unb = norms_b / np.sqrt(np.einsum('ij,ij->i', norms_b, norms_b))[:, None]    
    # now compare norms
    
    v1 = e[:, 0] - ori1 # could I pull this from the end instead of recalculating
    v2 = e[:, 1] - ori2
    
    d1 = np.einsum('ij,ij->i', v1, una)
    d2 = np.einsum('ij,ij->i', v2, unb)
        
    switch = (d1 * d2) < 0
    
    in_m = np.abs(d2) < sc.M
    
    col = in_m + switch
    #col = switch
    
    pos_neg = np.sign(d1)
    
    
#    the point starts out on one side.
#    It has to move either in the direction
#    of the normal or the opposite direction
#    if it starts on the positive side and its
#    in the margin: it needs to stay on the positive side.
    
    
    #move = d2 * pos_neg # * 1.1
    #col = move < sc.M
    #col = switch
    
    #print(sum(move < sc.M), 'move < sc.M')
    #print(sum(col), 'in_m + switch')
    
    if np.any(col):
        idx = np.array(ed)[col]
#        
#        I need to check if points need to move
#        if they are in the margin they need to move
#        they are in the margin if the dot of the vec
#        with the unorm is less than the margin. 
#        the direction they move will be determined
#        by what side of the face they were on at start
#        so collision = in the margin
#        direction = what side at start.


    
        friction = cloth.ob.MC_props.sc_friction
        if friction > 0:
            check, weights = inside_triangles(t[:, :3][col], cloth.select_start[idx], margin= -sc.M)

            check2, weights2 = inside_triangles(t[:, 3:][col], cloth.select_start[idx], margin= -sc.M)

            #check = check2

            


            plot = t[:, 3:][col] * weights[:, :, None]
            plot2 = t[:, 3:][col] * weights2[:, :, None]
            su1 = np.sum(plot[check], axis=1) * friction
            su2 = np.sum(plot2[check], axis=1) * (1 - friction)
            #check = check | check2

            su = su1 + su2

        else:
            check, weights = inside_triangles(t[:, 3:][col], cloth.select_start[idx], margin= -sc.M)
            plot = t[:, 3:][col] * weights[:, :, None]
            su = np.sum(plot[check], axis=1)# * friction

        test = 0
        if test:
            suu = su #+ surf

            surf = unb[col][check] * sc.M * pos_neg[col][check][:, None]
            cloth.co[idx[check]] = suu + surf
            return
        
        #print(cloth.co.shape, "cloth.co shape")
        idx_check = idx[check]
        #print(check.shape, "check shape")
        #print(idx.shape, "idx shape")
        #print(idx_check.shape, "idx check shape")
        #idx_check = idx#[check]
        co_pre = cloth.co[idx_check]
        vec = su - co_pre
        
        surf = unb[col][check] * sc.M * pos_neg[col][check][:, None]

        uni, inv, counts = np.unique(idx_check, return_inverse=True, return_counts=True)
        
        div = counts[inv][:, None]
            
        cv_ = True
        if cv_:    
            cv = vec# + surf
            
            move_l = np.sqrt(np.einsum('ij,ij->i', cv, cv))
            stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32)
            np.add.at(stretch_array, idx[check], move_l)

            this = stretch_array[idx[check]] / counts[inv]
            this2 = move_l / this
            cv *= this2[:, None]
            cv /= counts[inv][:, None]

            cv += (surf / div)
            cv *= sc.force

        if not cv_:    
            shift = ((vec + surf) / div) * sc.force
            np.add.at(cloth.co, idx_check, shift)
        else:
            np.add.at(cloth.co, idx[check], cv)

        co_post = cloth.co[idx_check]
        dif = co_post - co_pre
        damping = cloth.ob.MC_props.sc_vel_damping
        cloth.velocity[idx_check] += dif
        
        # think about this....
        # what should the cloth velocity do
        #   when a self collision happens...
        # We have existing velocity to consider
        # We have the movement caused by sc to consider
        
        #cloth.velocity[idx_check] *= 0
        cloth.velocity[idx_check] += dif * damping


def ray_check(sc, ed, trs, cloth):
    
    e = sc.edges[ed] 
    t = sc.tris[trs]
        
    #tv = [[v for v in sc.ob.data.polygons[t].vertices] for t in trs]
    
    ori1 = t[:, 0]
    ori2 = t[:, 3]
    
    t1a = t[:, 1] - ori1
    t2a = t[:, 2] - ori1
    norms_a = np.cross(t1a, t2a)
    una = norms_a / np.sqrt(np.einsum('ij,ij->i', norms_a, norms_a))[:, None]

    t1b = t[:, 4] - ori2
    t2b = t[:, 5] - ori2
    norms_b = np.cross(t1b, t2b)    
    unb = norms_b / np.sqrt(np.einsum('ij,ij->i', norms_b, norms_b))[:, None]    
    # now compare norms
    
    v1 = e[:, 0] - ori1 # could I pull this from the end instead of recalculating?
    v2 = e[:, 1] - ori2
    
    d1 = np.einsum('ij,ij->i', v1, una)
    d2 = np.einsum('ij,ij->i', v2, unb)
    
    pos_neg = np.sign(d1)
    

    #switch = (d1 * d2) < 0
    
    #in_m = np.abs(d2) < sc.M
    
    #col = in_m + switch

    
    move = d2 * pos_neg # * 1.1
    col = move < sc.M
    
    if np.any(col):
        idx = np.array(ed)[col]
        
#        getting weights on first tri makes it behave
#        like friction. Second tri no friction.
#        For some reason points are sliding when I move
#        to plot instead of making a vec and going towards plot
#        Cant figure out the difference Needs thought...
        friction = cloth.ob.MC_props.sc_friction
        if friction > 0:
            check, weights = inside_triangles(t[:, :3][col], cloth.select_start[idx], margin= -sc.M)

            check2, weights2 = inside_triangles(t[:, 3:][col], cloth.select_start[idx], margin= -sc.M)

            #check = check2
            


            plot = t[:, 3:][col] * weights[:, :, None]
            plot2 = t[:, 3:][col] * weights2[:, :, None]
            su1 = np.sum(plot[check], axis=1) * friction
            su2 = np.sum(plot2[check], axis=1) * (1 - friction)
            #check = check | check2

            su = su1 + su2

        else:
            check, weights = inside_triangles(t[:, 3:][col], cloth.select_start[idx], margin= -sc.M)
            plot = t[:, 3:][col] * weights[:, :, None]
            su = np.sum(plot[check], axis=1)# * friction

        test = False
        if test:
            suu = su #+ surf
            cloth.co[idx[check]] = suu    
            return
        
        idx_check = idx[check]
        co_pre = cloth.co[idx_check]
        vec = su - co_pre
        surf = unb[col][check] * sc.M * pos_neg[col][check][:, None]

        uni, inv, counts = np.unique(idx_check, return_inverse=True, return_counts=True)
        
        div = counts[inv][:, None]
            
        cv_ = True
        if cv_:    
            cv = vec# + surf
            
            move_l = np.sqrt(np.einsum('ij,ij->i', cv, cv))
            stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32)
            np.add.at(stretch_array, idx[check], move_l)

            this = stretch_array[idx[check]] / counts[inv]
            this2 = move_l / this
            cv *= this2[:, None]
            cv /= counts[inv][:, None]

            cv += (surf / div)
            cv *= sc.force

        if not cv_:    
            shift = ((vec + surf) / div) * sc.force
            np.add.at(cloth.co, idx_check, shift)
        else:
            np.add.at(cloth.co, idx[check], cv)

        co_post = cloth.co[idx_check]
        dif = co_post - co_pre
        damping = cloth.ob.MC_props.sc_vel_damping
        #cloth.velocity[idx_check] += (dif * 2)
        
        # think about this....
        # what should the cloth velocity do
        #   when a self collision happens...
        # We have existing velocity to consider
        # We have the movement caused by sc to consider
        
        cloth.velocity[idx_check] *= 0
        cloth.velocity[idx_check] += dif * damping
        

def ray_check_1(sc, ed, trs, cloth):
    
    e = sc.edges[ed] 
    t = sc.tris[trs]
        
    #tv = [[v for v in sc.ob.data.polygons[t].vertices] for t in trs]
    
    ori1 = t[:, 0]
    ori2 = t[:, 3]
    
    t1a = t[:, 1] - ori1
    t2a = t[:, 2] - ori1
    norms_a = np.cross(t1a, t2a)
    una = norms_a / np.sqrt(np.einsum('ij,ij->i', norms_a, norms_a))[:, None]
    

    t1b = t[:, 4] - ori2
    t2b = t[:, 5] - ori2
    norms_b = np.cross(t1b, t2b)    
    unb = norms_b / np.sqrt(np.einsum('ij,ij->i', norms_b, norms_b))[:, None]    
    # now compar norms
    
    v1 = e[:, 0] - ori1 # could I pull this from the end instead of recalculating
    v2 = e[:, 1] - ori2
    
    d1 = np.einsum('ij,ij->i', v1, una)
    d2 = np.einsum('ij,ij->i', v2, unb)
    
    pos_neg = np.sign(d1)
    
    
    move = d2 * pos_neg # * 1.1
    col = move < sc.M
    
    if np.any(col):
        idx = np.array(ed)[col]
        
#        getting weights on first tri makes it behave
#        like friction. Second tri no friction.
#        For some reason points are sliding when I move
#        to plot instead of making a vec and going towards plot
#        Cant figure out the difference Needs thought...

        friction = True
        if friction:    
            check_b, weights_b = inside_triangles(t[:, :3][col], cloth.select_start[idx], margin= -sc.M)
            check, weights = inside_triangles(t[:, 3:][col], cloth.select_start[idx], margin= -sc.M)
            check = check | check_b

        else:
            check, weights = inside_triangles(t[:, 3:][col], cloth.select_start[idx], margin= -sc.M)

        plot = t[:, 3:][col] * weights[:, :, None]
        su = np.sum(plot[check], axis=1)

        test = False
        if test:
            suu = su + surf
            cloth.co[idx[check]] = suu    
            return
        
        idx_check = idx[check]
        co_pre = cloth.co[idx_check]
        vec = su - co_pre
        surf = unb[col][check] * sc.M * pos_neg[col][check][:, None]

        uni, inv, counts = np.unique(idx_check, return_inverse=True, return_counts=True)
        
        div = counts[inv][:, None]
            
        cv_ = True
        if cv_:    
            cv = vec# + surf
            
            move_l = np.sqrt(np.einsum('ij,ij->i', cv, cv))
            stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32)
            np.add.at(stretch_array, idx[check], move_l)

            this = stretch_array[idx[check]] / counts[inv]
            this2 = move_l / this
            cv *= this2[:, None]
            cv /= counts[inv][:, None]

            cv += (surf / div)
            cv *= sc.force

        if not cv_:    
            shift = ((vec + surf) / div) * sc.force
            np.add.at(cloth.co, idx_check, shift)
        else:
            np.add.at(cloth.co, idx[check], cv)

        co_post = cloth.co[idx_check]
        dif = co_post - co_pre
        cloth.velocity[idx_check] += dif
        damping = cloth.ob.MC_props.sc_vel_damping
        cloth.velocity[idx_check] *= damping


class SelfCollide():
    name = "sc"
    
    def __init__(self, cloth):

        # -----------------------
        ob = cloth.ob
        tris_six = cloth.tris_six
        tridex = cloth.tridex
        
        cloth.sc_co[:cloth.v_count] = cloth.select_start
        cloth.sc_co[cloth.v_count:] = cloth.co

        tris_six[:, :3] = cloth.select_start[cloth.tridex]
        tris_six[:, 3:] = cloth.co[cloth.tridex]
        # -----------------------
        
        self.has_col = False

        #self.indexer = cloth.sc_indexer

        self.box_max = cloth.ob.MC_props.sc_box_max

        self.M = cloth.ob.MC_props.self_collide_margin
        self.force = cloth.ob.MC_props.self_collide_force
        
        self.tris = tris_six
        self.edges = cloth.sc_co[cloth.sc_edges]
        self.big_boxes = [] # boxes that still need to be divided
        self.small_boxes = [] # finished boxes less than the maximum box size

        # debug stuff
        self.sel = False
        #self.sel = True
        self.report = False
        #self.report = True
        if self.report:
            self.select_counter = np.zeros(cloth.sc_eidx.shape[0], dtype=np.int32)        
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
        

def detect_collisions(cloth):
    
    sc = SelfCollide(cloth)
    t = time.time()

    self_collisions_7(sc, sc.M, cloth)

    ray_check(sc, sc.ees, sc.trs, cloth)
        
    if sc.report:
        print(sc.box_max, "box max")
        print(np.sum(sc.select_counter > 1), ": In too many boxes")
        print(np.max(sc.select_counter), "max times and edge was selected")
        print(time.time() - t)
        
    if sc.sel:
        if ob.data.is_editmode:
            bmesh.update_edit_mesh(ob.data)
            
        ob.data.update()


'''



'''



'''
Might want to look into skipping
the bounds check. Just use smaller boxes
and go straight to check every pair for
which side of tri...

so I group the tris in sets of 6.
the start and end tri.
I get the bounds from that.
So bounding boxes will be around

a tri is like Nx3x3
a moving tri is like Nx6x3
I still get the bounds from axis 1

I can use the start and end of each point
just like an edge. Should be able
to set up phony edge indexing and
use the exact same system.
Should work all the way until raycast.

once we get to raycast...
were dealing with edges that are the
start and end of moving points
and two tris that ar the start and
end of a moving tri.
I can check what side Im on
in the beginning by getting the
dot of the point origin and the normal
for the start edge and the start tri...
If I get what side the end tri and end
point are on and its different I could
then do a bary check...

Should I bary check both points?
drop to plane with first tri?
drop to plane with second tri?
bary check without dropping to plane?
check 1st tri without dropping to plane?
check 2nd tri without dropping to plane?

'''
