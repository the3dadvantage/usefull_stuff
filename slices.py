try:
    import bpy
    import numpy as np
    import json

except:
    pass


def reset_shapes(ob):
    """Create shape keys if they are missing"""

    if ob.data.shape_keys == None:
        ob.shape_key_add(name='Basis')
    
    keys = ob.data.shape_keys.key_blocks    
    if 'MC_source' not in keys:
        ob.shape_key_add(name='MC_source')
        keys['MC_source'].value=1
    
    if 'MC_current' not in keys:
        ob.shape_key_add(name='MC_current')
        keys['MC_current'].value=1
        keys['MC_current'].relative_key = keys['MC_source']


def get_co_shape(ob, key=None, ar=None):
    """Get vertex coords from a shape key"""
    v_count = len(ob.data.shape_keys.key_blocks[key].data)
    if ar is None:
        ar = np.empty(v_count * 3, dtype=np.float32)
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar)
    ar.shape = (v_count, 3)
    return ar


def link_mesh(verts, edges=[], faces=[], name='name'):
    """Generate and link a new object from pydata"""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)  
    mesh.update()
    mesh_ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(mesh_ob)
    return mesh_ob


def create_triangles(Slice, s_count, x_off=None):
    """Creates equalateral triangles whose edge length
    is similar to the distance between slices so that
    the bend stifness is more stable.
    x_off is for debug moving the next slice over"""
    
    s = Slice.seam_sets[s_count]
    means = s['tri_means']
    dist = np.copy(s['dst'])
    dist[0] = s['avd']
    count = dist.shape[0]
    
    # constant values
    height = np.sqrt(3)/2
    offset = ((dist * height) * 0.5)
    
    # build tris
    a = np.copy(means)
    a[:, 0] -= (dist * .5)
    a[:, 1] -= offset - (offset * (1/3))
    b = np.copy(a)
    b[:, 0] += dist
    c = np.copy(means)
    c[:, 1] += offset + (offset * (1/3))
    # abc is counterclockwise starting at bottom left

    tri = np.zeros(count * 9)
    tri.shape = (count, 3, 3)
    tri[:, 0] += a
    tri[:, 1] += b
    tri[:, 2] += c
            
    if x_off is not None:
        tri[:, :, 0] += x_off
    
    return tri


def create_mesh_data(Slice, s_count):
    """Build edge and face data for
    the tubes of triangles"""
    
    s = Slice.seam_sets[s_count]
    tri = s['tris']
    count = tri.shape[0]

    # build edges
    edges = np.array([[0,1],[1,2],[2,0]])
    ed = np.zeros(count * 6, dtype=np.int32)
    ed.shape = (count, 3, 2)
    ed += edges    
    ed += np.arange(0, count * 3, 3)[:, None][:, None]

    # build faces    
    faces = np.array([[0,1,4,3], [2,0,3,5], [2,1,4,5]])
    fa = np.zeros((count -1) * 12, dtype=np.int32)
    fa.shape = (count -1, 3, 4)
    fa += faces
    fa += np.arange(0, (count -1) * 3, 3)[:, None][:, None]

    return ed, fa


def slice_setup(Slice, testing=True): # !!! set testing to False !!!
    print("seam wrangler is reminding you to set slice_setup testing to False")
    file = bpy.data.texts['slice_targets.json']
    slices = json.loads(file.as_string())
    Slice.count = len(slices)
    
    ob = Slice.ob
    
    # get the name of the cloth state shape key (numbers will vary)
    keys = ob.data.shape_keys.key_blocks
    cloth_key = [i.name for i in keys if i.name.startswith("CLOTH")][0]
    Slice.cloth_key = cloth_key
    
    # flat shape coords
    flat_co = get_co_shape(ob, 'flat')
    Slice.flat_co = flat_co

    # cloth shape coords
    cloth_co = get_co_shape(ob, cloth_key)
    Slice.cloth_co = cloth_co
    
    # ------------
    seam_sets = {}
    seam_sets['unresolved gaps'] = []
    name = 0
    
    for s in slices:
        vp_with_nones = np.array(s['vert_ptrs']).T
        
        xys = []
        vps = []
        vpsN = [] # with Nones
        dst = []
        avds = []
        idxs = []
        tri_means = []
        av_tri_mean = []
        last_idx = None
        last_j = None
        
        ticker = 0
        last_tick = 0
        
        for j in vp_with_nones:

            xy_with_nones = np.array(s['target_xys'], dtype=np.float32)
            flying_Nones = j != None
            
            vp = j[flying_Nones]
            xy = xy_with_nones[flying_Nones]

            # for testing !!! Disable !!! (already getting scaled in sims)
            if testing:    
                xy *= np.array([0.1, 0.05], dtype=np.float32)
            # for testing !!! Disable !!!                            

            # get some triangle means (check later to make sure there is at least one)
            tri_mean = None
            if np.all(flying_Nones):
                tri_mean = np.mean(xy, axis=0)
                av_tri_mean += [tri_mean]
                
            # get distances -------------
            vpc = len(j)
            dist = None
            
            if vpc > 0:    
                idx = np.arange(vpc, dtype=np.int32)[flying_Nones]
                idxs += [idx]
                
            if last_idx is not None:    
                in1d = np.in1d(idx, last_idx)
                
                if np.any(in1d):
                    good = np.array(j[idx[in1d]], dtype=np.int32)
                    last_good = np.array(last_j[idx[in1d]], dtype=np.int32)

                    vecs = flat_co[good] - flat_co[last_good]
                    dists = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
                    dist = np.mean(dists)
                    
                    # check if we stepped only once (for average distance)
                    if ticker - last_tick == 1:
                        avds += [dist]
                    last_tick = ticker
                    
            xys += [xy]
            vps += [vp]
            vpsN += [j] # with Nones
            dst += [dist]
            tri_means += [tri_mean]
            
            # -----------------
            last_idx = idx
            last_j = j
            
            # for getting average distance from single steps
            ticker += 1
        
        avtm = np.mean(av_tri_mean, axis=0)
        # in case there are no complete sets of points in a slice
        if np.any(np.isnan(avtm)):
            av_tri_means = []
            for j in vp_with_nones:

                xy_with_nones = np.array(s['target_xys'], dtype=np.float32)
                flying_Nones = j != None
                
                vp = j[flying_Nones]
                xy = xy_with_nones[flying_Nones]

                # for testing !!! Disable !!! (already getting scaled in sims)
                if testing:    
                    xy *= np.array([0.1, 0.05], dtype=np.float32)
                # for testing !!! Disable !!!                            

                # get some triangle means (check later to make sure there is at least one)
                tri_mean = None
                if np.any(flying_Nones):
                    tri_mean = np.mean(xy, axis=0)
                    av_tri_mean += [tri_mean]
            avtm = np.mean(av_tri_mean, axis=0)
        
        avd = np.mean(avds)
        seam_sets[name] = {'xys': xys,
                           'vps': vps,
                           'vpsN':vpsN,
                           'dst': dst,
                           'tri_means': tri_means,
                           'av_tri_mean':avtm,
                           'avd':avd,
                           'idx':idxs}            

        name += 1
    Slice.seam_sets = seam_sets


def missing_distance(Slice):
    """Fill in missing data.
    Find and deal with gaps between slices"""
    
    # Need a distance between each triangle (Some are None [wierd that that's a true statement])
    ob = Slice.ob
    flat_co = Slice.flat_co
    
    # -------------------------    
    s_count = 0
    for i in range(Slice.count):
        s = Slice.seam_sets[i]
        count = 0
        
        # Create a state that checks for Nones between non-None distances
        # This way if there is a gap we get the right distance between the sections where there is a gap

        switch1 = False
        switch2 = False
        None_state = False
        last_vpN = None
        lidx = None
                
        for i in range(len(s['dst'])):
            d = s['dst'][i]
            
            if not switch1:
                if d is not None:
                    switch1 = True
                    
            if switch1:
                if d is None:
                    switch2 = True
                    
            if switch2:
                if d is not None:
                    switch1 = False
                    switch2 = False
                    None_state = True

            if d is not None:
                if None_state:
                    cvpN = s['vpsN'][i-1]
                    idx = s['idx'][i-1]

                    in1d = np.in1d(idx, lidx)
                    if np.any(in1d):
                        good = np.array(cvpN[idx[in1d]], dtype=np.int32)
                        last_good = np.array(last_vpN[idx[in1d]], dtype=np.int32)

                        vecs = flat_co[good] - flat_co[last_good]
                        dists = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
                        dist = np.mean(dists)
                        
                        # count backwards to last good distance
                        div = 1
                        bc = i - 2
                        while s['dst'][bc] is None:
                            div += 1
                            bc -= 1
                       
                        # fast forward where we just rewound
                        for r in range(div):
                            s['dst'][i-div + r] = dist/div

                        print('Seam wrangler resolved gap')                        
                    
                    else:
                        Slice.seam_sets['unresolved gaps'] += [s_count]
                        print("Unresolved gaps in seam_wrangler")
                        print("Might distort some seams (but probably not)")
                    
                    None_state = False
                    for v in s['vps'][i]:
                        ob.data.vertices[v].select = True
            
                last_vpN = s['vpsN'][i]
                lidx = s['idx'][i]
                
            count += 1
        
        # overwrite remaining Nones with avd
        s['dst'][0] = 0.0
                
        for i in range(len(s['dst'])):
            d = s['dst'][i]
            if d is None:
                s['dst'][i] = s['avd']

        cum_dst = np.cumsum(s['dst'])
        s['cum_dst'] = cum_dst        

        # overwrite tri mean Nones
        for i in range(len(s['tri_means'])):
            if s['tri_means'][i] is None:
                s['tri_means'][i] = s['av_tri_mean']
        
        add_z = np.zeros(cum_dst.shape[0] * 3, dtype=np.float32)
        add_z.shape = (cum_dst.shape[0], 3)
        add_z[:, :2] = s['tri_means']
        add_z[:, 2] = cum_dst
        s['tri_means'] = add_z
        
        # iterate tick -----------                
        s_count += 1


def build_data(Slice):
    """Generate meshes and such"""
    
    ob = Slice.ob
    flat_co = Slice.flat_co
    cloth_co = Slice.cloth_co
    
    # -------------------------    
    s_count = 0
    for i in range(Slice.count):
        s = Slice.seam_sets[i]
    
        # build triangles for the mesh
        s['tris'] = create_triangles(Slice, s_count)        
        
        ed, fa = create_mesh_data(Slice, s_count)
        es = ed.shape
        ed.shape = (es[0] * 3, 2)
        
        fs = fa.shape
        fa.shape = (fs[0] * 3, 4)
        
        
        test_count = 1
        if s_count == test_count:
            
            # create the mesh or merge lists to make one mesh
            ts = s['tris'].shape
            s['tris'].shape = (ts[0] * 3, 3)
            
            if "tri_" + str(s_count) not in bpy.data.objects:
                tri_mesh = link_mesh(s['tris'].tolist(), ed.tolist(), fa.tolist(), "tri_" + str(s_count))
                reset_shapes(tri_mesh)
            
            vpm = []
            
            for v in s['vps']:
                idx = np.array(v, dtype=np.int32)
                m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                
                if v.shape[0] > 0:
                    m = [np.mean(cloth_co[idx], axis=0)]
                vpm += [m]
            
            #print(np.array(vpm))
            for i in np.array(vpm):
                print(i)
            
            
            
        # iterate tick -----------
        s_count += 1    
    
    


        
    
def testing(Slice):
    
    seam_sets = Slice.seam_sets    

    id0 = 5
    id1 = 17
    
    if False:
        print(seam_sets[id1]['xys'][id0])
        print(seam_sets[id1]['vps'][id0])
        print(seam_sets[id1]['dst'][id0])
        print(seam_sets[id1]['dst'][id0])
        print(seam_sets[id1]['avd'], "this val")
        #print(seam_sets[id1]['dst'])
        #print(seam_sets[id1]['vps'])
    
    print(seam_sets[id1]['dst'])
    print(seam_sets[id1]['cum_dst'])
    ob.data.update()
    return
    for i in seam_sets[id1]['vps']:
        for j in i:
            ob.data.vertices[j].select = True
    ob.data.update()
    
    

print()
print()
print()
print('start ========================')
print('start ========================')
print('start ========================')
print('start ========================')
print('start ========================')

class Slices():
    pass

def slices_main(ob):
    Slice = Slices()
    Slice.ob = ob
    
    # setup functions
    slice_setup(Slice)
    missing_distance(Slice)
    build_data(Slice)


ob = bpy.data.objects['g6774']
slices_main(ob)





# At each triangle we need to be spaced according to the
#   spacing between slices.
# Some slices have no way to give us distances
# We want to get the average of distances that are one step apart
# We also want to get the right distance where slices are more
#   than one step apart. 
