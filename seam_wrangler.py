# undo !!!!!!
#def b_log(a=None,b=None):
    #return


try:
    import bpy
    import numpy as np
    import json
    actual_frames = [0]
    test_cache = False
    #test_cache = True

except ImportError:
    pass

try:
    from garments_blender.utils.rich_blender_utils import fix_all_shape_key_nans
    from garments_blender.utils.rich_blender_utils import B_log
    from garments_render.simulation.MC_tools import read_python_script
    internal_log = B_log()
    internal_log.module = 'seam wrangler'
    internal_log.active = True
    internal_log.name = 'log_seam_wrangler'
    b_log = internal_log.b_log

except:
    print("if not calling seam manager outside p1 we have a problem")
    rbu = bpy.data.texts['rich_blender_utils.py'].as_module()

    internal_log = rbu.B_log()
    internal_log.module = 'MC_tools'
    internal_log.active = True
    internal_log.name = 'log_MC_tools'
    b_log = internal_log.b_log

    fix_all_shape_key_nans = rbu.fix_all_shape_key_nans
    mct = bpy.data.texts['MC_tools.py'].as_module()
    read_python_script = mct.read_python_script
    

def setup_pin_group(ob, vidx):
    # setup vertex pin group. (matches points manipulated by seam manager)

    if 'SW_seam_pin' not in ob.vertex_groups:
        ob.vertex_groups.new(name='SW_seam_pin')

    pidx = ob.vertex_groups['SW_seam_pin'].index
    vc = len(ob.data.vertices)
    ara = np.arange(vc)

    for v in vidx.tolist():
        ob.vertex_groups['SW_seam_pin'].add([pidx, v], 0.0, 'REPLACE')

    for mod in ob.modifiers:
        if mod.type == "CLOTH":
            mod.settings.vertex_group_mass = 'SW_seam_pin'
            mod.settings.pin_stiffness = 1.0


def get_proxy_co(ob, co=None):
    """Gets co with modifiers like cloth"""
    dg = bpy.context.evaluated_depsgraph_get()
    prox = ob.evaluated_get(dg)
    proxy = prox.to_mesh()
    if co is None:
        vc = len(ob.data.vertices)
        co = np.empty((vc, 3), dtype=np.float32)
    proxy.vertices.foreach_get('co', co.ravel())
    ob.to_mesh_clear()
    return co


def sw_shapes(ob):
    """Create shape key for viewing seam wrangler"""

    if ob.data.shape_keys == None:
        ob.shape_key_add(name='Basis')

    keys = ob.data.shape_keys.key_blocks
    if 'sw_view' not in keys:
        ob.shape_key_add(name='sw_view')
        keys['sw_view'].value=1


def reset_shapes(ob):
    """Create shape keys for guide mesh"""

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
    vc = len(ob.data.shape_keys.key_blocks[key].data)
    if ar is None:
        ar = np.empty((vc, 3), dtype=np.float32)
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
    return ar


def link_mesh(verts, edges=[], faces=[], name='name'):
    """Generate and link a new object from pydata"""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    mesh_ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(mesh_ob)
    return mesh_ob


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


def slice_setup(Slice, cloth_key=None): # !!! set testing to False !!!
    testing = False
    #print("seam wrangler is reminding you to set slice_setup testing to False")
    file = bpy.data.texts['slice_targets.json']
    slices = json.loads(file.as_string())
    Slice.count = len(slices)

    ob = Slice.ob

    Slice.cloth_key = None

    # get the name of the cloth state shape key (numbers will vary)
    if cloth_key is None:
        if ob.data.shape_keys is not None:
            keys = ob.data.shape_keys.key_blocks
            names = [i.name for i in keys if ((i.name != 'sw_view') & (i.value == 1))]
            if len(names) > 0:
                Slice.cloth_key = names[-1]
            if len(names) > 1:
                msg = ["!!! Warning. More than one shape key with value of 1" for i in range(3)]
                msg += names
                b_log(msg)

    # flat shape coords or mesh coords if there is no flat shape key
    flat_key = False

    if ob.data.shape_keys is not None:
        no_key = True
        if 'flat' in ob.data.shape_keys.key_blocks:
            flat_co = get_co_shape(ob, 'flat')
            flat_key = True

    if not flat_key:
        vc = len(ob.data.vertices)
        flat_co = np.empty((vc, 3), dtype=np.float32)
        ob.data.vertices.foreach_get('co', flat_co.ravel())
        b_log(['!!! No flat key when seam wrangler data was created !!!', 'Using mesh coords'])

    Slice.flat_co = flat_co

    # cloth shape coords
    cloth_co = get_proxy_co(ob)
    Slice.cloth_co = cloth_co

    # ------------
    seam_sets = {}
    seam_sets['unresolved gaps'] = []
    name = 0

    for s in slices:
        vp_with_nones = np.array(s['vert_ptrs']).T

        tri_tick = vp_with_nones.shape[0] # For offsetting the tri tiler when merging
        xys = []  # xy map
        vps = []  # vertex pointers
        vpsN = [] # with Nones
        dst = []  # distance between slices
        avds = [] # average distance for each seam
        idxs = [] # aranged indexer for each slice
        tri_means = []    # center of built triangles
        av_tri_mean = []  # average center for each seam
        tri_tiler = []    # tiled fancy index for getting weights
        good_xys = []     # xys where there is at least one vp
        good_bool = []    # True where there is at least one vp
        complete = []     # areas with no Nones. Used for placing tris

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

            good = False
            gvpc = vp.shape[0] # at least one vp in the set
            if gvpc > 0:
                tri_tiler  += ([ticker] * gvpc)
                good_xys += xy.tolist()
                good = True
            good_bool += [good]

            # get some triangle means (check later to make sure there is at least one)
            tri_mean = None
            comp = False
            if np.all(flying_Nones):
                comp = True
                tri_mean = np.mean(xy, axis=0)
                av_tri_mean += [tri_mean]
            complete += [comp]

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
        seam_sets[name] = {'tri_tick': tri_tick,
                           'xys': xys,
                           'vps': vps,
                           'vpsN': vpsN,
                           'dst': dst,
                           'tri_means': tri_means,
                           'av_tri_mean': avtm,
                           'tri_tiler': tri_tiler,
                           'good_xys': np.array(good_xys, dtype=np.float32),
                           'good_bool': np.array(good_bool, dtype=np.bool),
                           'avd': avd,
                           'idx': idxs,
                           'complete': np.array(complete, dtype=np.bool),
                           }

        name += 1
    Slice.seam_sets = seam_sets


def missing_distance(Slice, test_val):
    """Fill in missing data.
    Find and deal with gaps between slices"""

    # !!! Currently uses good distances. Could use good vps instead
    #   Most seams have a good vp at the start but the distance is not there
    #   Could resolve more gaps if we measured between the first vp and
    #   the next good vp.

    # Need a distance between each triangle (Some are None [wierd that that's a true statement])
    ob = Slice.ob
    flat_co = Slice.flat_co
    Slice.seam_sets['tri_means'] = np.empty((0,3), dtype=np.float32)
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

                        print('Seam wrangler resolved gap in seam', s_count)

                    else:
                        Slice.seam_sets['unresolved gaps'] += [s_count]
                        print("Unresolved gaps in seam_wrangler")
                        print("Might distort some seams (but probably not)")

                    None_state = False
                    #for v in s['vps'][i]:
                        #ob.data.vertices[v].select = True

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

        if test_val is None:
            Slice.seam_sets['tri_means'] = np.append(Slice.seam_sets['tri_means'], s['tri_means'], axis=0)
        # iterate tick -----------
        s_count += 1



def build_data(Slice, test_val):
    """Generate meshes and such"""

    ob = Slice.ob
    flat_co = Slice.flat_co
    cloth_co = Slice.cloth_co

    # -------------------------
    s_count = 0

    test1 = True
    if test_val is None:
        test1 = False

    if not test1:
        Slice.seam_sets['mega_tri_mesh'] = {'verts': [], 'edges':[], 'faces': []}
        Slice.seam_sets['ed_offset'] = 0
        Slice.seam_sets['fa_offset'] = 0
        Slice.seam_sets['cloth_key'] = Slice.cloth_key
        Slice.seam_sets['ob'] = Slice.ob
        Slice.seam_sets['springs'] = np.empty((0,2), dtype=np.int32)
        Slice.seam_sets['sp_offset'] = 0
        Slice.seam_sets['dists'] = np.array([], dtype=np.float32)
        Slice.seam_sets['vp_means'] = np.empty((0,3), dtype=np.float32)
        Slice.seam_sets['with_z'] = np.empty((0,3), dtype=np.float32)
        Slice.seam_sets['complete_ravel'] = np.array([], dtype=np.bool)
        Slice.seam_sets['good_ravel'] = np.array([], dtype=np.bool)
        Slice.seam_sets['weights'] = np.empty((0,3), dtype=np.float32)
        Slice.seam_sets['tris'] = np.empty((0,3,3), dtype=np.float32)
        Slice.seam_sets['cloth_co'] = Slice.cloth_co
        Slice.seam_sets['tri_tiler'] = []
        Slice.seam_sets['tri_tiler_tick'] = 0
        Slice.seam_sets['vps'] = []


    for i in range(Slice.count):
        s = Slice.seam_sets[i]

        # build triangles for the mesh
        s['tris'] = create_triangles(Slice, s_count)
        # add z values to xys

        ed, fa = create_mesh_data(Slice, s_count)
        es = ed.shape
        ed.shape = (es[0] * 3, 2)

        fs = fa.shape
        fa.shape = (fs[0] * 3, 4)

        # only run on test num if using a number
        test_num = s_count
        if test1:
            test_num = test_val

        if s_count == test_num:

            # create the mesh or merge lists to make one mesh
            ts = s['tris'].shape
            s['tris'].shape = (ts[0] * 3, 3)

            if test1:
                if "sw_tris_" + str(s_count) not in bpy.data.objects:
                    tri_mesh = link_mesh(s['tris'].tolist(), ed.tolist(), fa.tolist(), "sw_tris_" + str(s_count))
                    reset_shapes(tri_mesh)
                    tri_mesh.hide_render = True
                tri_mesh = bpy.data.objects["sw_tris_" + str(s_count)]

                tri_co_start = get_co_shape(tri_mesh, 'MC_source')
                t_shape = tri_co_start.shape
                tri_co_start.shape = (t_shape[0]//3, 3, 3)
                s['tri_co_start'] = tri_co_start

                s['tri_mesh_ob'] = bpy.data.objects["sw_tris_" + str(s_count)]
                s['ob'] = Slice.ob

            else:
                # join into one mesh
                Slice.seam_sets['mega_tri_mesh']['verts'] += s['tris'].tolist()

                ed_tick = ed[-1][0] + 1
                ed += Slice.seam_sets['ed_offset']
                Slice.seam_sets['mega_tri_mesh']['edges'] += ed.tolist()
                Slice.seam_sets['ed_offset'] += ed_tick

                fa_tick = fa[-1][-1] +1
                fa += Slice.seam_sets['fa_offset']
                Slice.seam_sets['mega_tri_mesh']['faces'] += fa.tolist()
                Slice.seam_sets['fa_offset'] += fa_tick

                tick_tiler = np.array(s['tri_tiler'], dtype=np.int32) + Slice.seam_sets['tri_tiler_tick']
                Slice.seam_sets['tri_tiler'] += tick_tiler.tolist()

                Slice.seam_sets['tri_tiler_tick'] += s['tri_tick']
                # worst but ever! Four days to find !!! Slice.seam_sets['tri_tiler_tick'] = tick_tiler[-1] + 1

                # join vps
                Slice.seam_sets['vps'] += s['vps']

            # iterators ----------------------------------
            tridex = np.array([0, 1, 2], dtype=np.int32)
            void_tris = []
            vpm = []

            a = 0
            for v in s['vps']:
                idx = np.array(v, dtype=np.int32)

                if v.shape[0] == 0:
                    m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    void_tris += [tridex]

                else:
                    m = np.mean(cloth_co[idx], axis=0)

                vpm += [m]

                a += 3
                tridex = [a, a+1, a+2]

            s['vp_means'] = np.array(vpm, dtype=np.float32)
            s['void_tris'] = np.array(void_tris, dtype=np.float32)
            s['dummies'] = s['void_tris'].shape[0] > 0

            # get nearby means for dummies
            # dummies only get moved to near means once.
            if s['dummies']:
                overwrite = []
                m_count = s['vp_means'].shape[0]
                for m in range(s['vp_means'].shape[0]):
                    mean = s['vp_means'][m]
                    bool = s['complete'][m]
                    if bool:
                        good = mean
                    if not bool:
                        overwrite.append(m)
                    if bool:
                        if len(overwrite) > 0:
                            for ov in overwrite:
                                if np.sum(s['vp_means'][ov]) == 0:
                                    s['vp_means'][ov] = good

                            overwrite = []
                    if m +1 == m_count:
                        for ov in overwrite:
                            if np.sum(s['vp_means'][ov]) == 0:
                                s['vp_means'][ov] = good

            s['complete_ravel'] = np.repeat(s['complete'], 9) # for indexing at cloth.co
            s['good_ravel'] = np.repeat(s['good_bool'], 9) # for indexing at cloth.co
            s['cloth_key'] = Slice.cloth_key
            s['cloth_co'] = Slice.cloth_co

            if not test1:
                Slice.seam_sets['complete_ravel'] = np.append(Slice.seam_sets['complete_ravel'], s['complete_ravel'])
                Slice.seam_sets['good_ravel'] = np.append(Slice.seam_sets['good_ravel'], s['good_ravel'])
                Slice.seam_sets['vp_means'] = np.append(Slice.seam_sets['vp_means'], s['vp_means'], axis=0)

            # other data functions ----------------
            start_a = 0 # offset the triangle spring index
            if not test1:
                start_a = Slice.seam_sets['sp_offset']
            sp, di = generate_external_springs(s)

            if not test1:
                sp[:,0] += Slice.seam_sets['sp_offset']
                sp_tick = Slice.seam_sets['mega_tri_mesh']['edges'][-1][0] + 1
                Slice.seam_sets['sp_offset'] = sp_tick
                Slice.seam_sets['springs'] = np.append(Slice.seam_sets['springs'], sp, axis=0)
                Slice.seam_sets['dists'] = np.append(Slice.seam_sets['dists'], di)

            w = barycentric_weights(s)
            if not test1:
                Slice.seam_sets['weights'] = np.append(Slice.seam_sets['weights'], w, axis=0)
                Slice.seam_sets['with_z'] = np.append(Slice.seam_sets['with_z'], s['with_z'], axis=0)

            # test plot
            if False:
                weight_plot(s)

            move = False
            if move:
                move_tris(s)

        if not test1:
            Slice.seam_sets['tris'] = np.append(Slice.seam_sets['tris'], s['tris'], axis=0)

        # iterate tick -----------
        s_count += 1

    if not test1:
        if 'mega_tri_mesh' not in bpy.data.objects:
            M = Slice.seam_sets['mega_tri_mesh']
            v = np.array(M['verts'])
            any_nan = np.any(np.isnan(v))
            any_inf = np.any(np.isinf(v))
            b_log(['nans or infs when tri mesh is created', any_nan, any_inf])

            tri_mesh = link_mesh(v.tolist(), M['edges'], M['faces'], 'mega_tri_mesh')
            reset_shapes(tri_mesh)

            tri_mesh.hide_render = True
        tri_mesh = bpy.data.objects['mega_tri_mesh']
        Slice.seam_sets['tri_mesh_ob'] = bpy.data.objects['mega_tri_mesh']

        tri_co = get_co_shape(tri_mesh, 'MC_source')
        t_shape = tri_co.shape
        tri_co.shape = (t_shape[0]//3, 3, 3)
        Slice.seam_sets['tri_co_start'] = tri_co
        Slice.seam_sets['vps'] = np.array(np.hstack(Slice.seam_sets['vps']), dtype=np.int32)

    return Slice.seam_sets, test_val


def generate_external_springs(s):
    # index arrays...
    sp = []
    xys = []

    a = 0
    for i in range(len(s['vps'])):
        tridex = [a, a + 1, a + 2]
        if len(s['vps'][i]) > 0:
            for j in range(len(s['vps'][i])):
                v = s['vps'][i][j]
                xy = s['xys'][i][j]
                for t in tridex:
                    sp.append([t, v])
                    xys.append(xy)
        # ticker --------------------
        a += 3

    xy_co = np.array(xys, dtype=np.float32)
    springs = np.array(sp, dtype=np.int32)

    ls = springs[:, 0]
    vecs = xy_co - s['tris'][ls][:,:2]
    dist = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
    s['dists'] = dist
    s['springs'] = springs
    return springs, dist


def barycentric_weights(s):
    """Get the barycentric weights for
    moving the seams to the triangles."""
    shape = s['tris'].shape
    s['tris'].shape = (shape[0]//3 ,3 ,3)

    t_count = len(s['tri_tiler'])
    tiled = s['tris'][s['tri_tiler']]
    s['tiled'] = tiled
    # add z values before gettig weights.
    new = np.zeros(t_count * 3, dtype=np.float32)
    new.shape = (t_count, 3)
    new[:, :2] += s['good_xys']
    z = s['cum_dst']
    new[:, 2] += z[s['tri_tiler']]

    s['weights'] = get_weights(tiled, new)
    s['with_z'] = new # for xy scale adjustment
    return s['weights']


def weight_plot(s):
    """Use trianlges to plot the slices from bary weights"""
    tri_co = get_co_shape(s['tri_mesh_ob'], "MC_current")
    t_shape = tri_co.shape
    tri_co.shape = (t_shape[0]//3, 3, 3)
    tiled = tri_co[s['tri_tiler']]
    plot = np.sum(tiled * s['weights'][:,:,None], axis=1)


def move_tris(s):
    """Move tris with complete vps to
    location on garment"""

    tri_mesh = s['tri_mesh_ob']

    # move to means
    vecs = s['vp_means'] - s['tri_means']
    vecs[~s['complete']] *= 0 # Don't move where points are missing

    tri_shape = False
    if tri_shape:
        shape = s['tris'].shape
        s['tris'].shape = (shape[0]//3 ,3 ,3)

    moved = s['tris'] + vecs[:, None]
    key = tri_mesh.data.shape_keys.key_blocks['MC_current']
    key.data.foreach_set('co', moved.ravel())
    tri_mesh.data.update()


class Slices():
    pass

def generate_data(ob, test_val, cloth_key=None):

    #global data
    Slice = Slices()
    Slice.ob = ob
    ob_name = ob.name

    # setup functions
    slice_setup(Slice, cloth_key)
    missing_distance(Slice, test_val)
    data_set, test1 = build_data(Slice, test_val)
    test1 = test_val is not None

    if test1:
        data_set = data_set[test_val]

    # undo !!!!!!!!


    bpy.types.Scene.MC_seam_wrangler = True # so the continuous handler can auto_kill

    tris = data_set['tri_mesh_ob']
    bpy.context.view_layer.objects.active = tris
    tris.MC_props.cloth = True
    tris.MC_props.seam_wrangler = False

    #tris.MC_props.cache = True
    #tris.MC_props.cache_desktop = True

    if not test_cache:
        return data_set
    # put the file path in the text at the top
    read_python_script('play_cache_files.py')
    py_text = bpy.data.texts['play_cache_files.py']
    py_text.cursor_set(0)
    py_text.write('# ' + "cache file save at:")
    py_text.cursor_set(1)
    py_text.write('# ' + tris.MC_props.cache_folder)
    py_text.cursor_set(2)
    py_text.write('# ' + tris.MC_props.cache_name)

    lc = len(py_text.lines)
    py_text.cursor_set(lc)
    py_text.write('MC = bpy.data.objects["mega_tri_mesh"].MC_props' + '\n')
    py_text.write('MC.cache_desktop = ' + "True" + '\n')
    py_text.write('MC.cache_only =  True' + '\n')
    py_text.write('MC.play_cache =  True' + '\n')
    #py_text.write('MC["cache_folder"] = ' + 'str(os.path.expanduser("~/Desktop"))' + '\n')

    #py_text.write('MC.cache_name = ' + "'" + tris.MC_props.cache_name + "'" + '\n')
    #py_text.write('MC.cache_folder = '  +  "'" + tris.MC_props.cache_folder + "'" + '\n')

    py_text.write('MC = bpy.data.objects[' + "'" + ob_name + "'" + '].MC_props' + '\n')
    py_text.write('MC.cache_desktop = ' + "True" + '\n')
    py_text.write('MC.cache_only =  True' + '\n')
    py_text.write('MC.play_cache =  True' + '\n')
    py_text.write('MC.cache_name = ' + "'" + ob.MC_props.cache_name + "'" + '\n')
    #py_text.write('MC["cache_folder"] = ' + 'str(os.path.expanduser("~/Desktop"))' + '\n')
    #py_text.write('MC.cache_desktop = ' + "True" + '\n')

    #tris.MC_props.cache_only = True

    return data_set


def position_triangles(ob, test_val, debug=None, cloth_key=None):
    """Generates all the needed data using a number
    of functions and positions triangles along seams"""

    if 'mega_tri_mesh' not in bpy.data.objects:
        data = generate_data(ob, test_val, cloth_key)
        if debug == 2:
            return

        tris = data['tri_mesh_ob']
        data['run_type'] = 1 # will run seam forces for positioning

        # inital position:
        tris.MC_props.stretch_iters = 10
        bpy.context.scene.MC_props.delay = 1
        data['linear_iters'] = 1
        data['stretch_array'] = np.zeros(len(tris.data.vertices), dtype=np.float32)
        data['seam_influence'] = .5
        data['partial_set'] = [0,1,2,3,4,5]
        data['velocity'] = 0.5
        data['decay'] = 1
        data['vel_floor'] = 0.77
        data['run_frames'] = 7
        data['sw_count'] = 0
        data['count'] = 0
        if debug == 1:
            tris.MC_props.continuous = True
            return
        # For the pin vertex group
        setup_pin_group(ob, data['vps'])

        # have to run once to postion before we move the mesh object.
        bpy.types.Scene.seam_wrangler_data = data
        tris.MC_props.seam_wrangler = True # debug 3 runs in callback without manage seams

        return data


def manage_seams(ob, cloth_key=None, settings={}, test_val=None, debug=None):
    """Run in frame handers with a variety of settings"""
    # generate data on first call
    #position_triangles(ob, test_val, debug, cloth_key)

    if debug in [1, 2, 3]:
        return

    data = bpy.context.scene.seam_wrangler_data
    # default settings
    data['linear_iters'] =        1 # iterations of seams pulling on tris
    data['velocity'] =         0.77 # movement between run_frames
    data['run_frames'] =          1 # calculate every force this many times
    data['seam_influence'] =    1.0 # seams pulling on triangles
    data['iterations'] =         12 # solves the trianlges this many times every run frame (higher means stiffer but slower to solve)
    data['tri_force'] =           1 # forces the trinalgs towards their original shape (1 or less to avoid exploding)
    data['tri_influence'] =     0.321 # triangles forcing seams to targets
    data['xy_scale'] =    [1.0, 1.0]# multiply the xy values by this amount
    data['vis_mesh'] = False # create points where the seams are being moved to

    for key, value in settings.items():
        if key in data:
            data[key] = value
        #else:
            #print('!!! manage_seams setting did not match:', key, value, '!!!')

    data['run_type'] = 2 # will run seam forces for update
    tris = data['tri_mesh_ob']

    tris.MC_props.stretch_iters = data['iterations']
    tris.MC_props.seam_wrangler = True # setting the property how we trigger it to run


# !!! call this one to drop in the frame handler !!!
def seam_manager(Bobj=None, frames=None, kill_frame=None, settings=None, cloth_key=None):
    """Adds a frame handler that will
    force seams into submission at frames
    in the list."""

    global actual_frames
    actual_frames = np.hstack(frames)

    logs1 = ["!!! called seam manager !!!" for i in range(3)]
    logs1 += ['settings from arg line 974:', settings]
    logs1 += ['frames coming in from arg:', frames]

    try:
        from garments_render.simulation.MC_tools import register
        from garments_render.simulation.MC_tools import cache_only
        register()
        logs1 += ['Succesfully registered MC_tools']
    except:
        b_log(['Got an error when trying to register modeling cloth', 'Probably already registered'])

    # for running in blender --------------
    try:
        MC_tools = bpy.data.texts['MC_tools.py'].as_module()
        MC_tools.register()
    except:
        pass
    # -------------------------------------

    data['sw_count'] = 0
    data['count'] = 0
    data['ob'] = Bobj
    data['frames'] = actual_frames
    data['init_frame'] = min(actual_frames)
    data['kill_frame'] = max(actual_frames)

    logs1 += ['init frame:', data['init_frame']]
    logs1 += ['kill frame from arg:', kill_frame, 'overwriting kill frame with:', data['kill_frame']]

    if settings is None:
        b_log(['!!! Settings coming in from arg is None !!!' for i in range(3)])
        settings = [{},
                    {},
                    {},
                    {}]

    data['settings'] = settings

    bpy.types.Scene.seam_wrangler_data = {'init': False}
    for k, v in data.items():
        bpy.context.scene.seam_wrangler_data[k] = v

    b_log(logs1)
    
    # the function that runs every frame
    def seam_wrangler_anim(scene):
        data = bpy.context.scene.seam_wrangler_data
        f = bpy.context.scene.frame_current
        active_object = bpy.context.object
        
        if f == data['init_frame']:
            tri_data = position_triangles(Bobj, test_val=None, debug=None, cloth_key=None)
            for k, v in tri_data.items():
                data[k] = v
            data['init'] = True    
        
        if not data['init']:
            b_log(['init frame not hit yet'])
            return
        
        if f not in actual_frames:
            b_log(['seam_wrangler_anim running not in frames at frame:', f, 'skipping'])
            return
        
        ob = data['ob']

        fix_all_shape_key_nans(ob)
        if 'mega_tri_mesh' in bpy.data.objects:
            fix_all_shape_key_nans(bpy.data.objects['mega_tri_mesh'])

        sc = len(data['settings'])
        if sc == 0:
            settings = {}
        else:
            settings = data['settings'][-1]
            if data['count'] < sc:
                settings = data['settings'][data['count']]

        keys = ob.data.shape_keys.key_blocks
        
        active_keys = [i.name for i in keys if ((i.name != 'sw_view') & (i.value == 1))]
        if len(active_keys) == 0:
            b_log(['!!! No active shape key for seam manager !!!' for i in range(3)])

        val = 0.0

        if f in actual_frames:

            for mod in ob.modifiers: # cloth modifiers can come and go
                if mod.type == "CLOTH":
                    mod.settings.vertex_group_mass = 'SW_seam_pin'
                    mod.settings.pin_stiffness = 1.0

            print('seam_manager adjusted garment at frame: ',f)
            val = 0.3

        pidx = ob.vertex_groups['SW_seam_pin'].index
        for v in data['vps'].tolist():
            ob.vertex_groups['SW_seam_pin'].add([pidx, v], val, 'REPLACE')

        # a bug in blender always sets vert 0 to a weight of 1
        if 0 not in data['vps']:
            ob.vertex_groups['SW_seam_pin'].add([pidx, 0], 0.0, 'REPLACE')

        manage_seams(ob, cloth_key, settings=settings, test_val=None, debug=None)
        bpy.context.view_layer.objects.active = active_object

        if True:
            fix_all_shape_key_nans(ob)
            fix_all_shape_key_nans(bpy.data.objects['mega_tri_mesh'])

        if test_cache:
            cache_only(bpy.data.objects['mega_tri_mesh'], Bobj.MC_props.current_cache_frame)
            cache_only(Bobj)

        data['count'] += 1
        data['sw_count'] += 1

        if f == data['kill_frame']:
            # clean dead versions of the animated handler
            b_log(['Hit the kill frame', f, "deleting bpy.context.scene.seam_wrangler_data"])
            handler_names = np.array([i.__name__ for i in bpy.app.handlers.frame_change_post])
            booly = [i == 'seam_wrangler_anim' for i in handler_names]
            idx = np.arange(handler_names.shape[0])
            idx_to_kill = idx[booly]
            for i in idx_to_kill[::-1]:
                del(bpy.app.handlers.frame_change_post[i])
                b_log(["deleted handler ", i])

            # clean up pin group
            pidx = ob.vertex_groups['SW_seam_pin'].index
            for v in data['vps'].tolist(): # without tolist() vertex group add breaks
                ob.vertex_groups['SW_seam_pin'].add([pidx, v], 0.0, 'REPLACE')

            for mod in ob.modifiers:
                if mod.type == "CLOTH":
                    mod.settings.vertex_group_mass = ''

            bpy.data.objects['mega_tri_mesh'].name = 'used_tris'
            del(bpy.types.Scene.seam_wrangler_data)

    handler_names = np.array([i.__name__ for i in bpy.app.handlers.frame_change_post])
    booly = [i == 'seam_wrangler_anim' for i in handler_names]
    if not np.any(booly):
        bpy.app.handlers.frame_change_post.append(seam_wrangler_anim)


def test():

    # testing
    # get the active state
    active_object = bpy.context.object


    ob = bpy.data.objects['g6774']
    #if False:
    if 'seam_wrangler_data' in dir(bpy.context.scene):
        del(bpy.types.Scene.seam_wrangler_data)

    reset_shape = get_co_shape(ob, 'save')
    ob.data.shape_keys.key_blocks['CLOTH1153'].data.foreach_set('co', reset_shape.ravel())
    ob.data.update()

    # generate data if needed

    # debug None: Run normal. Create data and position if needed, run seams
    # debug 1: Update in viewport with handler. Position only
    # debug 2: Just generate data. No meshes
    # debug 3: Just run position
    # test_val will run only one seam.

    td1 = {'run_frames':1, 'xy_scale': [2.1, 2.1]}
    td2 = {'run_frames':1, 'xy_scale': [1.0, 1.0]}
    td3 = {'run_frames':1, 'xy_scale': [1.0, 1.0], 'vis_mesh':True}

    #manage_seams(ob, settings=td2, test_val=None, debug=None)

    # restore the active state
    bpy.context.view_layer.objects.active = active_objec


# undo !!!!!!!!!!
#try:
#    del(bpy.types.Scene.seam_wrangler_data)
#except:
#    pass
#Bobj = bpy.data.objects['g7442']
#seam_manager(Bobj=Bobj, frames=[167,2,3], kill_frame=170, settings=None, cloth_key=None)

#position_triangles(Bobj, test_val=None, debug=None, cloth_key=None)
    
if False:    
    try:    
        del(bpy.types.Scene.seam_wrangler_data)
    except:
        pass
    Bobj = bpy.data.objects['g7442']
    seam_manager(Bobj=Bobj, frames=[167,168,169], kill_frame=170, settings=None, cloth_key=None)
