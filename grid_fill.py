import bpy
import bmesh
import numpy as np
from numpy import newaxis as nax


def get_co(mesh):
    v_count = len(mesh.vertices)
    co = np.zeros(v_count * 3, dtype=np.float32)
    mesh.vertices.foreach_get('co', co)
    co.shape = (v_count, 3)    
    return co


def get_volume_data(volume_object):
    co = get_co(volume_object.data)
    min_corner = np.min(co, axis=0)
    max_corner = np.max(co, axis=0)
    return min_corner, max_corner


def create_array(volume_object, resolution, override=False):
    min, max = get_volume_data(volume_object)
    dimensions = max - min
    counts = np.array(dimensions//np.array(resolution), dtype=np.int32)
    
    x_lin = np.linspace(min[0], max[0], counts[0])
    y_lin = np.linspace(min[1], max[1], counts[1])
    z_lin = np.linspace(min[2], max[2], counts[2])

    # can use it in 2d space or fill a 3d volume    
    if override:
        counts[2] = 1
        z_lin = np.array([min[2]])

    x1 = np.tile(x_lin, counts[1])
    x2 = np.tile(x1, counts[2])
    
    y1 = np.tile(y_lin, counts[0])
    y2 = np.tile(y1, counts[2])    
    y2.shape = (counts[0], counts[2] * counts[1])
    y2 = y2.T.ravel()
    
    z1 = np.tile(z_lin, counts[0])
    z2 = np.tile(z1, counts[1])    
    z2.shape = (counts[0], counts[1], counts[2])
    z2 = z2.T.ravel()
    
    total = np.product(counts)
    vec_array = np.zeros(total * 3, dtype=np.float32)
    vec_array.shape = (total, 3)

    vec_array[:,0] = x2
    vec_array[:,1] = y2
    vec_array[:,2] = z2
    
    return vec_array


def create_point_mesh(vec_array):
    mesh = bpy.data.meshes.new("points")
    mesh.from_pydata(vec_array, [], [])
    mesh.update()

    points = bpy.data.objects.new("points", mesh)
    bpy.context.collection.objects.link(points)
    

def append_values_to_array(vec_array, val):
    new_array = np.zeros(vec_array.shape[0] * 4, dtype=np.float32)
    new_array.shape = (vec_array.shape[0], 4)
    new_array[:, :3] = vec_array
    new_array[:, 3] = val
    return new_array

# end volume functions ------------------------------------


# begin edge loop functions -------------------------------

def get_proxy_eidx(ob, eval=True):
    """Get the edge indices as an Nx2 numpy array
    for the object with modifiers."""
    evob = ob
    if eval:    
        dg = bpy.context.evaluated_depsgraph_get()
        evob = ob.evaluated_get(dg)
    e_count = len(evob.data.edges)
    eidx = np.zeros(e_count * 2, dtype=np.int32)
    evob.data.edges.foreach_get('vertices', eidx)
    eidx.shape = (e_count, 2)    
    return eidx
    

def get_proxy_co(ob, co=None, eval=True):
    """Get coordiates with modifiers added.
    Can supply the array to avoid allocating the memory"""
    evob = ob
    if eval:    
        dg = bpy.context.evaluated_depsgraph_get()
        evob = ob.evaluated_get(dg)
    dg = bpy.context.evaluated_depsgraph_get()
    evob = ob.evaluated_get(dg)
    v_count = len(evob.data.vertices)
    if co is None:    
        co = np.zeros(v_count * 3, dtype=np.float32)
    evob.data.vertices.foreach_get('co', co)
    co.shape = (v_count, 3)
    return co


def get_bmesh(ob):
    obm = bmesh.new()
    if ob.mode == 'OBJECT':
        obm.from_mesh(ob.data)
    elif ob.mode == 'EDIT':
        obm = bmesh.from_edit_mesh(ob.data)
    return obm


def slide_points_to_plane(e1, e2, origin, normal=np.array([0,0,1])):
    '''Takes the start and end of a set of edges as Nx3 vector sets
    Returns where they intersect the plane with a bool array for the
    edges that pass through the plane'''
    e_vecs = e2 - e1
    e1or = e1 - origin
    edge_dots = np.einsum('j,ij->i', normal, e_vecs)
    dots = np.einsum('j,ij->i', normal, e1or)
    scale = dots / edge_dots  
    drop = (e1or - e_vecs * np.expand_dims(scale, axis=1)) + origin
    intersect = (scale < 0) & (scale > -1)
    return drop, intersect, scale


def measure_angle_at_each_vert(grid):
    """Provide mesh and anlge limit in degrees.
    Returns the indices of verts that are sharper than limit"""

    limit = np.cos(grid.angle_limit * (np.pi/180))
    eidx = grid.eidx
    co = grid.co
    order = grid.order
    
    ls = np.roll(order, 1)
    rs = np.roll(order, -1)
    
    v1 = co[ls] - grid.co_order
    v2 = co[rs] - grid.co_order
    
    # use the vecs pointing away from each vertex later
    grid.vls = v1
    grid.vrs = v2
    
    ls_dots = np.einsum('ij, ij->i', v1, v1)
    rs_dots = np.einsum('ij, ij->i', v2, v2)
    
    uv1 = v1 / np.sqrt(ls_dots)[:, nax]
    uv2 = v2 / np.sqrt(rs_dots)[:, nax]
    
    # used by a bunch of other functions later
    grid.uvls = uv1
    grid.uvrs = uv2
    
    angle = np.einsum('ij, ij->i', uv1, uv2)
    sharps = angle > -limit   
    
    return order[sharps]


def loop_order(ob):
    """takes an object consisting of a single loop of edges and gives the order"""
    obm = get_bmesh(ob)
    obm.edges.ensure_lookup_table()
    obm.verts.ensure_lookup_table()
    
    v = obm.verts[0]
    new_v = v.link_edges[0].other_vert(v)
    order = [0]
    last_v = v
    
    for i in range(len(obm.verts) - 1):
        other_vert = np.array([ed.other_vert(v).index for ed in v.link_edges])        
        not_v = other_vert[other_vert != last_v.index][0]
        order.append(not_v)
        last_v = v
        v = obm.verts[not_v]

    return np.array(order)


def get_segments(grid):
    """Generate a list of segments between sharp edges"""
    ob = grid.ob
    sharps = grid.sharps

    obm = get_bmesh(ob)
    obm.edges.ensure_lookup_table()
    obm.verts.ensure_lookup_table()
    
    # in case there are no sharps    
    if len(sharps) == 0:
        sharps = np.array([0])

    count = 0
    # start with the first sharp:
    sharp = sharps[0]
    v = obm.verts[sharp]
    other_verts = np.array([ed.other_vert(v).index for ed in v.link_edges])
    move = obm.verts[other_verts[0]]

    seg = [sharp]
    segs = []
    for i in range(len(sharps)):
        while True:    
            if move.index in sharps:
                seg.append(move.index)
                segs.append(seg)
                seg = []
            
            seg.append(move.index)
            other_verts = np.array([ed.other_vert(move).index for ed in move.link_edges])
            new = other_verts[other_verts != v.index][0]
            v = move
            move = obm.verts[new]
            
            if move.index == sharp:
                seg.append(move.index)
                segs.append(seg)
                return segs

        count +=1

        if count > grid.v_count:    
            print("ACK!!!!!! We almost got stuck in an infinite loop! Oh the humanity!")
            return segs


def get_seg_length(grid, seg):
    """returns the total length of a set
    of points that are in linear order"""
    co = grid.co
    vecs = co[seg[1:]] - co[seg[:-1]]
    grid.seg_vecs.append(vecs) # might as well save this for later
    seg_length = np.sqrt(np.einsum('ij, ij->i', vecs, vecs))
    grid.seg_lengths.append(seg_length) # saving this also
    total_length = np.sum(np.sqrt(np.einsum('ij, ij->i', vecs, vecs)))

    return total_length


def generate_perimeter(grid, current):
    """Place points around perimeter"""

    # get the length of each segments
    seg_lengths = np.array([get_seg_length(grid, s) for s in grid.segments])
    
    grid.point_counts = seg_lengths // grid.size

    # have to decide where to transition between splitting
    #   in half and doing just one point
    grid.point_counts[grid.size / seg_lengths > 0.5] = 0        
    
    
    grid.spacing = seg_lengths / grid.point_counts
    
    #current = 19
    
    # doing one example:    
    seg = grid.segments[current]
    seg_len = grid.seg_lengths[current]    
    
    # add the first point in the segment (second one gets added next time)   
    seg_sets = np.empty((0,3), dtype=np.float32)
    # 
    seg_sets = move_point_on_path(grid, current, seg_sets)

    return seg_sets
    

def move_point_on_path(grid, idx, seg_sets):
    """Walk the points until we are at the distance
    we space them"""
    
    co = grid.co
    seg = grid.segments[idx]
    lengths = grid.seg_lengths[idx]
    spacing = grid.spacing[idx]
    vecs = grid.seg_vecs[idx]
    count = grid.point_counts[idx]

    seg_co_set = [co[seg[0]]] # the last one will be filled in by the first one next time.
    if count == 0:
        return seg_co_set

    growing_length = 0
    len_idx = 0
    build = spacing
    
    counter = 0
    for x in range(int(count) - 1):    
        growing_length = 0
        len_idx = 0
        counter += 1
        while growing_length < spacing:
            growing_length += lengths[len_idx]
            len_idx += 1    
            
        # back up to the last point now 
        len_idx -= 1
        growing_length -= lengths[len_idx] 
        point = co[len_idx]
        
        # move from the past point along the last vector until we
        # hit the proper spacing
        end_offset = spacing - growing_length 
        last_dif = lengths[len_idx] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!  
        along_last = end_offset / last_dif
        
        move = vecs[len_idx]
        
        loc = co[seg[len_idx]] + move * along_last
        
        seg_co_set.append(loc)
        
        # start from the beginning because it's easier
        spacing += build
    
    # join to master array:
    seg_sets = np.append(seg_sets, seg_co_set, axis=0) 

    return seg_sets


def get_direction(grid, angle_limit=45):
    """Get the direction to move towards the inside
    of the shape at each vert around the edges."""
    
    print("--------------------------------------")
    print()
    
    #Z = np.array([0, 0, 1], dtype=np.float32)
    
    # get average normal of whole shape
    norms = np.cross(grid.vls, grid.vrs)
    av_norm = np.sum(norms, axis=0)
    
    # this normal gets dropped into the gengrid class
    grid.av_norm = av_norm 
    
    # plot it to an empty to test
    bpy.data.objects['ee'].location = av_norm
    
    # use the angle at each vert to determine how to plot points
    #   If more than 45 we 
    # get the angle at each
    
    # doing the infinite plane thing
    # get the perp from the two vectors around each vert
    perp = grid.uvls - grid.uvrs
    



    # check all the edges to see if they pass through the infinite
    #   plane defined by that perp vec with the point as the origin.
    
    # find the point of intersection for edges that pass through the plane
    
    
    
    
    
    """
    Weird thought while praying...
    get a vec that would go between the unit vecs of the left
    and right side at each point.
    Use that to define an infinite plane originating at the point
    or just do a closest point on edge calc to find points that are close by
    
    For points where the ls and rs are parallel, the direction could be
    where the nearest point intersects the infinite plane.
    
    Could smooth at each iteration
    Could use somthing like the cloth engine to pull points
    towards a target connected edge length. 
    """
    
    
    # For now... Fill with points
    
    # get bounds
    
    
    
    
    # get vecs pointing away from each vert
    
    
    
    
    # if the cross is zero use the z to get the perp
    # if 
    
    
    
    # get bounding box of entire shape
    
    
    # create a point outside of that for 2d intersection check
    
    
    # 
    
    # if the two vecs are paralell the cross will be zero
    # If I'm planning to use UV mapping I can use the z axis
    

def gengrid_angle(grid):
    """Provide mesh and anlge limit in degrees.
    Returns the indices of verts that are sharper than limit"""

    limit = np.cos(grid.angle_limit * (np.pi/180))
    eidx = grid.eidx
    co = grid.co
    
    ls = np.roll(co, 1, axis=0)
    rs = np.roll(co, -1, axis=0)
    
    v1 = ls - co
    v2 = rs - co
    
    # use the vecs pointing away from each vertex later
    grid.vls = v1
    grid.vrs = v2
    
    ls_dots = np.einsum('ij, ij->i', v1, v1)
    rs_dots = np.einsum('ij, ij->i', v2, v2)
    
    uv1 = v1 / np.sqrt(ls_dots)[:, nax]
    uv2 = v2 / np.sqrt(rs_dots)[:, nax]
    
    # used by a bunch of other functions later
    grid.uvls = uv1
    grid.uvrs = uv2
    
    angle = np.einsum('ij, ij->i', uv1, uv2)
    sharps = angle > -limit   
    
    return sharps    


def mag_set(mag, v2):    
    '''Applys the magnitude of v1 to v2'''
    d1 = mag ** 2
    d2 = v2 @ v2
    div = d1/d2
    return v2 * np.sqrt(div)


def gen_grid_project(grid):
    
    new_points = np.empty((0,3), dtype=np.float32)
    
    #idx = 6 # edge and origin matches
    
    co = grid.co
    shape = co.shape[0]
        
    e1 = co[grid.eidx[:, 0]]
    e2 = co[grid.eidx[:, 1]]
    
    perps = grid.uvls - grid.uvrs
    
    # iterate throuhg all points:
    for idx in range(shape):
        # remove edges connected to point if needed
        plane = slide_points_to_plane(e1, e2, co[idx], perps[idx])
        # !!! might work better to do cpoe back to origial perp
        #   vec with vec centered on the origin.
        inter = plane[0][plane[1]]
        print(inter, "this is the intersections")
        
        # eliminate points that are outside of the shape (not sure how to do that with a 3d grid...)


        # find the closest intersection for getting the vec direction
        vecs = inter - co[idx]
        dots = np.einsum('ij, ij->i', vecs, vecs)
        min = np.argmin(dots)
        vec = vecs[min]
        plot = co[idx] + mag_set(grid.size, vec)
        new_points = np.append(new_points, plot[nax], axis=0)

    bpy.data.objects['se'].location = inter[0]
    #bpy.data.objects['se2'].location = inter[1]

    
    return new_points

    #print(np.arange(shape)[plane[1]])
    #print(inter, "this is intersections")
    
    

    

class Grid(object):
    """The outline we start with"""
    pass


class GenGrid(object):
    """The generated grid"""
    pass
    

def main(ob):
    # walk around the edges and and plot evenly spaced points
    # respecting the sharpness of the angle
    M = ob.matrix_world.copy()
    obl = ob.location
    obr = ob.rotation_quaternion
    obre = ob.rotation_euler
    obs = ob.scale

    dg = bpy.context.evaluated_depsgraph_get()
    evob = ob.evaluated_get(dg)    
    
    grid = Grid()
    grid.ob = evob
    grid.co = get_proxy_co(evob, None, eval=False)
    grid.angle_limit = 30
    grid.eidx = get_proxy_eidx(evob, eval=False)
    grid.order = loop_order(evob)
    grid.co_order = grid.co[grid.order]
    grid.sharps = measure_angle_at_each_vert(grid)
    grid.segments = get_segments(grid)
    grid.size = 0.03
    grid.seg_vecs = [] # gets filled by the function below
    grid.seg_lengths = [] # gets filled by the function below
    iters = len(grid.segments)

    # generated grid
    gengrid = GenGrid()
    #gengrid.av_norm = grid.av_norm
    gengrid.co = np.empty((0,3), dtype=np.float32)    
    gengrid.angle_limit = 45
    gengrid.size = grid.size
    
    # create points for every segment between sharps --------------------
    for i in range(iters):
        x = generate_perimeter(grid, i)
        gengrid.co = np.append(gengrid.co, x, axis=0)
    # -------------------------------------------------------------------
    
    # create edges
    e_count = gengrid.co.shape[0]
    e1 = np.arange(e_count, dtype=np.int32)
    e2 = np.roll(e1, -1)
    gen_edges = np.append(e1[:,nax], e2[:,nax], axis=1)
    gengrid.eidx = gen_edges
    # get gengrid angles
    gengrid.angles = gengrid_angle(gengrid)    

    if False:    
        project = gen_grid_project(gengrid)
        gengrid.co = np.append(gengrid.co, project, axis=0)
        # create the grid ---------------------------------------------------
        # creates a grid of points. Currently not used
        grid_fill = True
        grid_fill = False
        if grid_fill:    
            resolution = np.array([grid.size, grid.size, 1], dtype = np.float32)
            vec_array = create_array(ob, resolution, override=True)
            gengrid.co = np.append(gengrid.co, vec_array, axis=0)
        #create_point_mesh(vec_array)
        # create the grid ---------------------------------------------------
        
        # begin filling in the grid

        get_direction(grid)
        
        # temporary return !!!
        #return
    

    create = False
    create = True
    if create:
        mesh = bpy.data.meshes.new('gen_grid')
        #print(v_locs)
        #for i in x_edges:
        #print(i, "this is x edges-----------")
        
        mesh.from_pydata(gengrid.co.tolist(), edges=gen_edges.tolist(), faces=[])
        #mesh.from_pydata(v_locs, edges=[[0,1],[1,2]], faces=[])
        mesh.update()
        
        #if False:    
        grid_ob = bpy.data.objects.new('gen_grid', mesh)
        bpy.context.collection.objects.link(grid_ob)
        
        #M = ob.matrix_world.to_translation()
        grid.ob.matrix_world = ob.matrix_world

        grid_ob.location = obl
        grid_ob.rotation_quaternion = obr
        grid_ob.rotation_euler = obre
        grid_ob.scale = obs



#ob = bpy.data.objects['spaced']
#ob = bpy.data.objects['start_grid']
#ob = bpy.data.objects['Circle']
ob = bpy.context.object
main(ob)



# notes:
"""
Probably need to have a distance threshold for 
the minimum length between sharps so I don't end up
with a million little tiny segments if someone does
something like a zigzag (might be working with procedural
geometry that has those kinds of artifacts.)
"""

# features:
# using it on a crappy topo terrain for example you could collide or shrinkwrap
#   to the crappy topo and smooth

# by uv unrwapping like with the uv shape tool you could grid fill a 3d curve
# could use the option of non-manifold edges so you could grid fill if there
#   is already a fill.
# need to be able to convert a real curve to a grid
# make the grid on top as a separate object.

# one possible approach, would be to project all the way to 
#   the edge that intersects and plot points close to the
#   grid size. 

# use the infinite plane thing to know which way to project
#   towards the inside of the grid. 
# Project these verts then form a row of faces. 
# Now we can do a uv shape key (can only unwrap faces)

# the angle of the edges seems irrelevant because
#   it could be a sharp angle but the plotted point
#   would be perp to both because of where the plane
#   intersects another edge. Like in the tennis ball
#   pattern in 3d space. 
#   Maybe just do a distance check for nearby points

# There could be more than one solution for the direction
#   to plot points based on the infinite plane
#   Could use a nearest intersection or compare nearby projections
#   or projections where there is one clear choice based on distance

"""features """
# !!! For something like text which already has faces
#   could do a non-manifold in edge mode to get perimeter
#   or just pull out the verts that are part of edges with
#   between zero and one link face...

# !!! add an edge smooth feature that respects sharps not smoothing them
#       There is an edge smooth feature in one of the dental tools
