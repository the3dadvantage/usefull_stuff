# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:39:26 2015

@author: rich.colburn
"""
import bpy
import rotate_methods
import mesh_info
import numpy as np

def get_triangle_normals(tr_co):    
    '''Takes N x 3 x 3 set of 3d triangles and 
    returns non-unit normals and origins'''    
    origins = tr_co[:,0]
    expanded = np.expand_dims(origins, axis=1)
    cross_vecs = tr_co[:,1:] - expanded
    return np.cross(cross_vecs[:,0], cross_vecs[:,1]), origins

def get_face_normals(tri_coords):
    '''does the same as get_triangle_normals 
    but I need to compare their speed'''    
    t0 = tri_coords[:, 0]
    t1 = tri_coords[:, 1]
    t2 = tri_coords[:, 2]
    return np.cross(t1 - t0, t2 - t0)

def edge_edge_intersect_2d(a1,a2, b1,b2, intersect=False):
    '''simple 2d line intersect'''    
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = da[::-1] * np.array([1,-1])
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    scale = (num / denom)
    if intersect:
        return b1 + db * scale, (scale > 0) & (scale < 1)
    else:
        return b1 + db * scale

def edge_edges_intersect_2d(edges1, edges2, vec1, vec2):
    '''Vector intersecting multiple edges in 2d
    requires N x 2 sets of edges and xy for vec1 and vec2.
    Returns the locations where intersections occur inside
    both the line and the edges and the bool arrays for each'''    
    e_vecs = edges2 - edges1
    vec = vec2 - vec1
    vec_ori = vec1 - edges1
    vec_perp = vec[[1, 0]] * np.array([1,-1])    
    denom = np.einsum('j,ij->i', vec_perp, e_vecs)
    num = np.einsum('j,ij->i', vec_perp, vec_ori)
    scale = (num / denom)
    scale_bool = (scale > 0) & (scale < 1)
    locations = edges1[scale_bool] + e_vecs[scale_bool] * np.expand_dims(scale[scale_bool], axis=1)
    dif = locations - vec1
    lvd = np.einsum('j,ij->i', vec, dif)
    d = np.dot(vec, vec)
    div = lvd / d
    check2 = (div > 0) & (div < 1)
    return locations[check2], scale_bool, check2

def cpoe(p, e1, e2):
    ev = e2 - e1
    pv = p - e1
    d = np.dot(ev, pv) / np.dot(ev,ev)
    cp = e1 + ev * d
    return cp, d

def barycentric_remap(p, s, t):
    '''Takes the position of the point relative to the first triangle
    and returns the point relative to the second triangle.
    p, s, t = point, source tri, target tri'''
    cp1, d1 = cpoe(p, s[0], s[1])
    t1 = (t[1] - t[0]) * d1
    cp2, d2 = cpoe(p, s[0], s[2])
    t2 = (t[2] - t[0]) * d2
    cp3, d3 = cpoe(p, cp1, cp2)
    t3 = (t2 - t1) * d3
    cp4, d4 = cpoe(s[0], cp1, cp2)
    t4 = (t2 - t1) * d4
    cp5, d5 = cpoe(p, s[0], cp4 )
    t5 = (t1 + t4) * d5 
    perp = (t5) - (t1 + t4) 
    return t[0] + t1 + t3 + perp


# !!! hey! I might be able to use this for edge intersections in the cloth engine.
def edge_to_edge(e1, e2, e3, e4 ):
    '''Takes two edges defined by four vectors.
    Returns the two points that describe the shortest
    distance between the two edges. The two points comprise
    a segment that is orthagonal to both edges. Think of it
    as an edge intersect.'''
    v1 = e2 - e1
    v2 = e3 - e4
    v3 = e3 - e1
    cross = np.cross(v1, v2)
    d = np.dot(v3, cross) / np.dot(cross, cross)
    spit = cross * d
    cp1 = e1 + spit # spit because if you stand on cp1 and spit this is where it lands.
    vec2 = cp1 - e3
    d = np.dot(vec2, v2) / np.dot(v2, v2)
    nor = v2 * d
    cp2 = e3 + nor 
    normal = cp1 - cp2
    or_vec = e1 - cp2
    e_dot = np.dot(normal, v1)
    e_n_dot = np.dot(normal, or_vec)
    scale = e_n_dot / e_dot  
    p_on_p =  (or_vec - v1 * scale) + cp2
    return p_on_p, p_on_p + spit

def edge_to_edges(coords, e1, e2, edges_idx):
    '''Takes an edge and finds the vectors orthagonal to it
    and a set of edges.'''
    ec = coords[edges_idx]
    e3 = ec[:,0]
    e4 = ec[:,1]
    v1 = e2 - e1
    v2 = e4 - e3
    v3 = e3 - e1
    cross = np.cross(v1, v2)
    d = np.einsum('ij,ij->i', v3, cross) / np.einsum('ij,ij->i', cross, cross)
    spit = cross * np.expand_dims(d, axis=1) #spit is where your spit would land if you stood on v3 and spit
    cp1 = e1 + spit 
    vec2 = cp1 - e3
    d2 = np.einsum('ij,ij->i', vec2, v2) / np.einsum('ij,ij->i', v2, v2)
    nor = v2 * np.expand_dims(d2, axis=1)
    cp2 = e3 + nor 
    normal = cp1 - cp2
    or_vec = e1 - cp2
    e_dot = np.einsum('j,ij->i', v1, normal)
    e_n_dot = np.einsum('ij,ij->i', normal, or_vec)
    scale = e_n_dot / e_dot  
    p_on_p =  (or_vec - v1 * np.expand_dims(scale, axis=1)) + cp2
    return p_on_p, p_on_p + spit

def edges_to_edges(e1, e2, e3, e4 ):
    '''Takes two sets of edges defined by four vectors.
    Returns the two points that describe the shortest
    distance between the two edges. The two points comprise
    a segment that is orthagonal to both edges. Think of it
    as an edge intersect.'''
    v1 = e2 - e1
    v2 = e3 - e4
    v3 = e3 - e1
    cross = np.cross(v1, v2)
    #d = np.dot(v3, cross) / np.dot(cross, cross)
    d = np.einsum('ij,ij->i',v3, cross) / np.einsum('ij,ij->i', cross, cross)
    spit = cross * np.expand_dims(d, axis=1) # spit because if you stand on cp1 and spit this is where it lands.
    cp1 = e1 + spit
    vec2 = cp1 - e3
    d2 = np.einsum('ij,ij->i', vec2, v2) / np.einsum('ij,ij->i', v2, v2)
    nor = v2 * np.expand_dims(d2, axis=1)
    cp2 = e3 + nor
    normal = cp1 - cp2
    or_vec = e1 - cp2
    e_dot = np.einsum('ij,ij->i', normal, v1)
    e_n_dot = np.einsum('ij,ij->i', normal, or_vec)
    scale = np.nan_to_num(e_n_dot / e_dot)
    p_on_p =  (or_vec - v1 * np.expand_dims(scale, axis=1)) + cp2
    return p_on_p, p_on_p + spit

def deflect_ray(e1, e2, normal, origin):
    '''Deflects a ray along the surface of the plane defined by the normal
    No Unit Vectors Required!! Normal does not requre unit vector!'''    
    e_vec = e2 - e1
    e_dot = np.dot(normal, e_vec)
    e_n_dot = np.dot(normal, e1 - origin)
    scale = e_n_dot / e_dot  
    hit = (e1 - e_vec * scale)   
    v2 = e2 - hit
    d = np.dot(v2, normal) / np.dot(normal, normal)
    cp = hit + normal * d     
    deflect = e2 - cp
    d1 = np.dot(v2, v2)
    d2 = np.dot(deflect, deflect)
    div = d1/d2
    swap = hit + deflect * np.sqrt(div) # * friction
    return swap

def reflect_ray(e1, e2, normal, origin):
    '''plots angle of reflection
    No Unit Vectors Required!! Normal does not requre unit vector!'''    
    e_vec = e2 - e1
    e_dot = np.dot(normal, e_vec)
    e_n_dot = np.dot(normal, e1 - origin)
    scale = e_n_dot / e_dot  
    hit = (e1 - e_vec * scale)   
    v2 = e2 - hit
    d = np.dot(v2, normal) / np.dot(normal, normal)
    cp = hit + normal * -d     
    deflect = e2 - cp
    return cp + deflect

def reflect_intersected_rays(e1, e2, origin, normal=np.array([0,0,1]), friction=1, sticky_threshold=.001, bounce=0.001):
    '''plots angle of reflection'''
    # could make friction behave more realistically by setting values below a certain level to zero
    e_vec = e2 - e1
    e_dot = np.einsum('j,ij->i', normal, e_vec)
    e_n_dot = np.einsum('j,ij->i', normal, e1 - origin)
    scale = e_n_dot / e_dot  
    intersect = (scale < 0) & (scale > -1) # screwing with these can lock the points on the wrong side of the plane.
    hit = (e1[intersect] - e_vec[intersect] * np.expand_dims(scale[intersect], axis=1))
    v2 = e2[intersect] - hit
    d = np.einsum('j,ij->i', normal, v2) / np.dot(normal, normal)
    cp = hit + normal * np.expand_dims(d, axis=1)     
    bp = hit + normal * np.expand_dims(-d, axis=1) * bounce
    deflect = e2[intersect] - cp
    sticky_check = np.sqrt(np.einsum('ij,ij->i', deflect, deflect))
    stuck = sticky_check < sticky_threshold
    deflect[stuck] = np.array([0.0,0.0,0.0])
    reflect = bp + deflect * friction
    return reflect, intersect

def deflect_intersected_rays(e1, e2, origin, normal=np.array([0,0,1]), back_check=0, friction=1):
    '''Deflects intersected rays along the surface of the plane defined by the normal
    No Unit Vectors Required!! Normal does not requre unit vector!    
    Returns deflected locations and bool array where intersected.
    'backcheck' will check backwards along the ray by the given amount'''    
    e_vec = e2 - e1
    e_dot = np.einsum('j,ij->i', normal, e_vec)
    e_n_dot = np.einsum('j,ij->i', normal, e1 - origin)
    scale = e_n_dot / e_dot  
    intersect = (scale < 0 + back_check) & (scale > -1)
    hit = (e1[intersect] - e_vec[intersect] * np.expand_dims(scale[intersect], axis=1))
    v2 = e2[intersect] - hit
    d = np.einsum('j,ij->i', normal, v2) / np.dot(normal, normal)
    cp = hit + normal * np.expand_dims(d, axis=1)     
    deflect = e2[intersect] - cp
    d1 = np.einsum('ij,ij->i', v2, v2)
    d2 = np.einsum('ij,ij->i', deflect, deflect)
    div = d1/d2
    new_vel = deflect * np.expand_dims(np.sqrt(div), axis=1) * friction
    swap = hit + new_vel
    return swap, intersect, new_vel

def closest_point_edge(e1, e2, p):
    '''Returns the location of the point on the edge'''
    vec1 = e2 - e1
    vec2 = p - e1
    d = np.dot(vec2, vec1) / np.dot(vec1, vec1)
    cp = e1 + vec1 * d 
    return cp

def closest_points_edge(e1, e2, p):
    '''Returns the location of the points on the edge'''
    vec1 = e2 - e1
    vec2 = p - e1
    d = np.einsum('j,ij->i', vec1, vec2) / np.expand_dims(np.dot(vec1, vec1),axis=1)
    cp = e1 + vec1 * np.expand_dims(d, axis=1) 
    return cp

def closest_points_edge_no_origin(vec, p):
    '''Returns the location of the points on the vector starting at [0,0,0]'''
    d = np.einsum('j,ij->i', vec, p) / np.expand_dims(np.dot(vec, vec),axis=1)
    cp = vec * np.expand_dims(d, axis=1) 
    return cp

def closest_points_edges(edges, points):
    '''Takes groups of edges in N x N x 2 x 3 and returns 
    the location of the points on each of the edges'''
    e1 = edges[:,0,:]
    e2 = edges[:,1,:]
    vec1 = e2 - e1
    vec2 = np.expand_dims(points, axis=1) - e1
    d = np.einsum('ijk, ijk->ij',vec2, vec1) / np.einsum('ijk, ijk->ij',vec1, vec1)
    return e1 + vec1 * np.expand_dims(d, axis=2)

def closest_point_edges(point, edges, e2='empty', merged_edges=True):
    '''Takes groups of edges in N x N x 2 x 3 or two sets of edges
    matching N x 3 and returns the location of the point on each of the edges.
    If two sets of edges are provided e1 is edges, e2 is e2, merged_edges=False'''
    if merged_edges:    
        e1 = edges[:,0,:]
        e2 = edges[:,1,:]
    else:
        e1 = edges
    vec1 = e2 - e1
    vec2 = point - e1
    d = np.einsum('ij, ij->i',vec2, vec1) / np.einsum('ij, ij->i',vec1, vec1)
    return e1 + vec1 * np.expand_dims(d, axis=2)

def drop_points_to_plane(points, normal=np.array([0,0,1]), origin='empty'):
    '''Points is an N x 3 array of vectors. Normal is perpindicular to the
    infinite plane the points will be dropped on. returns the points on the
    plane. This is the foundation for scaling on a custom axis'''
    if len(points) > 0:
        if origin != 'empty':
            points -= origin
        nor_dot = np.dot(normal, normal)
        dots = np.einsum('j,ij->i', normal, points)
        scale = dots / nor_dot   
        drop = normal * np.expand_dims(scale, axis=1)
        p = points - drop
        if origin != 'empty':
            p += origin
        return p
    else:
        return None

def slide_point_to_plane(e1, e2, normal, origin, intersect=False):
    '''Ray is defined by e1 and e2. Find where this
    ray intersects the plane defined by the normal
    and the origin. Normal does NOT need to be unit.
    Returns the hit location. If intersect: returns a
    tuple including the hit and true or false'''
    e_vec = e2 - e1
    or_vec = e1 - origin
    e_dot = np.dot(normal, e_vec)
    e_n_dot = np.dot(normal, or_vec)
    scale = e_n_dot / e_dot  
    if intersect:
        return (or_vec - e_vec * scale) + origin, (scale < 0) & (scale > -1)
    else:    
        return (or_vec - e_vec * scale) + origin

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

def slide_points_to_planes(e1, e2, origins, normals):
    '''Takes the start and end of a set of edges as Nx3 vector sets
    Returns where they intersect the planes with a bool array for the
    edges that pass through the planes. Point count must match triangle
    count so tile them in advance.'''
    e_vecs = e2 - e1
    e1or = e1 - origins
    edge_dots = np.einsum('ij,ij->i', normals, e_vecs)
    dots = np.einsum('ij,ij->i', normals, e1or)
    scale = dots / edge_dots  
    drop = (e1or - e_vecs * np.expand_dims(scale, axis=1)) + origins
    intersect = (scale < 0) & (scale > -1)
    return drop, intersect, scale

def raycast_no_check(ray_start, ray_end, tri_coords):
    '''Ray start and end take Nx3 arrays of vectors reperesenting the 
    start and the end of the rays (or edges). Use mesh_info.triangulate
    to get triangulated coords of the mesh with mesh_info.get_coords.
    tri_idx but store it instead of recalculating it on iterations. 
    Returns hit locations for each ray on each triangle with an indexing
    array containing the index of the rays that cross the planes inside
    the triangles. When a ray crosses more than one triangle, the index
    array contains the first triangle intersected.'''
    tri_vecs = np.take(tri_coords, [2, 0, 1], axis=1) - tri_coords
    origins = tri_coords[:, 0]
    rays = ray_end - ray_start
    w = np.expand_dims(ray_start, axis=1) - origins #$$
    normals = np.cross(tri_vecs[: ,0], tri_vecs[:, 1]) # if CP is there it's cheaper to get this from origin minus CP
    dots = np.einsum('ij,i...j->...i', rays, np.expand_dims(normals, axis=0))
    factor = -np.einsum('i...j,i...j->...i', w, np.expand_dims(normals, axis=0)) / dots
    hit = ray_start + rays * np.expand_dims(factor, axis=2)
    inside = (factor > 1e-6) & (factor < 1) #each column represents a ray
    return hit, inside

def collision_setup(ray_start, mesh):
    '''Used to generate the storable values needed for raycast.
    Currently assumes raycasting onto a static object. When it comes time
    to check animated objects, mesh_info.get_coords will have to be run
    every frame then viewed with the stored tri_idx'''
    tri_idx = mesh_info.triangulate(mesh)     
    tri_coords = mesh_info.get_coords(mesh, mesh)[tri_idx]
    tri_count = len(tri_coords)
    ray_count = len(ray_start)
    tri_indexing = (np.repeat([np.arange(tri_count)], ray_count, axis=0)).T    
    ray_indexing = (np.repeat([np.arange(ray_count)], tri_count, axis=0))  
    return tri_coords, tri_indexing, ray_indexing

def pre_check(ray_start, ray_end, tri_coords):
    '''creates a bool array based on the absolute bounds of all the rays
    checked against the bounds of all the triangles. Works best with single ray.
    Becomes mostly uselsess on the diagonal because the box of the ray grows.
    Consider checking coords rotated to world axis'''
    rays_combined = np.append(ray_start, ray_end, axis=0)
    tri_min = np.min(tri_coords, axis=1) # this represents the min corner of the b_box for each tri
    tri_max = np.max(tri_coords, axis=1) # this represents the max corner of the b_box for each tri
    min = np.min(rays_combined, axis=0)
    max = np.max(rays_combined, axis=0)
    outside = ((min < tri_min) & (max < tri_min)) | ((min > tri_max) & (max > tri_max))
    view = np.all(np.invert(outside), axis=1)
    return view

def pre_check_rotate(ray_start, ray_end, tri_coords):
    '''Example of raycasting a single ray and rotating
    the mesh so the diagonal doesn't create a giant box'''
    v1 = ray_end - ray_start
    Q = rotate_methods.get_quat_2(v1[0], np.array([0, 0, 1]))
    tri_vecs = tri_coords - ray_start
    shape = tri_vecs.shape
    v1r = np.array([rotate_methods.rotate_around_axis(v1[0], Q)])
    vecsr = rotate_methods.rotate_around_axis(tri_vecs.reshape(shape[0] * 3, 3), Q)
    tri_coords = vecsr.reshape(shape)
    rays_combined = np.append(np.array([[0,0]]), v1r[:, [0,1]], axis=0)
    tri_min = np.min(tri_coords[:, :, [0,1]], axis=1) # this represents the min corner of the b_box for each tri
    tri_max = np.max(tri_coords[:, :, [0,1]], axis=1) # this represents the max corner of the b_box for each tri
    min = np.min(rays_combined, axis=0)
    max = np.max(rays_combined, axis=0)
    outside = ((min - tri_min<0) & (max - tri_min<0)) | ((min - tri_max>0) & (max - tri_max>0))
    view = np.all(np.invert(outside), axis=1)
    return view

def raycast(ray_start, ray_end, tri_coords, tri_indexing, ray_indexing):
    '''Ray start and end take Nx3 arrays of vectors reperesenting the 
    start and the end of the rays (or edges). Use mesh_info.triangulate
    to get triangulated coords of the mesh with mesh_info.get_coords[tri_idx]
    but store it instead of recalculating it on iterations. 
    Returns hit locations for each ray on each triangle with an indexing
    array containing the index of the rays that cross the planes inside
    the triangles. When a ray crosses more than one triangle, the index
    array contains the first triangle intersected.'''
    if len(tri_coords)>0:    
        tri_vecs = np.take(tri_coords, [1, 2, 0], axis=1) - tri_coords # roll the triangle coords and subtract to get the vectors
        origins = tri_coords[:, 0] # Point of reference for ray intersect
        rays = ray_end - ray_start # convert rays to vectors
        w = np.expand_dims(ray_start, axis=1) - origins # Draw a vector from each ray start to each origin
        normals = np.cross(tri_vecs[: ,0], tri_vecs[:, 1]) # Cross product used to place the ray on the plane
        dots = np.einsum('ij,i...j->...i', rays, np.expand_dims(normals, axis=0)) # Ray and line perpindicular to triangle
        factor = -np.einsum('i...j,i...j->...i', w, np.expand_dims(normals, axis=0)) / dots # How far along the ray we go to hit the plane
        hits = ray_start + rays * np.expand_dims(factor, axis=2) # final location of hits
        
        # Check if rays crossed planes:
        intersected = (factor > 0) & (factor < 1) #each column represents a ray
        view = tri_indexing[intersected] # fancy indexing for tiling triangles
        hits_culled = hits[intersected] # view from edges passing through planes

        # Check for inside triangle where rays crossed planes
        # Phase 1: closest points on triangle vectors
        t_vecs = tri_vecs[view] # triangles repeated to match hits_culled
        tri_coords_view = tri_coords[view]
        p_vecs = np.expand_dims(hits_culled, axis=1) - tri_coords_view
        t_p_dot = np.einsum('ijk, ijk->ij', t_vecs, p_vecs)
        t_t_dot = np.einsum('ijk, ijk->ij', t_vecs, t_vecs)
        scalar = t_p_dot / t_t_dot 
        CP1 = t_vecs * np.expand_dims(scalar, axis=2) + tri_coords_view
        # Phase 2: closest points on vectors made from closest points
        CP_vecs = CP1 - np.take(tri_coords_view, [2, 0, 1], axis=1)
        p_vecs_rollback = np.expand_dims(hits_culled, axis=1) - np.take(tri_coords_view, [2, 0, 1], axis=1)
        CP_p_dot = np.einsum('ijk, ijk->ij', CP_vecs, p_vecs_rollback)
        CP_CP_dot = np.einsum('ijk, ijk->ij', CP_vecs, CP_vecs)
        scalar2 = CP_p_dot / CP_CP_dot 
        inside = np.all(scalar2<1, axis=1)
        intersected[intersected] = inside # set intersected outside to False
        
        # Set all but best hits to False (indexing arrays here are a nightmare. Needs optomizing. slicing the slices allows modifying in place: coords[view[set]] not coords[view][set])
        # Creates an array of zeros matching tiled triangles then uses argmax measuring the ray backwards for 
        #   furthest hit from end. 
        hits_culled = hits[intersected] # Reset hits culled after culling points not in triangles
        shape = intersected.shape
        rows = shape[0]
        columns = shape[1]
        ray_view = ray_indexing[intersected]
        ray_to_hits = hits_culled - ray_end[ray_view]
        mags = np.einsum('ij,ij->i', ray_to_hits, ray_to_hits)
        zeros = np.zeros((shape))
        zeros[intersected]=mags
        grid = np.expand_dims(np.arange(0, rows * columns, rows), axis=1) #create a grid to offset rows by number in each row
        argsort = (np.argsort(zeros.T,axis=1) + grid)[:,:-1].T.ravel() # offset each row by number in row to get number sequence
        intersected_t = intersected.T
        raveled = intersected_t.ravel()
        raveled[argsort] = False
        final = raveled.reshape(intersected_t.shape).T
        if len(ray_indexing[final])> 0:    
            return hits[final], ray_indexing[final]
        else:
            return None
    else:
        return None
