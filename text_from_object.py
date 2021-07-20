import numpy as np
import bpy

def py_from_object(ob, round=3):
    """Writes the verts and faces to this file
    when in blender"""
    np.set_printoptions(suppress=True)
    vc = len(ob.data.vertices)
    co = np.empty((vc, 3), dtype=np.float32)
    ob.data.vertices.foreach_get('co', co.ravel())
    r = np.round(co, round)
    
    col = []
    for i in r:
        vco = []
        for c in i:
            vco.append(c)
        col.append(vco)
    
    f = [[v for v in f.vertices] for f in ob.data.polygons]
    #print(col)
            

    return str(col), str(f)
    
v, f = py_from_object(bpy.context.object)

t = bpy.data.texts['py_from_object.py']
t.cursor_set(line = 34)
t.write('verts = ' + v)    
t.cursor_set(line = 35)
t.write('faces = ' + f)    








# --------------------------
