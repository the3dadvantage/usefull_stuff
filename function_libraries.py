# to save an external text in a blend file
def save_text_in_blend_file(path, file_name='my_text.py'):
    """Run this then save the blend file.
    file_name is the key blender uses to store the file:
    bpy.data.texts[file_name]"""
    t = bpy.data.texts.new(file_name)
    read = open(path).read()
    t.write(read)


# to import the text as a module from within the blend file
def get_internal_text_as_module(filename, key):
    """Load a module and return a dictionary from
    that module."""
    module = bpy.data.texts[filename].as_module()
    return module.points[key]


def get_co(ob):
    """Returns Nx3 numpy array of vertex coords as float32"""
    v_count = len(ob.data.vertices)
    co = np.empty(v_count * 3, dtype=np.float32)
    ob.data.vertices.foreach_get('co', co)
    co.shape = (v_count, 3)
    return co



def new_shape_key(ob, name, arr=None, value=1):
    """Create a new shape key, set it's coordinates
    and set it's value"""
    new_key = ob.shape_key_add(name=name)
    new_key.value = value
    if arr is not None:
        new_key.data.foreach_set('co', arr.ravel())
    return new_key


def get_verts_in_group(ob, name):
    """Returns the indices of the verts that belong to the group"""
    idx = ob.vertex_groups[name].index
    vg = [v.index for v in ob.data.vertices if idx in [vg.group for vg in v.groups]]
    return np.array(vg)


def apply_shape(ob, modifier_name='Cloth', update_existing_key=False, keep=['Cloth'], key_name='Cloth'):
    """Apply modifier as shape without using bpy.ops.
    Does not apply modifiers.
    Mutes modifiers not listed in 'keep.'
    Using update allows writing to an existing shape_key."""
    
    def turn_off_modifier(modifier, on_off=False):
        modifier.show_viewport = on_off
    
    mod_states = [mod.show_viewport for mod in ob.modifiers]
    [turn_off_modifier(mod, False) for mod in ob.modifiers if mod.name not in keep]
    
    dg = bpy.context.evaluated_depsgraph_get()
    proxy = ob.evaluated_get(dg)
    co = get_co(proxy)
    
    if update_existing_key:    
        key = ob.data.shape_keys.key_blocks[key_name]    
    else:
        key = new_shape_key(ob, name=key_name, arr=None, value=0)

    key.data.foreach_set('co', co.ravel())
    
    for i, j in zip(mod_states, ob.modifiers):
        j.show_viewport = i
    
    return key


# get depsgraph co with various modifiers turned off
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
