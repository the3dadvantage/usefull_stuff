# -------------------------------- START save states
# Save current state will let you create a shape key for each time you like your cloth settings.
# It will need to respect armature or other deforms so we'll have to think about the modifier stack and so on.
# maybe create the little arrows in the ui to let you move up and down through your saved state.
# Name each shape key with something that let's the UI know which keys to scroll through.
# !!! Will need the modeling cloth to switch to your current selected state instead of writing to the modeling cloth key!!!
# Need to think about which modifiers to turn on and off here... Anything that changes the vert count has to go.
#   Might be able to check which modifiers have the "apply as shape" option.
#   Blender might have already sorted mods that change vertex counts in this way

def soft_grab(cloth):
    """
    uses various falloff curves to grab points in the cloth.
    As the size of the area increases the points around the selection are
    expanded. The distance along the surface is then measured to apply grab
    motion with the appropriate level of force/falloff for the distance from the selected point.
    Needs to work with hooks such that each hook can have a falloff setting. This way
    You can animate hooks to behave more like fingers pusing or pulling the cloth
    instead of just a single point or a selection area behaving as if rigid. 
    """
    

    
    

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
    # -------------------------------- END save states
