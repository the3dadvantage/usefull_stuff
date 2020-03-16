""" New Features: """
# someone requested a feature to let the settings apply to all selected cloth objects. It would have to be in a preference setting.
#   so probably a user preferences section is in order. The preferences need to save to the addon
#   directory so when blender loads a new file the settings come in.

# pause button using space bar or something with modal grab
# cache file option
# awesome sew doesn't care how many verts (target a location based on a percentage of the edge)
# Could I pull the normals off a normal map and add them to the bend springs for adding wrinkles?
# For adding wrinkles maybe I can come up with a clever way to put them into the bend springs.
# Could even create a paint modal tool that expands the source where the user paints to create custom wrinkles.
#   Wrinkle brush could take into account stroke direction, or could grow in all directions.
#   Crease brush would be different making wrinkles more like what you would need to iron out.

# could have a button to switch modes from animate to model. Model mode would turn down velocity and such.
#   could even have stored user settings


# Target:
# Could make the cloth always read from the source shape key
#   and just update target changes to the source shape.

# Bugs (not bunny)
# Don't currently have a way to update settings on duplicated objects.
#   !! could load a separate timer that both updates cloth objects when
#       loading a saved file and updates duplicates that have cloth properties
#   I think it would load whenever blender reopens if the module is registered
#   I'll have to see if I need to regen springs or if the global variable is overwritten every time


""" create my own pin function that writes to the vertex group
    so that I don't have to pop in and out of edit mode """

""" create my own button to switch between source and current """

""" can't currently run more than one object at a time """


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
