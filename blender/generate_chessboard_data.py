import bpy
import os

def set_material_to_current_object(material_name):
    # Get the material
    material = bpy.data.materials.get(material_name)

    # If the material doesn't exist, exit the function
    if material is None:
        print("Material", material_name, "does not exist.")
        return

    # Assign the material to the active object
    bpy.context.object.data.materials.append(material)
    

def copy_model_by_name(model_name, color, collection_name, x_pos, y_pos):
    # Duplicate the object by name
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[model_name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[model_name]
    bpy.ops.object.duplicate()

    # Get the duplicated object
    duplicated_obj = bpy.context.active_object
    if color=="Black":
        set_material_to_current_object("BlackModel")
    else:
        set_material_to_current_object("WhiteModel")

    # Create collection if it doesn't exist
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)

    # Move the duplicated object to the specified collection
    bpy.data.collections[collection_name].objects.link(duplicated_obj)

    # Set the location of the duplicated object
    duplicated_obj.location.x = x_pos
    duplicated_obj.location.y = y_pos


def delete_objects_in_collection(collection_name='models_temp'):
    # Check if the collection exists
    if collection_name in bpy.data.collections:
        # Get the collection
        collection = bpy.data.collections[collection_name]
        
        # Iterate over all objects in the collection
        for obj in collection.objects:
            # Remove the object from the scene
            bpy.data.objects.remove(obj, do_unlink=True)

def render(filename):
    blend_file_path = bpy.data.filepath
    blend_file_directory = os.path.dirname(blend_file_path)

    # Set the render resolution (optional)
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    # Set the output file format and path
    output_filename = f"renders/{filename}.png" 
    output_path = os.path.join(blend_file_directory, output_filename)
    bpy.context.scene.render.filepath = output_path

    # Render the scene
    bpy.ops.render.render(write_still=True)

    print("Image rendered and saved to:", output_path)


# Example usage

copy_model_by_name("Pawn", "Black", "models_temp", 0.1, 0.4)
copy_model_by_name("Bishop", "Black", "models_temp", 0.4, 0.4)
copy_model_by_name("King", "White", "models_temp", -0.4, 0.4)
render("0001")

delete_objects_in_collection()