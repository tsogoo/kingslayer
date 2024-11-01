import random
import bpy
import os

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def camera_view_bounds_2d(scene, cam_ob, me_ob):
    
    # https://blender.stackexchange.com/questions/7198/save-the-2d-bounding-box-of-an-object-in-rendered-image-to-a-text-file
    
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    
    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    
    lx = []
    ly = []

    
    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    
    mesh_eval.to_mesh_clear()

    
    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    # Sanity check
    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        return (0, 0, 0, 0)

    return (
        round(min_x * dim_x ) / r.resolution_x,     # X
        round(dim_y - max_y * dim_y) / r.resolution_y,    # Y
        round((max_x - min_x) * dim_x) / r.resolution_x,  # Width
        round((max_y - min_y) * dim_y) / r.resolution_y  # Height
    )
    

class BlenderChess:
    
    def __init__(self):
        blend_file_path = bpy.data.filepath
        self.blender_dir = os.path.dirname(blend_file_path)
        self.models = ["Pawn", "Bishop", "King", "Queen", "Rook", "Knight"]
        self.BLACK = 0
        self.WHITE = 1
        self.MODEL_RADIUS = 0.15
        self.MIN_MODELS = 1
        self.MAX_MODELS = 32
        self.MIN_X = -1
        self.MAX_X = 1
        self.MIN_Y = -1
        self.MAX_Y = 1
        self.RENDER_WIDTH = 640
        self.RENDER_HEIGHT = 640
        self.CROP_WIDTH = 640
        self.CROP_HEIGHT= 640
        self.IMG_WIDTH = 640
        self.IMG_HEIGHT = 640
        self.TRAIN_ITER = 13000
        self.VAL_ITER = 200
        self.TEST_ITER = 20

        # world = bpy.data.worlds['World']
        # world.use_nodes = True
        bpy.data.objects['BgPlane'].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects['BgPlane']
        self.board_model = bpy.data.objects["Board"]
        material = bpy.data.objects['BgPlane'].material_slots[0].material
        # bgplane.use_nodes = True
        self.bg_node = material.node_tree.nodes['Image Texture']

    def set_random_background(self):
        bg_imgs = os.listdir(os.path.join(self.blender_dir,"..", "backgrounds"))
        bg_img = random.choice(bg_imgs)
        self.bg_node.image = bpy.data.images.load(os.path.join(self.blender_dir,"..", "backgrounds", bg_img))

    def set_material_to_current_object(self, obj, material_name):
        if obj is None:
            print(f"Object not found.")
            return
        
        # Get the material by name
        mat = bpy.data.materials.get(material_name)
        if mat is None:
            print(f"Material '{material_name}' not found.")
            return
        
        # Assign the material to the object
        if obj.data.materials:
            # If the object already has materials assigned, replace the first one
            obj.data.materials[0] = mat
        else:
            # Otherwise, add the material to the object
            obj.data.materials.append(mat)
        # Get the material
        material = bpy.data.materials.get(material_name)

        # If the material doesn't exist, exit the function
        if material is None:
            print("Material", material_name, "does not exist.")
            return

        # Assign the material to the active object
        
        #bpy.context.object.data.materials.append(material)
        

    def copy_model_by_name(self, model_index, color, collection_name, y_pos, x_pos):
        # Duplicate the object by name
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[self.models[model_index]].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[self.models[model_index]]
        bpy.ops.object.duplicate()

        # Get the duplicated object
        duplicated_obj = bpy.context.active_object
        if color==self.BLACK:
            self.set_material_to_current_object(duplicated_obj, "BlackModel")
        else:
            self.set_material_to_current_object(duplicated_obj, "WhiteModel")

        # Create collection if it doesn't exist
        if collection_name not in bpy.data.collections:
            new_collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(new_collection)

        # Move the duplicated object to the specified collection
        bpy.data.collections[collection_name].objects.link(duplicated_obj)

        # Set the location of the duplicated object
        duplicated_obj.location.x = x_pos
        duplicated_obj.location.y = y_pos
        duplicated_obj.rotation_euler.z = random.uniform(0, 360)
        
        return duplicated_obj
        

    def delete_objects_in_collection(self, collection_name='models_temp'):
        # Check if the collection exists
        if collection_name in bpy.data.collections:
            # Get the collection
            collection = bpy.data.collections[collection_name]
            
            # Iterate over all objects in the collection
            for obj in collection.objects:
                # Remove the object from the scene
                bpy.data.objects.remove(obj, do_unlink=True)

    def save_render(self, filename):
        self.set_random_background()
        # Set the render resolution (optional)
        bpy.context.scene.render.resolution_x = self.RENDER_WIDTH
        bpy.context.scene.render.resolution_y = self.RENDER_HEIGHT
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 8
        # Set the output file format and path
        output_filename = f"{filename}.png" 
        
        bpy.context.scene.render.filepath = output_filename

        # Render the scene
        bpy.ops.render.render(write_still=True)

        print("Image rendered and saved to:", output_filename)

     
        image = bpy.data.images.load(output_filename)
      

        #image.scale(self.IMG_WIDTH, self.IMG_HEIGHT)

        image.save_render(output_filename)


    def get_random_not_collided_position(self, data, radius):
        while True:
            x_pos = random.uniform(self.MIN_X, self.MAX_X)
            y_pos = random.uniform(self.MIN_Y, self.MAX_Y)
            collided = False
            for d in data:
                if (d[2] - x_pos)**2 + (d[3] - y_pos)**2 < radius**2:
                    collided = True
                    break
            if not collided:
                return x_pos, y_pos

   
    def draw_chessboard(self):
        total_models = random.randint(self.MIN_MODELS, self.MAX_MODELS)
        self.board_hide = False # random.choice([True, False])
        self.board_model.hide_render = self.board_hide

        data = []
        for i in range(total_models):
            model_index = random.randint(0, len(self.models)-1)
            side = random.choice([self.BLACK, self.WHITE])
            
            x_pos, y_pos = self.get_random_not_collided_position(data, 1.5 * self.MODEL_RADIUS)
            model = self.copy_model_by_name(model_index, side, "models_temp", x_pos, y_pos)
            data.append([f"{model_index + side * len(self.models)}", side, x_pos, y_pos, model])
        return  data


    def calculate_label(self, row):
       
        scale=1
        if row[0]==0 and row[0]==6: # pawn
            scale=1
        elif row[0]==1 and row[0]==7: # bishop
            scale=1.75
        elif row[0]==2 and row[0]==8: # king 
            scale=1.9
        elif row[0]==3 and row[0]==9: # queen
            scale=1.8
        elif row[0]==4 and row[0]==10: # rook
            scale=1.5
        elif row[0]==5 and row[0]==11: # knight
            scale=1.7
        
        trapets_ind_x = 1 + (row[3] / 8.6)
        scale = scale * (1.5 - (row[3] / 5 ))
        trapets_ind_y = 1

        y = (row[3] * trapets_ind_x) / (2.5 * self.MAX_Y) + 0.440 * self.MAX_Y
        print(row[3], y)
        x = (row[2] * trapets_ind_x) / (2.3 * self.MAX_X)  + 0.490 * self.MAX_X
        print(row[2], x)
        height = (scale) * self.MODEL_RADIUS  / (2.4     * self.MAX_Y * trapets_ind_x) + 0.04
        width = scale * self.MODEL_RADIUS * trapets_ind_x / (2.4 * self.MAX_X) + abs(row[2])/50
        if x < 0 or x > 1 or y < 0 or y > 1:
            print("Invalid label", row, x, y, width, height)
            exit()
        return f"{row[0]} {x} {y} {width} {height}"

    def calculate_label_by_blender(self, model_index, blender_model):    
        x, y, width, height = camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera, blender_model)
        if width < 0.05 or height < 0.05:
            return None
        return f"{model_index} {x+width/2} {y+height/2} {width} {height}"

    def save_label(self, path):
        with open(path, "w") as f:
            label_line = self.calculate_label_by_blender(0, self.board_model)
            if label_line:
                f.write(f"{label_line}\n")

    def move_camera_and_environment_randomly(self):
        
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = random.uniform(0, 1)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (random.uniform(0, 1),random.uniform(0,1), random.uniform(0,1), 1)

        bpy.data.objects["Sun"].data.energy = random.uniform(0.1, 8.5)

        bpy.data.objects["Camera"].location.x = random.uniform(1, 2)
        #bpy.data.objects["Camera"].location.y = random.uniform(-1, 1)
        bpy.data.objects["Camera"].location.z = random.uniform(2.8, 3.5)
        bpy.data.objects["Camera"].rotation_euler.x = random.uniform(0.28, 0.5)
        #bpy.data.objects["Camera"].rotation_euler.y = random.uniform(0, 0.5)
        bpy.data.objects["Camera"].rotation_euler.z = random.uniform(1.3, 1.8)


    def generate_data(self, mode, iter, start_index = 0):
        # make dir if not exists named mode
        data_dir = os.path.join(self.blender_dir, "..", "datasets_board")
        if not os.path.exists(os.path.join(data_dir, mode)):
            os.makedirs(os.path.join(data_dir, mode))
            os.makedirs(os.path.join(data_dir, mode, "images"))
            os.makedirs(os.path.join(data_dir, mode, "labels"))

        for i in range(start_index, start_index + iter):
            self.move_camera_and_environment_randomly()
            data = self.draw_chessboard()
            self.save_render(os.path.join(data_dir,mode, "images", str(i).zfill(4)))
            self.save_label(os.path.join(data_dir,mode, "labels", f"{str(i).zfill(4)}.txt"))
            self.delete_objects_in_collection()


blender_chess = BlenderChess()
#blender_chess.generate_data("test",blender_chess.TEST_ITER, 0)
#blender_chess.generate_data("val", blender_chess.VAL_ITER, 0)
blender_chess.generate_data("train", blender_chess.TRAIN_ITER, 3000)
