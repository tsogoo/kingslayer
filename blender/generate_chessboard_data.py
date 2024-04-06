import random
import bpy
import os
import numpy as np


class BlenderChess:
    
    def __init__(self):
        blend_file_path = bpy.data.filepath
        self.blender_dir = os.path.dirname(blend_file_path)
        self.models = ["Pawn", "Bishop", "King", "Queen", "Rook", "Knight"]
        self.BLACK = 0
        self.WHITE = 1
        self.MODEL_RADIUS = 0.15
        self.MIN_MODELS = 1
        self.MAX_MODELS = 36
        self.MIN_X = -1
        self.MAX_X = 1
        self.MIN_Y = -1
        self.MAX_Y = 1
        self.RENDER_WIDTH = 1920
        self.RENDER_HEIGHT = 1920
        self.CROP_WIDTH = 900
        self.CROP_HEIGHT= 900
        self.IMG_WIDTH = 640
        self.IMG_HEIGHT = 640
        self.TRAIN_ITER = 200
        self.VAL_ITER = 20
        self.TEST_ITER = 10


    def set_material_to_current_object(self, material_name):
        # Get the material
        material = bpy.data.materials.get(material_name)

        # If the material doesn't exist, exit the function
        if material is None:
            print("Material", material_name, "does not exist.")
            return

        # Assign the material to the active object
        bpy.context.object.data.materials.append(material)
        

    def copy_model_by_name(self, model_index, color, collection_name, y_pos, x_pos):
        # Duplicate the object by name
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[self.models[model_index]].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[self.models[model_index]]
        bpy.ops.object.duplicate()

        # Get the duplicated object
        duplicated_obj = bpy.context.active_object
        if color==self.BLACK:
            self.set_material_to_current_object("BlackModel")
        else:
            self.set_material_to_current_object("WhiteModel")
        self.set_material_to_current_object("BlackModel")

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
        # Set the render resolution (optional)
        bpy.context.scene.render.resolution_x = self.RENDER_WIDTH
        bpy.context.scene.render.resolution_y = self.RENDER_HEIGHT

        # Set the output file format and path
        output_filename = f"{filename}.png" 
        
        bpy.context.scene.render.filepath = output_filename

        # Render the scene
        bpy.ops.render.render(write_still=True)

        print("Image rendered and saved to:", output_filename)

     
        image = bpy.data.images.load(output_filename)
      

        image.scale(self.IMG_WIDTH, self.IMG_HEIGHT)

        image.save_render(output_filename)


    def get_random_not_collided_position(self, data, radius):
        while True:
            x_pos = random.uniform(self.MIN_X, self.MAX_X)
            y_pos = random.uniform(self.MIN_Y, self.MAX_Y)
            collided = False
            for d in data:
                if (d[1] - x_pos)**2 + (d[2] - y_pos)**2 < radius**2:
                    collided = True
                    break
            if not collided:
                return x_pos, y_pos

   
    def draw_chessboard(self):
        total_models = random.randint(self.MIN_MODELS, self.MAX_MODELS)
        data = []
        for i in range(total_models):
            model_index = random.randint(0, len(self.models)-1)
            side = random.choice([self.BLACK, self.WHITE])
            x_pos, y_pos = self.get_random_not_collided_position(data, self.MODEL_RADIUS)
            self.copy_model_by_name(model_index, side, "models_temp", x_pos, y_pos)
            data.append([f"{model_index + side * len(self.models)}", side, x_pos, y_pos])

        # self.copy_model_by_name(0, 0, "models_temp", self.MAX_X, -self.MAX_Y)
        # data.append([0,0, self.MAX_X, -self.MAX_Y])
        # self.copy_model_by_name(0, 0, "models_temp", -self.MAX_X, -self.MAX_Y)
        # data.append([0,0, -self.MAX_X, -self.MAX_Y])
        # self.copy_model_by_name(0, 0, "models_temp", self.MAX_X, -self.MAX_Y/2)
        # data.append([0,0, self.MAX_X, -self.MAX_Y/2])
        # self.copy_model_by_name(0, 0, "models_temp", -self.MAX_X, -self.MAX_Y/2)
        # data.append([0,0, -self.MAX_X, -self.MAX_Y/2])
        # self.copy_model_by_name(0, 0, "models_temp", self.MAX_X, 0)
        # data.append([0,0, self.MAX_X, 0])
        # self.copy_model_by_name(0, 0, "models_temp", -self.MAX_X, 0)
        # data.append([0,0, -self.MAX_X, 0])
        # self.copy_model_by_name(0, 0, "models_temp", self.MAX_X, self.MAX_Y)
        # data.append([0,0, self.MAX_X, self.MAX_Y])
        # self.copy_model_by_name(0, 0, "models_temp", -self.MAX_X, self.MAX_Y)
        # data.append([0,0, -self.MAX_X, self.MAX_Y])
        # self.copy_model_by_name(0, 0, "models_temp", self.MAX_X, self.MAX_Y/2)
        # data.append([0,0, self.MAX_X, self.MAX_Y/2])
        # self.copy_model_by_name(0, 0, "models_temp", -self.MAX_X, self.MAX_Y/2)
        # data.append([0,0, -self.MAX_X, self.MAX_Y/2])
        # self.copy_model_by_name(0, 0, "models_temp", 0, -self.MAX_Y)
        # data.append([0,0, 0, -self.MAX_Y])
        # self.copy_model_by_name(0, 0, "models_temp", 0, self.MAX_Y)
        # data.append([0,0, 0, self.MAX_Y])
        return  data


    def calculate_label(self, row):
       
        scale=1
        if row[0]!=0 and row[0]!=6: # not pawn
            scale=1.3
        
        trapets_ind_x = 1 + (row[3] / 8.7)
        
        trapets_ind_y = 1

        y = (row[3] * trapets_ind_x) / (2.6 * self.MAX_Y) + 0.47 * self.MAX_Y 
        print(row[3], y)
        x = (row[2] * trapets_ind_x) / (2.4 * self.MAX_X)  + 0.4799 * self.MAX_X
        print(row[2], x)
        height = scale * self.MODEL_RADIUS / (1.7 * self.MAX_Y)           
        width = self.MODEL_RADIUS / (2.4 * self.MAX_X)           
        if x < 0 or x > 1 or y < 0 or y > 1:
            print("Invalid label", row, x, y, width, height)
            exit()
        return f"{row[0]} {x} {y} {width} {height}"

    def save_label(self, path, data):
        with open(path, "w") as f:
            for row in data:
                f.write(f"{self.calculate_label(row)}\n")

    def generate_data(self, mode, iter):
        # make dir if not exists named mode
        data_dir = os.path.join(self.blender_dir, "..", "datasets")
        if not os.path.exists(os.path.join(data_dir, mode)):
            os.makedirs(os.path.join(data_dir, mode))
            os.makedirs(os.path.join(data_dir, mode, "images"))
            os.makedirs(os.path.join(data_dir, mode, "labels"))

        for i in range(0, iter):
            data = self.draw_chessboard()
            self.save_render(os.path.join(data_dir,mode, "images", str(i).zfill(4)))
            self.save_label(os.path.join(data_dir,mode, "labels", f"{str(i).zfill(4)}.txt"), data)
            self.delete_objects_in_collection()



blender_chess = BlenderChess()
blender_chess.generate_data("train", blender_chess.TRAIN_ITER)
blender_chess.generate_data("val", blender_chess.VAL_ITER)
blender_chess.generate_data("test", blender_chess.TEST_ITER)
