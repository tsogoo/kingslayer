import random
import bpy
import os


def clamp(x, minimum, maximum):
    return x
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

    # Transform the mesh to camera space
    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != "ORTHO"

    lx = []
    ly = []
    prev_v = None
    pivot_point = None

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)
        if prev_v is not None:
            if v.co == prev_v:
                pivot_point = {"x": x, "y": y}
        prev_v = v.co

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    # Get pivot point location in camera space
    if pivot_point is None:
        pivot_point = mat @ me_ob.location
        pivot_point = {"x": pivot_point.x, "y": pivot_point.y, "z": pivot_point.z}
    if camera_persp:
        if z != 0.0:
            frame = [(v / (v.z / z)) for v in frame]

    pivot_x = pivot_point["x"]
    pivot_y = pivot_point["y"]

    # Clear the mesh evaluation to prevent memory leaks
    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    # Sanity check
    # if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
    #     return (0, 0, 0, 0, 0, 0)
    return (
        (min_x * dim_x) / r.resolution_x,  # X
        (dim_y - max_y * dim_y) / r.resolution_y,  # Y
        ((max_x - min_x) * dim_x) / r.resolution_x,  # Width
        ((max_y - min_y) * dim_y) / r.resolution_y,  # Height
        (pivot_x * dim_x) / r.resolution_x,
        (dim_y - pivot_y * dim_y) / r.resolution_y,  # pivot_y
    )


def look_at(target):
    camera = bpy.context.scene.camera

    target_location = target.location
    dimensions = target.dimensions
    half_width = dimensions[0] / 2
    half_height = dimensions[1] / 2

    # set camera location
    camera.location.x = random.uniform(
        target_location.x - half_width, target_location.x + half_width
    )
    camera.location.y = random.uniform(
        target_location.y - half_height, target_location.y + half_height
    )
    camera.location.z = 4.2  # adjust camera height, or random height

    # Calculate the direction vector from camera to target
    direction = target.location - camera.location

    # Calculate the rotation quaternion to point the camera at the target
    rotation_quaternion = direction.to_track_quat("-Z", "Y")

    # Set the rotation of the camera
    camera.rotation_euler = rotation_quaternion.to_euler()


class BlenderChess:

    def __init__(self):
        blend_file_path = bpy.data.filepath
        self.blender_dir = os.path.dirname(blend_file_path)
        self.models = ["Pawn", "Bishop", "King", "Queen", "Rook", "Knight"]
        self.BLACK = 0
        self.WHITE = 1
        self.MODEL_RADIUS = 0.15
        self.MIN_MODELS = 1
        self.MAX_MODELS = 50
        self.MIN_X = -1
        self.MAX_X = 2
        self.MIN_Y = -1
        self.MAX_Y = 2
        self.RENDER_WIDTH = 840
        self.RENDER_HEIGHT = 840
        self.CROP_WIDTH = 840
        self.CROP_HEIGHT = 840
        self.IMG_WIDTH = 840
        self.IMG_HEIGHT = 840
        self.TRAIN_ITER = 6000
        self.VAL_ITER = 500
        self.TEST_ITER = 20

        # world = bpy.data.worlds['World']
        # world.use_nodes = True
        bpy.data.objects["BgPlane"].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects["BgPlane"]
        material = bpy.data.objects["BgPlane"].material_slots[0].material
        # bgplane.use_nodes = True
        self.bg_node = material.node_tree.nodes["Image Texture"]
        self.hdr_imgs = os.listdir(os.path.join(self.blender_dir, "hdri"))

    def set_with_pivot(self, bool):
        self.with_pivot = bool

    def set_random_background(self):
        bg_imgs = os.listdir(os.path.join(self.blender_dir, "..", "backgrounds"))
        bg_img = random.choice(bg_imgs)
        self.bg_node.image = bpy.data.images.load(
            os.path.join(self.blender_dir, "..", "backgrounds", bg_img)
        )

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

        # bpy.context.object.data.materials.append(material)

    def copy_model_by_name(self, model_index, color, collection_name, y_pos, x_pos):
        # Duplicate the object by name
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects[self.models[model_index]].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[
            self.models[model_index]
        ]
        bpy.ops.object.duplicate()

        # Get the duplicated object
        duplicated_obj = bpy.context.active_object
        if color == self.BLACK:
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
        bpy.ops.transform.resize(
            value=(1, 1, random.uniform(1, 2.5)),
            orient_type="GLOBAL",
            orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            orient_matrix_type="GLOBAL",
            constraint_axis=(False, False, True),
            mirror=False,
            use_proportional_edit=False,
            proportional_edit_falloff="SMOOTH",
            proportional_size=1,
            use_proportional_connected=False,
            use_proportional_projected=False,
            snap=False,
            snap_elements={"INCREMENT"},
            use_snap_project=False,
            snap_target="CLOSEST",
            use_snap_self=True,
            use_snap_edit=True,
            use_snap_nonedit=True,
            use_snap_selectable=False,
            release_confirm=True,
        )

        duplicated_obj.rotation_euler.z = random.uniform(0, 360)

        return duplicated_obj

    def delete_objects_in_collection(self, collection_name="models_temp"):
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
        bpy.context.scene.render.engine = random.choice(["CYCLES"])  # , "BLENDER_EEVEE"
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.cycles.samples = 8
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
                if (d[2] - x_pos) ** 2 + (d[3] - y_pos) ** 2 < radius**2:
                    collided = True
                    break
            if not collided:
                return x_pos, y_pos

    def draw_chessboard(self):
        total_models = random.randint(self.MIN_MODELS, self.MAX_MODELS)
        data = []
        bpy.data.objects["Board"].hide_render = False  # random.choice([True, False])
        for i in range(total_models):
            model_index = random.randint(0, len(self.models) - 1)
            side = random.choice([self.BLACK, self.WHITE])
            x_pos, y_pos = self.get_random_not_collided_position(
                data, 1.5 * self.MODEL_RADIUS
            )
            model = self.copy_model_by_name(
                model_index, side, "models_temp", x_pos, y_pos
            )
            data.append(
                [f"{model_index + side * len(self.models)}", side, x_pos, y_pos, model]
            )

        # self.copy_model_by_name(2, 0, "models_temp", self.MAX_X, -self.MAX_Y)
        # data.append([2,0, self.MAX_X, -self.MAX_Y])
        # self.copy_model_by_name(2, 0, "models_temp", -self.MAX_X, -self.MAX_Y)
        # data.append([2,0, -self.MAX_X, -self.MAX_Y])
        # self.copy_model_by_name(2, 0, "models_temp", self.MAX_X, -self.MAX_Y/2)
        # data.append([2,0, self.MAX_X, -self.MAX_Y/2])
        # self.copy_model_by_name(2, 0, "models_temp", -self.MAX_X, -self.MAX_Y/2)
        # data.append([2,0, -self.MAX_X, -self.MAX_Y/2])
        # self.copy_model_by_name(2, 0, "models_temp", self.MAX_X, 0)
        # data.append([2,0, self.MAX_X, 0])
        # self.copy_model_by_name(2, 0, "models_temp", -self.MAX_X, 0)
        # data.append([2,0, -self.MAX_X, 0])
        # self.copy_model_by_name(2, 0, "models_temp", self.MAX_X, self.MAX_Y)
        # data.append([2,0, self.MAX_X, self.MAX_Y])
        # self.copy_model_by_name(2, 0, "models_temp", -self.MAX_X, self.MAX_Y)
        # data.append([2,0, -self.MAX_X, self.MAX_Y])
        # self.copy_model_by_name(2, 0, "models_temp", self.MAX_X, self.MAX_Y/2)
        # data.append([2,0, self.MAX_X, self.MAX_Y/2])
        # self.copy_model_by_name(2, 0, "models_temp", -self.MAX_X, self.MAX_Y/2)
        # data.append([2,0, -self.MAX_X, self.MAX_Y/2])
        # self.copy_model_by_name(2, 0, "models_temp", 0, -self.MAX_Y)
        # data.append([2,0, 0, -self.MAX_Y])
        # self.copy_model_by_name(2, 0, "models_temp", 0, self.MAX_Y)
        # data.append([2,0, 0, self.MAX_Y])

        # look at board
        # look_at(target = bpy.data.objects["Board"])

        return data

    # def calculate_label(self, row):

    #     scale = 1
    #     if row[0] == 0 and row[0] == 6:  # pawn
    #         scale = 1
    #     elif row[0] == 1 and row[0] == 7:  # bishop
    #         scale = 1.75
    #     elif row[0] == 2 and row[0] == 8:  # king
    #         scale = 1.9
    #     elif row[0] == 3 and row[0] == 9:  # queen
    #         scale = 1.8
    #     elif row[0] == 4 and row[0] == 10:  # rook
    #         scale = 1.5
    #     elif row[0] == 5 and row[0] == 11:  # knight
    #         scale = 1.7

    #     trapets_ind_x = 1 + (row[3] / 8.6)
    #     scale = scale * (1.5 - (row[3] / 5))
    #     trapets_ind_y = 1

    #     y = (row[3] * trapets_ind_x) / (2.5 * self.MAX_Y) + 0.440 * self.MAX_Y
    #     print(row[3], y)
    #     x = (row[2] * trapets_ind_x) / (2.3 * self.MAX_X) + 0.490 * self.MAX_X
    #     print(row[2], x)
    #     height = (scale) * self.MODEL_RADIUS / (2.4 * self.MAX_Y * trapets_ind_x) + 0.04
    #     width = (
    #         scale * self.MODEL_RADIUS * trapets_ind_x / (2.4 * self.MAX_X)
    #         + abs(row[2]) / 50
    #     )
    #     if x < 0 or x > 1 or y < 0 or y > 1:
    #         print("Invalid label", row, x, y, width, height)
    #         exit()
    #     return f"{row[0]} {x} {y} {width} {height} "

    def calculate_label_by_blender(self, model_index, blender_model):
        x, y, width, height, pivot_x, pivot_y = camera_view_bounds_2d(
            bpy.context.scene, bpy.context.scene.camera, blender_model
        )
        if (
            x + width < width * 0.8
            or y + height < height * 0.8
            or x + width * 0.8 > 1
            or y + height * 0.8 > 1
        ):
            return None
        x = 1 if x + width / 2 > 1 else round(x + width / 2, 4)
        y = 1 if y + height / 2 > 1 else round(y + height / 2, 4)
        width = round(width, 4)
        height = round(height, 4)
        pivot_x = 1 if pivot_x + width / 2 > 1 else round(pivot_x + width / 2, 4)
        pivot_y = 1 if pivot_y + height / 2 > 1 else round(pivot_y + height / 2, 4)
        if self.with_pivot:
            return f"{model_index} {x} {y} {width} {height} {pivot_x} {pivot_y}"
        return f"{model_index} {x+width/2} {y+height/2} {width} {height}"

    def save_label(self, path, data):
        with open(path, "w") as f:
            for row in data:
                label_line = self.calculate_label_by_blender(row[0], row[4])
                if label_line:
                    f.write(f"{label_line}\n")

    def move_camera_and_environment_randomly(self):

        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
            1
        ].default_value = random.uniform(0, 1)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
            0
        ].default_value = (
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            1,
        )

        bg_img = os.path.join(self.blender_dir, "hdri", random.choice(self.hdr_imgs))
        bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].image = (
            bpy.data.images.load(bg_img)
        )

        bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[
            2
        ] = random.uniform(0, 6.28)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
            1
        ].default_value = random.uniform(0.8, 2)

        bpy.data.objects["Camera"].location.x = random.uniform(1, 2)
        # bpy.data.objects["Camera"].location.y = random.uniform(-1, 1)
        bpy.data.objects["Camera"].location.z = random.uniform(2.5, 3.5)
        bpy.data.objects["Camera"].rotation_euler.x = random.uniform(0.28, 0.5)
        # bpy.data.objects["Camera"].rotation_euler.y = random.uniform(0, 0.5)
        bpy.data.objects["Camera"].rotation_euler.z = random.uniform(1.3, 1.8)
        bpy.data.objects["Camera"].data.dof.use_dof = True
        bpy.data.objects["Camera"].data.dof.focus_distance = random.uniform(1.5, 2.2)
        # bpy.data.objects["Camera"].data.dof.aperture_fstop = random.uniform(2.2, 4)

    def generate_data(self, mode, iter, start_index=0):
        # make dir if not exists named mode
        data_dir = os.path.join(self.blender_dir, "..", "cm_datasets")
        if not os.path.exists(os.path.join(data_dir, mode)):
            os.makedirs(os.path.join(data_dir, mode))
            os.makedirs(os.path.join(data_dir, mode, "images"))
            os.makedirs(os.path.join(data_dir, mode, "labels"))

        for i in range(start_index, start_index + iter):
            self.move_camera_and_environment_randomly()
            data = self.draw_chessboard()
            self.save_render(os.path.join(data_dir, mode, "images", str(i).zfill(4)))
            self.save_label(
                os.path.join(data_dir, mode, "labels", f"{str(i).zfill(4)}.txt"), data
            )
            self.delete_objects_in_collection()


blender_chess = BlenderChess()
blender_chess.set_with_pivot(True)
# blender_chess.generate_data("test", blender_chess.TEST_ITER, 0)
# blender_chess.generate_data("val", blender_chess.VAL_ITER, 330)
if blender_chess.with_pivot:
    blender_chess.generate_data("torch_train", blender_chess.TRAIN_ITER, 0)

# blender_chess.generate_data("train", blender_chess.TRAIN_ITER, 13042)
