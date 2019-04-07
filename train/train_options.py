
class TrainOptions:

    def __init__(self):
        self.images_root = None
        self.vm_root = None
        self.vm_size = None
        self.image_size = None
        self.patch_size = None
        self.decrease_rate = None
        self.perturbate = None
        self.opacity_var = None
        self.use_rgb = None
        self.rotate_vm = False
        self.scale_vm = False
        self.crop_vm = False
        self.batch_vm = 0
        self.weight = None
        self.font = None
        self.noise = None
        self.bounding_box = None
        self.bound_offset = None
        self.batch_size = None
        self.shared_depth = 0
        self.num_blocks = None
        self.use_vm_decoder = False
        self.text_border = False
        self.blur = False
