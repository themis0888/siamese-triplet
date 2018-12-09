
class TripletDataset(data.Dataset):
    def __init__(self, args, name='CUB' ,train=True):
        super(DatasetFromFolder, self).__init__()
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale[0]
        self.bin = self.args.ext.find('sep') >=0 or self.args.ext.find('bin') >=0
        self.Anchor_names, self.POS_names, self.NEG_names = self._get_filenames(name)

        self.patch_size = args.patch_size
        
    
    def __getitem__(self, index):
        time1 = time.time()
        Anchor_name, POS_name, NEG_name = self.Anchor_names[index], self.POS_names[index], self.NEG_names[index]
        Anchor, POS, NEG = self._open_file(Anchor_name, POS_name, NEG_name)
        time2 = time.time()
        
        Anchor_labelname, POS_labelname, NEG_labelname = self.Anchor_labelnames[index], self.POS_labelnames[index], self.NEG_labelnames[index]
        time3 = time.time()
        ime4 = time.time()
        if self.train:
            if self.args.n_colors == 4 and self.name.find('bin') == -1:
                Anchor, POS, Anchorlabel, POSlabel = self._random_crop(Anchor, POS, Anchorlabel, POSlabel)
            else:
                
                Anchor, POS = random_crop(Anchor, POS, patch_size = self.patch_size, scale = self.scale)
            Anchor, POS = augment(Anchor, POS)
        
        Anchor, POS = np2tensor(Anchor, POS)
        
        filename = POS_name
        time5 = time.time()
        #if time3:
            #print(time2-time1, time3-time2, time4-time3, time5-time4)
        #else:
            #print(time2-time1, time4-time2, time5-time4)
        
        if self.args.n_colors == 4 and self.name.find('bin') == -1:
            Anchorlabel, POSlabel = torch.Tensor(Anchorlabel), torch.Tensor(POSlabel)
            return (Anchor, Anchorlabel), (POS, POSlabel), filename
        else:
            return Anchor, POS, filename


    def __len__(self):
        return len(self.POS_names)

    def get_patch(self, Anchor, POS):
        scale = self.args.scale
        if self.train:
            Anchor, POS = common.get_patch(
                Anchor, POS,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(scale > 1),
                input_large=False
            )
            if not self.args.no_augment: Anchor, POS = common.augment(Anchor, POS)
        else:
            ih, iw = Anchor.shape[:2]
            POS = POS[0:ih * scale, 0:iw * scale]

        return Anchor, POS
        
    def _open_file(self, Anchor_filename, POS_name):
        if self.bin:
            with open(POS_name, 'rb') as _f: POS = pickle.load(_f)
        else:
            POS = load_img(POS_name)
            POS = np.asarray(POS)

        if Anchor_filename:
            if self.bin:
                with open(Anchor_filename, 'rb') as _f: Anchor = pickle.load(_f)
            else:
                Anchor = load_img(Anchor_filename)
                Anchor = np.asarray(Anchor)
        else:
            POS = np.asarray(POS)
            size = np.shape(POS)
            h, w = size(0), size(1)
            h2, w2 = (h//2)*2, (w//2)*2
            POS = POS[0:h2, 0:w2, :]
            Anchor = cv2.resize(POS, None, fx = 0.5, fy = 0.5,interpolation = cv2.INTER_CUBIC)
        
        return Anchor, POS

    def _get_filenames(self, name):
        root_dir = join(self.args.dir_data, name)
        if name == 'DIV2K':
            Anchor_dir = join(root_dir, 'DIV2K_train_Anchor_bicubic')
            Anchor_dir = join(Anchor_dir, 'X'+ str(self.args.scale[0]))

            POS_dir = join(root_dir, 'DIV2K_train_POS')
            r = self.args.data_range.split('/')
            if self.train:
                data_range = r[0].split('-')
            elif self.args.test_only:
                data_range = r[0].split('-')
            else:    
                data_range = r[1].split('-')

            POS_names = sorted(listdir(POS_dir))
            POS_names = POS_names[int(data_range[0])-1:int(data_range[1])]
            Anchor_names = sorted(listdir(Anchor_dir))
            Anchor_names = Anchor_names[int(data_range[0])-1:int(data_range[1])]
            Anchor_names = [join(Anchor_dir, x) for x in Anchor_names]
            POS_names = [join(POS_dir, x) for x in POS_names]
        elif name == 'cityscapes/leftImg8bit' or name == 'cityscapes/gtFine':
            if self.train:
                Anchor_dir = join(root_dir, 'train_Anchor_bicubic')
                POS_dir = join(root_dir, 'train_POS')
            else:
                Anchor_dir = join(root_dir, 'val_Anchor_bicubic')
                POS_dir = join(root_dir, 'val_POS')
            Anchor_dir = join(Anchor_dir, 'X'+ str(self.args.scale[0]))

            POS_names = sorted(listdir(POS_dir))
            Anchor_names = sorted(listdir(Anchor_dir))

            Anchor_names = [join(Anchor_dir, x) for x in Anchor_names]
            POS_names = [join(POS_dir, x) for x in POS_names]
        elif name == 'CUB200':
            image_dir = join(root_dir, 'images', 'images')
            mid_dir = [join(image_dir, x) for x in listdir(image_dir)]
            class_dir = sorted(mid_dir)
            class_dir = class_dir[200:400]

            POS_names = []
            classes = []
            for idx in range(len(class_dir)):
                image_names = listdir(class_dir[idx])
                image_names = sorted(image_names)
                image_names = image_names[len(image_names)//2: len(image_names)]

                POS_names += [join(class_dir[idx], x) for x in image_names]
                
                classes += [idx for x in range(len(image_names))]

            self.classes = classes
            Anchor_names = None

        return Anchor_names, POS_names

    def _get_bin(self, name, Anchor_names , POS_names):
        if name.find('cityscapes/leftImg8bit') >= 0:
            image_path = 'leftImg8bit/'
            bin_path = 'leftImg8bit/bin/'
        elif name.find('cityscapes/gtFine') >=0:
            image_path = 'gtFine/'
            bin_path = 'gtFine/bin/'
        elif name.find('DIV2K') >=0:
            image_path = 'DIV2K/'
            bin_path = 'DIV2K/bin/'


        bin_POS = [x.replace(image_path, bin_path) for x in POS_names]
        bin_Anchor = [x.replace(image_path, bin_path) for x in Anchor_names]
        bin_POS = [x.replace('png', 'pt') for x in bin_POS]
        bin_Anchor = [x.replace('png', 'pt') for x in bin_Anchor]

        if self.args.ext.find('sep') >= 0:  
            for idx, (img_path, bin_path) in enumerate(tqdm(zip(POS_names, bin_POS), ncols=80)):
                dir_path, basename = os.path.split(bin_path)
                os.makedirs(dir_path, exist_ok=True)
                self._load_and_make(img_path, bin_path)
                #print('Making binary files ' + bin_path)


            for idx, (img_path, bin_path) in enumerate(tqdm(zip(Anchor_names, bin_Anchor), ncols=80)):
                dir_path, basename = os.path.split(bin_path)
                os.makedirs(dir_path, exist_ok=True)
                self._load_and_make(img_path, bin_path)
                #print('Making binary files ' + bin_path)


        return bin_Anchor, bin_POS

    def _load_and_make(self, img_path, bin_path):
        image = load_img(img_path)
        bin = np.asarray(image)
        with open(bin_path, 'wb') as _f: pickle.dump(bin, _f)
    def _random_crop(self, Anchor, POS, Anchorlabel, POSlabel):
        h, w, c = np.shape(Anchor)

        crop_w = self.patch_size//self.scale
        crop_h = self.patch_size//self.scale
        i = random.randint(0, h- crop_h)
        j = random.randint(0, w - crop_w)
        #print(i//scale, (i+crop_h)//scale, j//scale, (j+crop_w)//scale)
        Anchor = Anchor[i:(i+crop_h), j:(j+crop_w),:]
        POS = POS[i*self.scale:(i+crop_h)*self.scale, j*self.scale:(j+crop_w)*self.scale, :]
        Anchorlabel = Anchorlabel[i:(i+crop_h), j:(j+crop_w),:]
        POSlabel = POSlabel[i*self.scale:(i+crop_h)*self.scale, j*self.scale:(j+crop_w)*self.scale, :]
    
        return Anchor, POS, Anchorlabel, POSlabel