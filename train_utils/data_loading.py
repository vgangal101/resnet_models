import tensorflow as tf 


def get_imagenet_labels_classes(labels_file):
    label_map = {}
    
    with open(labels_file,'r') as f:
        for l in f.readlines():
            proc_l = l.strip()
            data = proc_l.split(' ')
            imgnt_class = data[0]
            real_class = ' '.join(data[1:])
            label_map[imgnt_class] = real_class
        
    labels_list = list(label_map.values())
    
    return label_map, labels_list


def process_paths_train(img_path,label):
    img_raw = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img_raw,channels=3)
    img = tf.image.random_crop(img,(224,224,3))
    return img,label

def process_paths_val(img_path,label)
    return img,label

def load_imagenet_dataset(root_dir,labels_file):
    
    train_dir = os.path.join(path_dir,'train')
    val_dir = os.path.join(path_dir,'val')
    
    label_map, labels_list = get_imagenet_labels_classes(labels_file)
    
    raw_train_dataset_paths = []
    raw_train_dataset_labels = []

    for c in os.listdir(train_dir):
        category_path = os.path.join(train_dir,c)
        for e in os.listdir(category_path):
            data_example = os.path.join(category_path,e)
            label = data_example.split('/')[-2]
            real_label =  label_map[label]
            label_num = labels_list.index(real_label)
            example = (data_example,label_num)
            raw_train_dataset_paths.append(data_example)
            raw_train_dataset_labels.append(label_num)
            
    train_dataset_paths = tf.data.Dataset.from_tensor_slices(raw_train_dataset_paths)
    train_dataset_labels = tf.data.Dataset.from_tensor_slices(raw_train_dataset_labels)
    
    train_dataset = tf.data.Dataset.zip((train_dataset_paths,train_dataset_labels))
    
    train_dataset = train_dataset.map(process_paths_train,num_parallel_calls=tf.data.AUTOTUNE)
    
    val_dir = os.path.join(path_dir,'val')

    raw_val_dataset_paths = []
    raw_val_dataset_labels = []

    for c in os.listdir(val_dir):
        category_path = os.path.join(val_dir,c)
        for e in os.listdir(category_path):
            data_example = os.path.join(category_path,e)
            label = data_example.split('/')[-2]
            real_label =  label_map[label]
            label_num = labels_list.index(real_label)
            example = (data_example,label_num)
            raw_val_dataset_paths.append(data_example)
            raw_val_dataset_labels.append(label_num)
    
    val_dataset_paths = tf.data.Dataset.from_tensor_slices(raw_val_dataset_paths)
    val_dataset_labels = tf.data.Dataset.from_tensor_slices(raw_val_dataset_labels)
    
    val_dataset = tf.data.Dataset.zip((val_dataset_paths,val_dataset_labels))
    
    #val_dataset = 
    

    

    
    
    return train_dataset, val_dataset