from MLNamespace import MLNamespace
import pickle
from resize_images import resize_all


def feature_extraction(mode='cnn_pca'):
    """Attempt to read features from a file, and if that fails recalculate features."""
    filepath = f'saved_features/{mode}.p'
    function_name = f'feature_extraction_{mode}'
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except:
        namespace = MLNamespace()
        namespace.data_dir = 'data'
        namespace.batch_size = 512
        getattr(namespace, function_name)()
        with open(filepath, 'wb') as file:
            pickle.dump(namespace, file)
        return namespace


# resize_all(data_dir='data', target_reslution=224)


namespace = feature_extraction()
namespace.random_state = 100
namespace.max_pca_features = 64
namespace.cross_val_k = 5
del namespace.cnn
print(namespace.__dict__.keys())
namespace.load_model('LogisticRegression', max_iter=100)
#namespace.load_model('RandomForest', n_estimators=10, class_weight='balanced', min_samples_split=100, max_depth=4)
namespace.model_training_kfold()