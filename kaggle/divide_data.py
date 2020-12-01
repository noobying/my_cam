import os
import shutil


def divide_train_datasets(src, dst):
    dataset_trian_cats = os.path.join(dst, 'train', 'cat')
    dataset_trian_dogs = os.path.join(dst, 'train', 'dog')

    if not os.path.isdir(dataset_trian_cats):
        os.makedirs(dataset_trian_cats)

    if not os.path.isdir(dataset_trian_dogs):
        os.makedirs(dataset_trian_dogs)

    dogs_and_cats = os.listdir(src)

    cats = [s for s in dogs_and_cats if s.startswith('cat')]
    dogs = [s for s in dogs_and_cats if s.startswith('dog')]

    for dog in dogs:
        shutil.copy(os.path.join(src, dog), os.path.join(dataset_trian_dogs, dog))

    for cat in cats:
        shutil.copy(os.path.join(src, cat), os.path.join(dataset_trian_cats, cat))



if __name__ == '__main__':

    divide_train_datasets('./train', './train1')