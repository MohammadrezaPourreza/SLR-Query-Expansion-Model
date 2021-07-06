from types import FunctionType
from typing import List
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def cosine(vectors):
    a = vectors[0]
    b = vectors[1]
    a = tf.expand_dims(a, axis=0)
    a = tf.expand_dims(b, axis=0)

    loss = tf.keras.losses.cosine_similarity(a, b, axis=0)
    return loss


def get_size(ds):
    count = 0
    for d in ds:
        # print(d)
        count += 1

    return count


def zip_datasets(ds, anchor_label: int, target_label: int, label: int):

    def add_label(anchor, right):

        return (anchor, right), tf.constant(label)
        # return ({'input_1': anchor, 'input_2': right}, tf.constant(label))

    return tf.data.Dataset.zip(
        (
            ds.filter(lambda _, label: tf.equal(label, anchor_label))
            .map(lambda x, _: x)
            .shuffle(128),
            ds.filter(lambda _, label: tf.equal(label, target_label))
            .map(lambda x, _: x)
            .shuffle(128)
        ),
    ).map(add_label)


def split_and_merge_for_label(anchor_label: int, labels: List[any]) -> FunctionType:
    def split_and_merge(ds):
        final_dataset = zip_datasets(ds, anchor_label, anchor_label, 1)
        for l in labels:
            if l == anchor_label:
                continue
            final_dataset = final_dataset.concatenate(
                zip_datasets(ds, anchor_label, l, 0)
            )
        return final_dataset

    return split_and_merge


def create_gen(values):
    x = values[0]
    y = values[1]

    def gen():
        for i in range(len(x)):
            yield (x[i], y[i])
    return gen


def to_one_hot(x, label):
    return x, tf.one_hot(label, 2)


def create_dataset(values, labels: List[any], one_hot=False):

    def get_from_generator():
        return tf.data.Dataset.from_generator(
            generator=create_gen(values),
            output_types=(tf.float64, tf.int32),
            output_shapes=(values[0][0].shape, values[1][0].shape),
        )

    dataset = get_from_generator().apply(
        split_and_merge_for_label(labels[0], labels=labels))
    for l in labels[1:]:

        dataset = dataset.concatenate(get_from_generator().apply(
            split_and_merge_for_label(l, labels=labels)))

    if one_hot:
        dataset = dataset.map(to_one_hot)
    return dataset


if __name__ == '__main__':
    np.random.seed(1)

    y = np.random.randint(0, high=3, size=200)
    x = np.random.rand(200, 2, 4)
    dataset = create_dataset((x, y), [0, 1, 2], one_hot=False)
    print(dataset)
    # for idx, d in enumerate(dataset):
    #     print(cosine(d[0]))
    #     if idx > 10:
    #         exit(0)
    # ds = tf.data.Dataset.from_generator(
    #     generator=create_gen((x, y)),
    #     output_shapes=(x[0].shape, y[0].shape),
    #     output_types=(tf.float64, tf.int32))
    # ds = ds.apply(split_and_merge_for_label(0, labels=[0, 1, 2]))
    # print(ds)
