import os
import numpy as np
from math import log2
import gmpy2
from gmpy2 import mpz

# Константа для битовой длины генерируемых простых чисел (p и q будут 2048 бит)
KEY_BIT_LENGTH = 2048

# Глобальное множество для отслеживания использованных изображений
used_images = set()

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    g, y, x = egcd(b % a, a)
    return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, _ = egcd(a, m)
    if g != 1:
        raise Exception('Multiplicative inverse does not exist')
    return x % m


def logistic_map(x, r=3.99):
    return r * x * (1 - x)

def generate_logistic_map_image(image_size=28, initial_value=0.4, r=3.99):
    iterations = image_size * image_size
    x = initial_value
    chaotic_sequence = []
    for i in range(iterations):
        x = logistic_map(x, r)
        chaotic_sequence.append(x)
    img = np.array(chaotic_sequence).reshape((image_size, image_size))
    return img

def generate_logistic_map_images_dataset(num_images, image_size=28, r=3.99, fixed_initial=True):
    dataset = []
    for _ in range(num_images):
        if fixed_initial:
            init_val = 0.4
        else:
            init_val = np.random.rand()
        img = generate_logistic_map_image(image_size=image_size, initial_value=init_val, r=r)
        dataset.append(img)
    dataset = np.array(dataset)
    return dataset[..., np.newaxis]

def generate_unique_random_images(num_images, shape=(28, 28, 1), used_images=used_images):
    new_images = []
    while len(new_images) < num_images:
        img_bytes = os.urandom(np.prod(shape))
        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape) / 255.0
        img_hash = hash(img.tobytes())
        if img_hash not in used_images:
            used_images.add(img_hash)
            new_images.append(img)
    return np.array(new_images)

def arnold_cat_map(img, iterations=1):
    h, w = img.shape
    result = img.copy()
    for _ in range(iterations):
        temp = np.zeros_like(result)
        for y in range(h):
            for x in range(w):
                new_x = (2 * x + y) % h
                new_y = (x + y) % w
                temp[new_y, new_x] = result[y, x]
        result = temp.copy()
    return result

def generate_arnold_cat_dataset(num_images=100, image_size=28, iterations=5):
    dataset = []
    for _ in range(num_images):
        base_img = np.random.rand(image_size, image_size)
        transformed_img = arnold_cat_map(base_img, iterations=iterations)
        dataset.append(transformed_img)
    return np.array(dataset)[..., np.newaxis]

def generate_prime(seed):
    seed_int = int.from_bytes(seed, "big") | (1 << (KEY_BIT_LENGTH - 1))
    return int(gmpy2.next_prime(mpz(seed_int)))

def shannon_entropy(data):
    if len(data) == 0:
        return 0
    freq = {b: data.count(b) for b in set(data)}
    entropy = -sum((count / len(data)) * log2(count / len(data)) for count in freq.values())
    return entropy
