import os
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import psutil
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import tensorflow as tf

from . import utils

def generate_enhanced_rsa_keys_from_image(encoder, used_images=utils.used_images):
    # Генерируем уникальное изображение с помощью функции из utils
    image = utils.generate_unique_random_images(1, shape=(28, 28, 1), used_images=used_images)[0]
    latent_repr = encoder.predict(image[np.newaxis], verbose=0)
    system_entropy = os.urandom(32)
    timestamp = datetime.utcnow().isoformat().encode('utf-8')
    cpu_usage = str(psutil.cpu_percent(interval=0.01)).encode('utf-8')
    combined = latent_repr.tobytes() + system_entropy + timestamp + cpu_usage

    # Производное значение с использованием PBKDF2HMAC
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA512(), length=64,
        salt=system_entropy[:16], iterations=5000,
        backend=default_backend()
    )
    derived_key = kdf.derive(combined)

    # Генерация seed для простых чисел p и q
    seed_p_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
    seed_p_hash.update(derived_key[:32] + b"p")
    seed_p = seed_p_hash.finalize()
    
    seed_q_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
    seed_q_hash.update(derived_key[32:] + b"q")
    seed_q = seed_q_hash.finalize()
    
    # Параллельная генерация простых чисел p и q
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_p = executor.submit(utils.generate_prime, seed_p)
        future_q = executor.submit(utils.generate_prime, seed_q)
        p = future_p.result()
        q = future_q.result()
    
    if p == q:
        # Если числа совпали, выбираем следующее простое для q
        import gmpy2
        q = int(gmpy2.next_prime(q + 2))
    n = p * q
    e = 65537
    phi = (p - 1) * (q - 1)
    d = utils.modinv(e, phi)
    private_numbers = rsa.RSAPrivateNumbers(
        p, q, d,
        d % (p - 1),
        d % (q - 1),
        utils.modinv(q, p),
        rsa.RSAPublicNumbers(e, n)
    )
    private_key = private_numbers.private_key(default_backend())
    public_key = private_key.public_key()
    return private_key, public_key, system_entropy, timestamp

def secure_decrypt(private_key, ciphertext):
    start_time = time.time()
    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA512()),
            algorithm=hashes.SHA512(),
            label=None
        )
    )
    elapsed = time.time() - start_time
    target_time = 0.1  # Фиксированное время выполнения для защиты от атак по временным каналам
    if elapsed < target_time:
        time.sleep(target_time - elapsed)
    return plaintext

def dynamic_retraining_test(autoencoder, encoder, num_images=500, epochs=2, used_images=utils.used_images):
    # Дообучение автоэнкодера на уникальных изображениях
    new_images = utils.generate_unique_random_images(num_images, shape=(28, 28, 1), used_images=used_images)
    # Замораживаем нижние слои для ускорения переобучения
    for layer in autoencoder.layers[:-3]:
        layer.trainable = False
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    start_time = time.time()
    autoencoder.fit(new_images, new_images, epochs=epochs, batch_size=32, verbose=1)
    training_time = time.time() - start_time
    print(f"Время динамического дообучения автоэнкодера (на уникальных изображениях): {training_time:.3f} сек")
    reconstructed = autoencoder.predict(new_images, verbose=0)
    mse = np.mean((new_images - reconstructed) ** 2)
    print(f"Reconstruction MSE после дообучения: {mse:.6f}")
    start_time = time.time()
    private_key, public_key, _, _ = generate_enhanced_rsa_keys_from_image(encoder, used_images)
    key_time = time.time() - start_time
    print(f"Время генерации RSA-ключей: {key_time:.3f} сек")
    original_message = b"Dynamic retraining test"
    encrypted = public_key.encrypt(
        original_message,
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA512()),
            algorithm=hashes.SHA512(),
            label=None
        )
    )
    decrypted = secure_decrypt(private_key, encrypted)
    print("Проверка: оригинальное сообщение:", original_message)
    print("Проверка: дешифрованное сообщение:", decrypted)
    return training_time, mse, key_time

def dynamic_retraining_with_chaos_maps(autoencoder, encoder, num_images=500, epochs=2):
    # Дообучение автоэнкодера на изображениях, сгенерированных по хаотической логистической карте
    new_images = utils.generate_logistic_map_images_dataset(num_images=num_images, image_size=28, r=3.99, fixed_initial=False)
    for layer in autoencoder.layers[:-3]:
        layer.trainable = False
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    start_time = time.time()
    autoencoder.fit(new_images, new_images, epochs=epochs, batch_size=32, verbose=1)
    training_time = time.time() - start_time
    print(f"Время динамического дообучения автоэнкодера на случайных картах хаотичности: {training_time:.3f} сек")
    reconstructed = autoencoder.predict(new_images, verbose=0)
    mse = np.mean((new_images - reconstructed) ** 2)
    print(f"Reconstruction MSE после дообучения: {mse:.6f}")
    return training_time, mse
