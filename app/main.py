import os
import numpy as np
import time
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import base64
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException
from app.schemas import EncryptRequest, EncryptResponse, DecryptRequest, DecryptResponse
from app.crypto.utils import generate_logistic_map_images_dataset, used_images, generate_unique_random_images, shannon_entropy
from app.crypto.autoencoder import build_autoencoder
from app.crypto.encryption import (
    generate_enhanced_rsa_keys_from_image,
    secure_decrypt,
    dynamic_retraining_with_chaos_maps,
    dynamic_retraining_test
)
from app.crypto.utils import generate_logistic_map_images_dataset, used_images

app = FastAPI(title="Encryption Service with Dynamic Training, Quality Tests and Optimizations")

# Глобальные переменные для модели и RSA‑ключей
model_autoencoder = None
encoder_model = None
current_private_key = None
current_public_key = None

# Кэш для результатов динамического переобучения (действителен 10 сек)
dynamic_training_cache = {
    "timestamp": 0,
    "result": None,
    "private_key": None,
    "public_key": None,
}
executor = ThreadPoolExecutor(max_workers=2)

@app.on_event("startup")
def startup_event():
    global model_autoencoder, encoder_model, current_private_key, current_public_key
    model_autoencoder, encoder_model = build_autoencoder((28, 28))
    primary_images = generate_logistic_map_images_dataset(1000, image_size=28, r=3.99, fixed_initial=False)
    print("Первичное обучение автоэнкодера на картах хаотичности...")
    model_autoencoder.fit(primary_images, primary_images, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
    current_private_key, current_public_key, _, _ = generate_enhanced_rsa_keys_from_image(encoder_model, used_images)
    print("Сервис запущен. RSA‑ключи сгенерированы.")

def get_dynamic_training_results():
    global dynamic_training_cache, model_autoencoder, encoder_model
    cache_validity_seconds = 10
    if time.time() - dynamic_training_cache["timestamp"] < cache_validity_seconds and dynamic_training_cache["result"] is not None:
        return (dynamic_training_cache["result"],
                dynamic_training_cache["private_key"],
                dynamic_training_cache["public_key"])
    else:
        future = executor.submit(dynamic_retraining_test, model_autoencoder, encoder_model, 500, 2, used_images)
        training_time, mse, key_time = future.result()
        dynamic_training_cache["timestamp"] = time.time()
        dynamic_training_cache["result"] = {
            "training_time": training_time,
            "reconstruction_mse": mse,
            "key_generation_time": key_time
        }
        new_priv, new_pub, _, _ = generate_enhanced_rsa_keys_from_image(encoder_model, used_images)
        dynamic_training_cache["private_key"] = new_priv
        dynamic_training_cache["public_key"] = new_pub
        return dynamic_training_cache["result"], new_priv, new_pub

@app.post("/encrypt", response_model=EncryptResponse)
def encrypt_message(req: EncryptRequest):
    global current_private_key, current_public_key
    try:
        test_results, new_private_key, new_public_key = get_dynamic_training_results()
        current_private_key = new_private_key
        current_public_key = new_public_key
        message_bytes = req.message.encode('utf-8')
        ciphertext = current_public_key.encrypt(
            message_bytes,
            padding.OAEP(
                mgf=padding.MGF1(hashes.SHA512()),
                algorithm=hashes.SHA512(),
                label=None
            )
        )
        enc_b64 = base64.b64encode(ciphertext).decode('utf-8')
        current_timestamp = datetime.utcnow().isoformat()
        return EncryptResponse(
            encrypted=enc_b64,
            timestamp=current_timestamp,
            encryption_test=test_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decrypt", response_model=DecryptResponse)
def decrypt_message(req: DecryptRequest):
    global current_private_key, current_public_key
    try:
        ciphertext = base64.b64decode(req.encrypted.encode('utf-8'))
        plaintext = secure_decrypt(current_private_key, ciphertext)
        return DecryptResponse(
            message=plaintext.decode('utf-8'),
            decryption_test={"test_original": "Decryption Test", "test_decrypted": plaintext.decode('utf-8')}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quality")
def quality_checks():
    try:
        quality_results = {}
        num_quality_images = 10

        # 1. Латентная вариативность: вычисляем среднее евклидово расстояние между латентными представлениями
        latent_reps = []
        for _ in range(num_quality_images):
            img = generate_unique_random_images(1, shape=(28, 28, 1), used_images=used_images)[0]
            latent = encoder_model.predict(img[np.newaxis], verbose=0)
            latent_reps.append(latent)
        latent_reps = np.array(latent_reps).squeeze()
        distances = [
            np.linalg.norm(latent_reps[i] - latent_reps[j])
            for i in range(len(latent_reps)) for j in range(i + 1, len(latent_reps))
        ]
        avg_latent_variation = float(np.mean(distances))
        quality_results["avg_latent_variation"] = avg_latent_variation

        # 2. Эффект лавины: проверка, насколько небольшое изменение входного изображения влияет на латентное представление
        test_img = generate_unique_random_images(1, shape=(28, 28, 1), used_images=used_images)[0]
        latent_orig = encoder_model.predict(test_img[np.newaxis], verbose=0)
        test_img_mod = test_img.copy()
        # Изменяем один пиксель
        test_img_mod[14, 14, 0] = np.clip(test_img_mod[14, 14, 0] + 0.1, 0, 1)
        latent_modified = encoder_model.predict(test_img_mod[np.newaxis], verbose=0)
        avalanche_diff = float(np.linalg.norm(latent_orig - latent_modified))
        quality_results["avalanche_effect_difference"] = avalanche_diff

        # 3. Шифрование/Дешифрование: замер времени и оценка энтропии шифротекста
        encryption_times = []
        decryption_times = []
        ciphertext_entropies = []
        messages = [f"Benchmark message {i}".encode('utf-8') for i in range(5)]
        for msg in messages:
            # Генерируем ключи для каждого тестового сообщения
            private_key, public_key, _, _ = generate_enhanced_rsa_keys_from_image(encoder_model, used_images)
            start_enc = time.time()
            ct = public_key.encrypt(
                msg,
                padding.OAEP(
                    mgf=padding.MGF1(hashes.SHA512()),
                    algorithm=hashes.SHA512(),
                    label=None
                )
            )
            encryption_times.append(time.time() - start_enc)
            start_dec = time.time()
            dec = secure_decrypt(private_key, ct)
            decryption_times.append(time.time() - start_dec)
            ciphertext_entropies.append(shannon_entropy(ct))
        quality_results["avg_encryption_time"] = float(np.mean(encryption_times))
        quality_results["avg_decryption_time"] = float(np.mean(decryption_times))
        quality_results["avg_ciphertext_entropy"] = float(np.mean(ciphertext_entropies))

        # 4. Динамическое дообучение: запуск теста дообучения и сбор статистики
        # Здесь используется функция dynamic_retraining_test с 100 изображениями и 1 эпохой для быстроты теста.
        training_time, reconstruction_mse, key_generation_time = dynamic_retraining_test(
            model_autoencoder, encoder_model, num_images=100, epochs=1, used_images=used_images
        )
        quality_results["dynamic_training"] = {
            "training_time": training_time,
            "reconstruction_mse": reconstruction_mse,
            "key_generation_time": key_generation_time
        }

        return {"quality_tests": quality_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
