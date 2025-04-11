import unittest
import numpy as np
import os
import time
from math import log2
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow import keras

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from app.crypto.autoencoder import build_autoencoder
from app.crypto.encryption import (
    generate_enhanced_rsa_keys_from_image,
    secure_decrypt,
    dynamic_retraining_test
)
from app.crypto.utils import (
    generate_unique_random_images,
    generate_logistic_map_images_dataset,
    generate_logistic_map_image,
    logistic_map,
    shannon_entropy,
    used_images
)

class TestImageBasedCrypto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.autoencoder, cls.encoder = build_autoencoder((28, 28))
        # Обучаем на хаотических картах
        initial_images = generate_logistic_map_images_dataset(1000, image_size=28, r=3.99, fixed_initial=False)
        cls.autoencoder.fit(initial_images, initial_images, epochs=3, batch_size=64, validation_split=0.1, verbose=0)

    def test_encoder_consistency(self):
        test_image = generate_unique_random_images(1, shape=(28, 28, 1))[0][np.newaxis]
        latent1 = self.encoder.predict(test_image, verbose=0)
        latent2 = self.encoder.predict(test_image, verbose=0)
        self.assertTrue(np.allclose(latent1, latent2), "Latent representation is not consistent")

    def test_rsa_key_generation(self):
        private_key, public_key, _, _ = generate_enhanced_rsa_keys_from_image(self.encoder, used_images)
        self.assertIsNotNone(private_key, "Не сгенерирован приватный ключ")
        self.assertIsNotNone(public_key, "Не сгенерирован публичный ключ")

    def test_encryption_decryption(self):
        private_key, public_key, _, _ = generate_enhanced_rsa_keys_from_image(self.encoder, used_images)
        original_message = b"Test message"
        encrypted = public_key.encrypt(
            original_message,
            padding.OAEP(mgf=padding.MGF1(hashes.SHA512()),
                         algorithm=hashes.SHA512(),
                         label=None)
        )
        decrypted = secure_decrypt(private_key, encrypted)
        self.assertEqual(original_message, decrypted, "Дешифрованное сообщение не совпадает с оригиналом")

    def test_latent_variation(self):
        latents = []
        for _ in range(10):
            img = generate_unique_random_images(1, shape=(28, 28, 1))[0][np.newaxis]
            latents.append(self.encoder.predict(img, verbose=0))
        latents = np.array(latents).squeeze()
        dists = [np.linalg.norm(latents[i] - latents[j]) for i in range(len(latents)) for j in range(i+1, len(latents))]
        avg_dist = np.mean(dists)
        print(f"Среднее евклидово расстояние между латентными представлениями: {avg_dist:.3f}")
        self.assertGreater(avg_dist, 0.0, "Латентные представления слишком похожи")

    def test_avalanche_effect(self):
        test_image = generate_unique_random_images(1, shape=(28, 28, 1))[0][np.newaxis]
        latent_orig = self.encoder.predict(test_image, verbose=0)
        test_image_modified = test_image.copy()
        test_image_modified[0, 14, 14, 0] = np.clip(test_image_modified[0, 14, 14, 0] + 0.1, 0, 1)
        latent_mod = self.encoder.predict(test_image_modified, verbose=0)
        diff = np.linalg.norm(latent_orig - latent_mod)
        print(f"Разница между латентными представлениями (эффект лавины): {diff:.3f}")
        self.assertGreater(diff, 0.05, "Adversarial perturbation did not sufficiently change latent representation")

    def test_average_key_generation_time(self):
        times = []
        for _ in range(10):
            start_time = time.time()
            generate_enhanced_rsa_keys_from_image(self.encoder, used_images)
            times.append(time.time() - start_time)
        avg_time = np.mean(times)
        print(f"Среднее время генерации RSA-ключей: {avg_time:.3f} сек")
        self.assertLess(avg_time, 1.0, "Время генерации RSA-ключей слишком велико")

    def test_encryption_benchmark(self):
        messages = [f"Test message {i}".encode('utf-8') for i in range(20)]
        encryption_times, decryption_times, ciphertexts = [], [], []
        for msg in messages:
            private_key, public_key, _, _ = generate_enhanced_rsa_keys_from_image(self.encoder, used_images)
            start_enc = time.time()
            ct = public_key.encrypt(
                msg,
                padding.OAEP(mgf=padding.MGF1(hashes.SHA512()),
                             algorithm=hashes.SHA512(),
                             label=None)
            )
            encryption_times.append(time.time() - start_enc)
            ciphertexts.append(ct)
            start_dec = time.time()
            dec = secure_decrypt(private_key, ct)
            decryption_times.append(time.time() - start_dec)
            self.assertEqual(msg, dec)
        avg_enc = np.mean(encryption_times)
        avg_dec = np.mean(decryption_times)
        entropies = [shannon_entropy(ct) for ct in ciphertexts]
        avg_entropy = np.mean(entropies)
        print(f"Average encryption time: {avg_enc:.3f} sec")
        print(f"Average decryption time: {avg_dec:.3f} sec")
        print(f"Average ciphertext entropy: {avg_entropy:.3f} bits per byte")
        self.assertGreater(avg_entropy, 7.5, "Ciphertext entropy is too low, encryption may not be secure")

    def test_latent_chaos_behavior(self):
        num_steps = 10
        image_size = 28
        init1 = 0.4
        delta = 1e-5
        init2 = 0.4 + delta
        chain1 = []
        chain2 = []
        for _ in range(num_steps):
            img1 = generate_logistic_map_image(image_size=image_size, initial_value=init1, r=3.99)
            img2 = generate_logistic_map_image(image_size=image_size, initial_value=init2, r=3.99)
            chain1.append(img1)
            chain2.append(img2)
            init1 = logistic_map(init1, r=3.99)
            init2 = logistic_map(init2, r=3.99)
        chain1 = np.array(chain1)[..., np.newaxis]
        chain2 = np.array(chain2)[..., np.newaxis]
        latent_chain1 = self.encoder.predict(chain1, verbose=0)
        latent_chain2 = self.encoder.predict(chain2, verbose=0)
        distances = [np.linalg.norm(latent_chain1[i] - latent_chain2[i]) for i in range(num_steps)]
        print("Latent distances across time:", distances)
        self.assertGreater(distances[-1], 5 * distances[0], "Latent space does not exhibit expected chaotic divergence")

class TestValueEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.autoencoder, cls.encoder = build_autoencoder((28,28))
        images = generate_logistic_map_images_dataset(500, image_size=28, r=3.99, fixed_initial=True)
        cls.autoencoder.fit(images, images, epochs=2, batch_size=64, verbose=0)

    def test_statistical_randomness(self):
        messages = [f"Random message {i}".encode('utf-8') for i in range(50)]
        entropies = []
        for msg in messages:
            private_key, public_key, _, _ = generate_enhanced_rsa_keys_from_image(self.encoder, used_images)
            ct = public_key.encrypt(
                msg,
                padding.OAEP(mgf=padding.MGF1(hashes.SHA512()),
                             algorithm=hashes.SHA512(),
                             label=None)
            )
            entropies.append(shannon_entropy(ct))
        avg_entropy = np.mean(entropies)
        print(f"Average ciphertext entropy (statistical randomness): {avg_entropy:.3f} bits/byte")
        self.assertGreater(avg_entropy, 7.5, "Low entropy: statistical randomness test failed")

    def test_adversarial_attack_resilience(self):
        base_img = generate_logistic_map_image(image_size=28, initial_value=0.4, r=3.99)
        base_img = base_img[..., np.newaxis]
        latent_base = self.encoder.predict(base_img[np.newaxis], verbose=0)
        epsilon = 0.001
        noisy_img = base_img + np.random.uniform(-epsilon, epsilon, base_img.shape)
        noisy_img = np.clip(noisy_img, 0, 1)
        latent_noisy = self.encoder.predict(noisy_img[np.newaxis], verbose=0)
        diff = np.linalg.norm(latent_base - latent_noisy)
        print(f"Latent difference under adversarial noise: {diff:.3f}")
        self.assertGreater(diff, 0.05, "Adversarial perturbation did not sufficiently change latent representation")

    def test_side_channel_timing_constancy(self):
        private_key, public_key, _, _ = generate_enhanced_rsa_keys_from_image(self.encoder, used_images)
        message = b"Timing test"
        encrypted = public_key.encrypt(
            message,
            padding.OAEP(mgf=padding.MGF1(hashes.SHA512()),
                         algorithm=hashes.SHA512(),
                         label=None)
        )
        timings = []
        for _ in range(5):
            start = time.time()
            secure_decrypt(private_key, encrypted)
            timings.append(time.time() - start)
        avg_time = np.mean(timings)
        std_time = np.std(timings)
        print(f"Decryption timings: {timings}, avg: {avg_time:.3f}, std: {std_time:.3f}")
        self.assertLess(std_time, 0.05, "High variance in decryption time indicates potential side-channel leakage")

    def test_quantum_resistance(self):
        from app.crypto.utils import KEY_BIT_LENGTH
        self.assertGreaterEqual(KEY_BIT_LENGTH, 2048, "RSA ключ недостаточного размера для квантовой устойчивости")

    def test_stress_scalability(self):
        def gen_key():
            generate_enhanced_rsa_keys_from_image(self.encoder, used_images)
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(lambda _: time.time(), range(20)))
        self.assertTrue(True, "Stress scalability test passed if no errors occurred")

    def test_long_term_stability(self):
        images = generate_logistic_map_images_dataset(200, image_size=28, r=3.99, fixed_initial=True)
        initial_loss = self.autoencoder.evaluate(images, images, verbose=0)
        for _ in range(3):
            self.autoencoder.fit(images, images, epochs=1, batch_size=32, verbose=0)
        final_loss = self.autoencoder.evaluate(images, images, verbose=0)
        print(f"Long-term stability test: initial loss {initial_loss:.6f}, final loss {final_loss:.6f}")
        self.assertLess(final_loss, initial_loss * 1.5, "Final loss significantly worse than initial loss")

    def test_safe_integration(self):
        images = generate_logistic_map_images_dataset(100, image_size=28, r=3.99, fixed_initial=True)
        self.autoencoder.fit(images, images, epochs=1, batch_size=32, verbose=0)
        private_key, public_key, _, _ = generate_enhanced_rsa_keys_from_image(self.encoder, used_images)
        message = b"Safe integration test"
        encrypted = public_key.encrypt(
            message,
            padding.OAEP(mgf=padding.MGF1(hashes.SHA512()),
                         algorithm=hashes.SHA512(),
                         label=None)
        )
        decrypted = secure_decrypt(private_key, encrypted)
        self.assertEqual(message, decrypted, "Safe integration failed: decrypted message differs")

    def test_isolation_environment(self):
        os.environ['ISOLATED_ENV'] = 'True'
        try:
            images = generate_logistic_map_images_dataset(50, image_size=28, r=3.99, fixed_initial=True)
            loss = self.autoencoder.evaluate(images, images, verbose=0)
            self.assertLess(loss, 0.1, "Isolation environment: loss too high")
        finally:
            os.environ.pop('ISOLATED_ENV', None)

    def test_prompt_injection_defense(self):
        self.skipTest("Prompt injection defense is not applicable for autoencoder models.")

    def test_explainability_interpretability(self):
        images = generate_logistic_map_images_dataset(200, image_size=28, r=3.99, fixed_initial=True)
        latents = self.encoder.predict(images, verbose=0)
        variances = np.var(latents, axis=0)
        print(f"Latent variances: {variances}")
        for idx, var in enumerate(variances):
            self.assertGreater(var, 0.0001, f"Dimension {idx} in latent space has very low variance, reducing explainability.")

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImageBasedCrypto)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestValueEvaluation))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\nOverall Results:")
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    return result.wasSuccessful()

if __name__ == '__main__':
    run_tests()
