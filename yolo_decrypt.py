"""Decrypt the YOLOv11 .pt.enc weights (Stage Y).

Replicates the public loader's own decrypt path (weights/loader.pyc),
because that .pyc is compiled for Python 3.12 and Colab runs Python 3.13
("bad magic number"). The loader's logic, read from the .pyc constants:
  - password: the loader's DEFAULT_PASSWORD constant
    (env YOLOV11_WEIGHTS_PASSWORD override, else the embedded default)
  - KDF: scrypt N=32768, r=8, p=1, 32-byte key (AES-256)
  - format: magic b'G5ENCv1' | salt(32) | nonce(12) | ciphertext | tag(16)
  - verification: SHA-256 of the plaintext against the .meta.txt
    (the authors publish sha256_plain in each .meta.txt)
This script only verifies the logic locally; the Colab run script carries
the same decrypt function and does the decrypt in-session.
"""
import hashlib
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

MAGIC = b"G5ENCv1"
DEFAULT_PASSWORD = "20divhno34kejh"  # loader's embedded default (public .pyc constant)


def decrypt_file(enc_path, out_path, meta_path, password=None):
    pw = password or os.environ.get("YOLOV11_WEIGHTS_PASSWORD", DEFAULT_PASSWORD)
    data = Path(enc_path).read_bytes()
    assert data[:7] == MAGIC, f"bad magic in {enc_path}"
    salt, nonce = data[7:39], data[39:51]
    tag, ciphertext = data[-16:], data[51:-16]
    kdf = Scrypt(salt=salt, length=32, n=32768, r=8, p=1)
    key = kdf.derive(pw.encode("utf-8"))
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
    dec = cipher.decryptor()
    dec.authenticate_additional_data(MAGIC)  # loader line 94: AAD = magic bytes
    plain = dec.update(ciphertext) + dec.finalize()
    # verify against the author's published sha256_plain
    meta = dict(l.split("=", 1) for l in Path(meta_path).read_text().splitlines() if "=" in l)
    expected = meta["sha256_plain"].strip().lower()
    actual = hashlib.sha256(plain).hexdigest()
    assert actual == expected, f"SHA256 mismatch {enc_path}: {actual} != {expected}"
    Path(out_path).write_bytes(plain)
    return len(plain), actual


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent / "yolo_repo" / "weights"
    outdir = Path(__file__).resolve().parent / "yolo_weights_plain"
    outdir.mkdir(exist_ok=True)
    for v in ["n", "s", "m", "l"]:
        enc = repo / f"best_{v}.pt.enc"
        meta = repo / f"best_{v}.pt.enc.meta.txt"
        out = outdir / f"best_{v}.pt"
        n, sha = decrypt_file(enc, out, meta)
        print(f"{v}: {n} bytes, sha256 {sha[:16]}... OK")
    print("all 4 decrypted + SHA-256 verified against the authors' .meta.txt")
