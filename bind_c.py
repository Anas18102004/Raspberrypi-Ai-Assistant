# laptop_server.py
import cv2
import socket
import struct
import pickle
import threading
import pyaudio

# ----------------------------
# Video Streaming
# ----------------------------
def video_stream(server_socket):
    cap = cv2.VideoCapture(0)  # Laptop camera
    conn, addr = server_socket.accept()
    print(f"[VIDEO] Connection from {addr}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        data = pickle.dumps(frame)
        size = len(data)
        try:
            conn.sendall(struct.pack(">L", size) + data)
        except Exception as e:
            print("[VIDEO ERROR]", e)
            break
    cap.release()
    conn.close()

# ----------------------------
# Audio Streaming
# ----------------------------
def audio_stream(server_socket):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    conn, addr = server_socket.accept()
    print(f"[AUDIO] Connection from {addr}")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            conn.sendall(data)
    except Exception as e:
        print("[AUDIO ERROR]", e)

    stream.stop_stream()
    stream.close()
    p.terminate()
    conn.close()

if __name__ == "__main__":
    # Video socket
    video_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_server.bind(("0.0.0.0", 9999))
    video_server.listen(1)

    # Audio socket
    audio_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    audio_server.bind(("0.0.0.0", 9998))
    audio_server.listen(1)

    print("[SERVER] Waiting for connections...")

    threading.Thread(target=video_stream, args=(video_server,)).start()
    threading.Thread(target=audio_stream, args=(audio_server,)).start()
