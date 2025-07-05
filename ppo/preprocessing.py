import cv2

def preprocess_observation(observation):
    # Converti in scala di grigi
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    # Ridimensiona a 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalizza
    normalized = resized / 255.0
    return normalized