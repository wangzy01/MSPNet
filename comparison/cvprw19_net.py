import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift
class CAPrivacyActionRecognizer:
    def __init__(self, input_shape=(256, 256), crop_shape=(224, 224), epsilon=1e-3, keep_resolution=False):
        self.h, self.w = input_shape
        self.crop_h, self.crop_w = crop_shape
        self.epsilon = epsilon
        self.keep_resolution = keep_resolution
    def _phase_correlation(self, F1, F2):
        cross_power = np.conj(F1) * F2
        cross_power /= (np.abs(cross_power) + self.epsilon)
        correlation_map = np.real(ifft2(cross_power))
        correlation_map = fftshift(correlation_map)
        return correlation_map
    def extract_T_feature(self, img1, img2):
        F1 = fft2(img1)
        F2 = fft2(img2)
        t_feature = self._phase_correlation(F1, F2)
        return t_feature
    def extract_RS_feature(self, img1, img2):
        M1 = np.abs(fftshift(fft2(img1)))
        M2 = np.abs(fftshift(fft2(img2)))
        h, w = img1.shape
        center = (w // 2, h // 2)
        max_radius = min(center)
        lp1 = cv2.warpPolar(M1, (w, h), center, max_radius, cv2.WARP_POLAR_LOG)
        lp2 = cv2.warpPolar(M2, (w, h), center, max_radius, cv2.WARP_POLAR_LOG)
        LP_F1 = fft2(lp1)
        LP_F2 = fft2(lp2)
        rs_feature = self._phase_correlation(LP_F1, LP_F2)
        return rs_feature
    def crop_center(self, img):
        y, x = img.shape
        start_x = x // 2 - self.crop_w // 2
        start_y = y // 2 - self.crop_h // 2
        return img[start_y:start_y+self.crop_h, start_x:start_x+self.crop_w]
    def process_rs_feature(self, rs_map):
        h, w = rs_map.shape
        start_x = w // 2 - self.crop_w // 2
        rs_cropped = rs_map[:, start_x:start_x + self.crop_w]
        rs_resized = cv2.resize(rs_cropped, (self.crop_w, self.crop_h), interpolation=cv2.INTER_LINEAR)
        return rs_resized
    def process_video_clip(self, ca_frames, strides=[2, 3, 4, 6]):
        features_list = []
        num_frames = len(ca_frames)
        for s in strides:
            max_i = (num_frames - 1 - s) // s
            for i in range(max_i + 1):
                frame_curr = ca_frames[i * s]
                frame_next = ca_frames[i * s + s]
                t_map = self.extract_T_feature(frame_curr, frame_next)
                rs_map = self.extract_RS_feature(frame_curr, frame_next)
                if self.keep_resolution:
                    t_map_final = t_map
                    rs_map_final = rs_map
                else:
                    t_map_final = self.crop_center(t_map)
                    rs_map_final = self.process_rs_feature(rs_map)
                t_norm = (t_map_final - t_map_final.min()) / (t_map_final.max() - t_map_final.min() + 1e-8)
                rs_norm = (rs_map_final - rs_map_final.min()) / (rs_map_final.max() - rs_map_final.min() + 1e-8)
                features_list.append(t_norm)
                features_list.append(rs_norm)
        if not features_list:
            raise ValueError("Video clip too short for the requested strides.")
        stacked_features = np.dstack(features_list)
        return stacked_features
class ActionClassifierNetwork:
    def __init__(self, input_channels, num_classes):
        self.model_name = "VGG-16"
        pass
    def predict(self, trs_features):
        print(f"Running {self.model_name} on input shape {trs_features.shape}...")
        return "Jump (Example Prediction)"
if __name__ == "__main__":
    dummy_video_clip = [np.random.rand(256, 256).astype(np.float32) for _ in range(20)]
    feature_extractor = CAPrivacyActionRecognizer()
    print("Extracting MS-TRS features (Phase Correlation & Log-Polar)...")
    try:
        trs_input_tensor = feature_extractor.process_video_clip(dummy_video_clip, strides=[2, 4])
        print(f"Feature extraction completed。")
        print(f"Output tensor shape: {trs_input_tensor.shape}")
        classifier = ActionClassifierNetwork(input_channels=trs_input_tensor.shape[2], num_classes=10)
        result = classifier.predict(trs_input_tensor)
        print(f"Recognition result: {result}")
    except ValueError as e:
        print(f"Error: {e}")
