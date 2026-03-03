import imghdr
import json

class MainOrchestrator:

    def detect(self, data):

        if isinstance(data, str):
            return "detected_text"

        if isinstance(data, dict):
            if "text" in data:
                return "detected_text"
            if "image" in data:
                return "detected_image"
            if "audio" in data:
                return "detected_audio"
            if "video" in data:
                return "detected_video"

        if isinstance(data, bytes):
            if imghdr.what(None, data) is not None:
                return "detected_image"

        return "unknown"

    def route(self, data):
        dtype = self.detect(data)
        print(f"[MainOrchestrator] Type → {dtype}")
        return dtype
