# ONNX validation (laptop, ONNX Runtime CPU vs PyTorch)

40 frames from tilt.mp4, webcam_clip.mp4 resized to 1280x720; person class, conf > 0.25. Boxes matched one-to-one at IoU >= 0.5.

A = same letterboxed image through the .pt and the ONNX, same head (export numerics): `yolo26n_*` = NMS-free head (.pt with nms=False), `yolo26n-raw_*` / `yolo11n_*` = raw head + NMS (.pt default).  
B = our letterbox + decode + un-letterbox (= the C++ path) vs Ultralytics predicting with the same ONNX.

| file | A recall | A precision | A IoU mean | A IoU min | A max abs(dconf) | B IoU mean | B recall | B precision | ORT CPU ms |
|---|---|---|---|---|---|---|---|---|---|
| yolo11n_640x384.onnx | 1.000 | 1.000 | 1.0000 | 1.0000 | 0.0000 | 1.00000 | 1.000 | 1.000 | 28.3 |
| yolo11n_640x640.onnx | 1.000 | 1.000 | 1.0000 | 1.0000 | 0.0000 | 1.00000 | 1.000 | 1.000 | 44.6 |
| yolo26n-raw_640x384.onnx | 1.000 | 1.000 | 0.9999 | 0.9962 | 0.0000 | 1.00000 | 1.000 | 1.000 | 23.0 |
| yolo26n-raw_640x640.onnx | 1.000 | 1.000 | 0.9999 | 0.9947 | 0.0000 | 1.00000 | 1.000 | 1.000 | 37.9 |
| yolo26n_320x192.onnx | 1.000 | 1.000 | 1.0000 | 0.9999 | 0.0000 | 1.00000 | 1.000 | 1.000 | 7.6 |
| yolo26n_320x320.onnx | 1.000 | 1.000 | 1.0000 | 1.0000 | 0.0000 | 1.00000 | 1.000 | 1.000 | 11.4 |
| yolo26n_416x256.onnx | 1.000 | 1.000 | 0.9999 | 0.9972 | 0.0000 | 1.00000 | 1.000 | 1.000 | 11.6 |
| yolo26n_416x416.onnx | 1.000 | 1.000 | 0.9999 | 0.9972 | 0.0000 | 1.00000 | 1.000 | 1.000 | 18.0 |
| yolo26n_640x384.onnx | 1.000 | 1.000 | 0.9999 | 0.9952 | 0.0000 | 1.00000 | 1.000 | 1.000 | 23.6 |
| yolo26n_640x640.onnx | 1.000 | 1.000 | 0.9999 | 0.9937 | 0.0000 | 1.00000 | 1.000 | 1.000 | 38.8 |
