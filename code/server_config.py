image_data = {
    'dataType': 'image',
    'filepath': '../data/test/1.png'
}
video_data = {
    'dataType': 'video',
    'filepath': '../data/test/2.mp4'
}
# xception_240_98.723483_220705050224.pth
# ../checkpoints/BSL/xceptionbsl_ghm_full_204_trt_A10_32.pth
image_servers_data = [
    {
        'device_id': '0', 'port': 10101, 'init_data': image_data,
         'facedetect_batchsize': 1, 
        'facedetect_model_path': '../checkpoints/retinaface/Retinaface_mobilenet0.25.pth', 
        'frames_per_video': 2,
        'deepfake_batchsize': 4, 
        'deepfake_model_path': '../checkpoints/capsule/capsule_11.pt'
    }
]

video_servers_data = [
    {
        'device_id': '0', 'port': 10201, 'init_data': video_data,
        'facedetect_batchsize': 16, 
        'facedetect_model_path': '../checkpoints/retinaface/Retinaface_mobilenet0.25.pth', 
        'frames_per_video': 2,
        'deepfake_batchsize': 8, 
        'deepfake_model_path': '../checkpoints/capsule/capsule_11.pt'
    }
]
