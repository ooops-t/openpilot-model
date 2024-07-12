# Run openpilot supercombo model use ONNX

## Download ONNXRuntime library
```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
tar -xvf onnxruntime-linux-x64-gpu-1.18.1.tgz
```

## Build
``` bash
clang++ main.cpp supercombomodel.cpp -Ionnxruntime-linux-x64-gpu-1.18.1/include -Lonnxruntime-linux-x64-gpu-1.18.1/lib -lonnxruntime -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
```

## Test video information

``` json
// ffprobe -v quiet -print_format json -show_format -show_streams  video.hevc
{
    "streams": [
        {
            "index": 0,
            "codec_name": "hevc",
            "codec_long_name": "H.265 / HEVC (High Efficiency Video Coding)",
            "profile": "Main",
            "codec_type": "video",
            "codec_tag_string": "[0][0][0][0]",
            "codec_tag": "0x0000",
            "width": 1164,
            "height": 874,
            "coded_width": 1184,
            "coded_height": 896,
            "closed_captions": 0,
            "film_grain": 0,
            "has_b_frames": 0,
            "pix_fmt": "yuv420p",
            "level": 93,
            "color_range": "tv",
            "chroma_location": "left",
            "refs": 1,
            "r_frame_rate": "25/1",
            "avg_frame_rate": "25/1",
            "time_base": "1/1200000",
            "extradata_size": 74,
            "disposition": {
                "default": 0,
                "dub": 0,
                "original": 0,
                "comment": 0,
                "lyrics": 0,
                "karaoke": 0,
                "forced": 0,
                "hearing_impaired": 0,
                "visual_impaired": 0,
                "clean_effects": 0,
                "attached_pic": 0,
                "timed_thumbnails": 0,
                "captions": 0,
                "descriptions": 0,
                "metadata": 0,
                "dependent": 0,
                "still_image": 0
            }
        }
    ],
    "format": {
        "filename": "video.hevc",
        "nb_streams": 1,
        "nb_programs": 0,
        "format_name": "hevc",
        "format_long_name": "raw HEVC video",
        "size": "37467994",
        "probe_score": 51
    }
}
```