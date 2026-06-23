vllm serve YOUR_MODEL_PATH --port 5002 --served-model-name glyph --allowed-local-media-path / --media-io-kwargs '{"video": {"num_frames": -1}}'

python inference_pipeline_gradio_flow_en.py