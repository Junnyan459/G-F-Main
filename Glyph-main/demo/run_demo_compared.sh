vllm serve YOUR_MODEL_PATH_Qwen3 --port 8000 --served-model-name Qwen3-8B --max-model-len 131072 --trust-remote-code --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'

vllm serve YOUR_MODEL_PATH --port 5002 --served-model-name glyph --allowed-local-media-path / --media-io-kwargs '{"video": {"num_frames": -1}}'

python inference_pipeline_gradio_flow_en_only_glyph.py