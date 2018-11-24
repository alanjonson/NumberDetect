#用新api训练，训练过程中自带eval
python object_detection/model_main.py --model_dir=number_detection/training  --pipeline_config_path=number_detection/training/ssd_mobilenet_v2_number.config
#用 train.py训练
python object_detection/legacy/train.py --train_dir=number_detection/training --logtostderr --pipeline_config_path=number_detection/training/ssd_mobilenet_v2_number.config
#导出模型
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path number_detection/training/ssd_mobilenet_v2_number.config --trained_checkpoint_prefix number_detection/training/model.ckpt-112952 --output_directory number_detection/result
#导出ssd模型
python object_detection/export_tflite_ssd_graph.py --input_type image_tensor --pipeline_config_path number_detection/training/ssd_mobilenet_v2_number.config --trained_checkpoint_prefix number_detection/training/model.ckpt-112952 --output_directory number_detection/ssd_result
