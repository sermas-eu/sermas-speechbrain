# from speechbrain.inference import classifiers, interfaces


# run_opts = {
#     # enable CUDA
#     # "device": "cuda",
#     # "data_parallel_count": -1,
#     # "data_parallel_backend": False,
#     # "distributed_launch": False,
#     # "distributed_backend": "nccl",
#     # "jit_module_keys": None
# }

# speaker_encoder = classifiers.EncoderClassifier.from_hparams(
#     source="speechbrain/spkrec-xvect-voxceleb",
#     savedir='models/spkrec-xvect-voxceleb',
#     run_opts=run_opts
# )
#
# emotion_classifier = interfaces.foreign_class(
#     source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
#     pymodule_file="custom_interface.py",
#     classname="CustomEncoderWav2vec2Classifier",
#     savedir='models/emotion-recognition-wav2vec2-IEMOCAP',
#     run_opts=run_opts
# )
#
# language_classifier = classifiers.EncoderClassifier.from_hparams(
#     source="speechbrain/lang-id-commonlanguage_ecapa",
#     savedir="models/lang-id-commonlanguage_ecapa",
#     run_opts=run_opts
# )