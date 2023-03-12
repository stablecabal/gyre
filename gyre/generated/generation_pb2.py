# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: generation.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import tensors_pb2 as tensors__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10generation.proto\x12\x07gooseai\x1a\rtensors.proto\"/\n\x05Token\x12\x11\n\x04text\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\n\n\x02id\x18\x02 \x01(\rB\x07\n\x05_text\"T\n\x06Tokens\x12\x1e\n\x06tokens\x18\x01 \x03(\x0b\x32\x0e.gooseai.Token\x12\x19\n\x0ctokenizer_id\x18\x02 \x01(\tH\x00\x88\x01\x01\x42\x0f\n\r_tokenizer_id\"X\n\x18ImageAdjustment_Gaussian\x12\r\n\x05sigma\x18\x01 \x01(\x02\x12-\n\tdirection\x18\x02 \x01(\x0e\x32\x1a.gooseai.GaussianDirection\"\x18\n\x16ImageAdjustment_Invert\"h\n\x16ImageAdjustment_Levels\x12\x11\n\tinput_low\x18\x01 \x01(\x02\x12\x12\n\ninput_high\x18\x02 \x01(\x02\x12\x12\n\noutput_low\x18\x03 \x01(\x02\x12\x13\n\x0boutput_high\x18\x04 \x01(\x02\"\xd2\x01\n\x18ImageAdjustment_Channels\x12&\n\x01r\x18\x01 \x01(\x0e\x32\x16.gooseai.ChannelSourceH\x00\x88\x01\x01\x12&\n\x01g\x18\x02 \x01(\x0e\x32\x16.gooseai.ChannelSourceH\x01\x88\x01\x01\x12&\n\x01\x62\x18\x03 \x01(\x0e\x32\x16.gooseai.ChannelSourceH\x02\x88\x01\x01\x12&\n\x01\x61\x18\x04 \x01(\x0e\x32\x16.gooseai.ChannelSourceH\x03\x88\x01\x01\x42\x04\n\x02_rB\x04\n\x02_gB\x04\n\x02_bB\x04\n\x02_a\"t\n\x17ImageAdjustment_Rescale\x12\x0e\n\x06height\x18\x01 \x01(\x04\x12\r\n\x05width\x18\x02 \x01(\x04\x12\"\n\x04mode\x18\x03 \x01(\x0e\x32\x14.gooseai.RescaleMode\x12\x16\n\x0e\x61lgorithm_hint\x18\x04 \x03(\t\"P\n\x14ImageAdjustment_Crop\x12\x0b\n\x03top\x18\x01 \x01(\x04\x12\x0c\n\x04left\x18\x02 \x01(\x04\x12\r\n\x05width\x18\x03 \x01(\x04\x12\x0e\n\x06height\x18\x04 \x01(\x04\"2\n\x15ImageAdjustment_Depth\x12\x19\n\x11\x64\x65pth_engine_hint\x18\x01 \x03(\t\"J\n\x19ImageAdjustment_CannyEdge\x12\x15\n\rlow_threshold\x18\x01 \x01(\x02\x12\x16\n\x0ehigh_threshold\x18\x02 \x01(\x02\"\x1f\n\x1dImageAdjustment_EdgeDetection\"\x1e\n\x1cImageAdjustment_Segmentation\"\xbf\x04\n\x0fImageAdjustment\x12\x31\n\x04\x62lur\x18\x01 \x01(\x0b\x32!.gooseai.ImageAdjustment_GaussianH\x00\x12\x31\n\x06invert\x18\x02 \x01(\x0b\x32\x1f.gooseai.ImageAdjustment_InvertH\x00\x12\x31\n\x06levels\x18\x03 \x01(\x0b\x32\x1f.gooseai.ImageAdjustment_LevelsH\x00\x12\x35\n\x08\x63hannels\x18\x04 \x01(\x0b\x32!.gooseai.ImageAdjustment_ChannelsH\x00\x12\x33\n\x07rescale\x18\x05 \x01(\x0b\x32 .gooseai.ImageAdjustment_RescaleH\x00\x12-\n\x04\x63rop\x18\x06 \x01(\x0b\x32\x1d.gooseai.ImageAdjustment_CropH\x00\x12/\n\x05\x64\x65pth\x18\x07 \x01(\x0b\x32\x1e.gooseai.ImageAdjustment_DepthH\x00\x12\x38\n\ncanny_edge\x18\x08 \x01(\x0b\x32\".gooseai.ImageAdjustment_CannyEdgeH\x00\x12@\n\x0e\x65\x64ge_detection\x18\t \x01(\x0b\x32&.gooseai.ImageAdjustment_EdgeDetectionH\x00\x12=\n\x0csegmentation\x18\n \x01(\x0b\x32%.gooseai.ImageAdjustment_SegmentationH\x00\x42\x0c\n\nadjustment\"-\n\x0fSafetensorsMeta\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"A\n\x11SafetensorsTensor\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1f\n\x06tensor\x18\x02 \x01(\x0b\x32\x0f.tensors.Tensor\"f\n\x0bSafetensors\x12*\n\x08metadata\x18\x01 \x03(\x0b\x32\x18.gooseai.SafetensorsMeta\x12+\n\x07tensors\x18\x02 \x03(\x0b\x32\x1a.gooseai.SafetensorsTensor\"0\n\nLoraWeight\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x0e\n\x06weight\x18\x02 \x01(\x02\"T\n\x04Lora\x12\"\n\x04lora\x18\x01 \x01(\x0b\x32\x14.gooseai.Safetensors\x12(\n\x07weights\x18\x02 \x03(\x0b\x32\x13.gooseai.LoraWeightB\x02\x18\x01\"e\n\x11\x41rtifactReference\x12\x0c\n\x02id\x18\x01 \x01(\x04H\x00\x12\x0e\n\x04uuid\x18\x02 \x01(\tH\x00\x12%\n\x05stage\x18\x03 \x01(\x0e\x32\x16.gooseai.ArtifactStageB\x0b\n\treference\"?\n\x0eTokenEmbedding\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x1f\n\x06tensor\x18\x02 \x01(\x0b\x32\x0f.tensors.Tensor\"X\n\x0c\x43\x61\x63heControl\x12\x10\n\x08\x63\x61\x63he_id\x18\x01 \x01(\t\x12\x0f\n\x07max_age\x18\x02 \x01(\r\x12%\n\x05stage\x18\x03 \x01(\x0e\x32\x16.gooseai.ArtifactStage\"\xac\x06\n\x08\x41rtifact\x12\n\n\x02id\x18\x01 \x01(\x04\x12#\n\x04type\x18\x02 \x01(\x0e\x32\x15.gooseai.ArtifactType\x12\x0c\n\x04mime\x18\x03 \x01(\t\x12\x12\n\x05magic\x18\x04 \x01(\tH\x01\x88\x01\x01\x12\x10\n\x06\x62inary\x18\x05 \x01(\x0cH\x00\x12\x0e\n\x04text\x18\x06 \x01(\tH\x00\x12!\n\x06tokens\x18\x07 \x01(\x0b\x32\x0f.gooseai.TokensH\x00\x12\x33\n\nclassifier\x18\x0b \x01(\x0b\x32\x1d.gooseai.ClassifierParametersH\x00\x12!\n\x06tensor\x18\x0e \x01(\x0b\x32\x0f.tensors.TensorH\x00\x12*\n\x03ref\x18\xff\x03 \x01(\x0b\x32\x1a.gooseai.ArtifactReferenceH\x00\x12\x0e\n\x03url\x18\x81\x04 \x01(\tH\x00\x12,\n\x0bsafetensors\x18\x82\x04 \x01(\x0b\x32\x14.gooseai.SafetensorsH\x00\x12\x13\n\x08\x63\x61\x63he_id\x18\xa6\x04 \x01(\tH\x00\x12\"\n\x04lora\x18\xfe\x03 \x01(\x0b\x32\r.gooseai.LoraB\x02\x18\x01H\x00\x12\x37\n\x0ftoken_embedding\x18\x80\x04 \x01(\x0b\x32\x17.gooseai.TokenEmbeddingB\x02\x18\x01H\x00\x12\r\n\x05index\x18\x08 \x01(\r\x12,\n\rfinish_reason\x18\t \x01(\x0e\x32\x15.gooseai.FinishReason\x12\x0c\n\x04seed\x18\n \x01(\r\x12\x0c\n\x04uuid\x18\x0c \x01(\t\x12\x0c\n\x04size\x18\r \x01(\x04\x12.\n\x0b\x61\x64justments\x18\xf4\x03 \x03(\x0b\x32\x18.gooseai.ImageAdjustment\x12\x32\n\x0fpostAdjustments\x18\xf5\x03 \x03(\x0b\x32\x18.gooseai.ImageAdjustment\x12\x1d\n\x0fhint_image_type\x18\x88\x04 \x01(\tH\x02\x88\x01\x01\x12\x32\n\rcache_control\x18\xa7\x04 \x01(\x0b\x32\x15.gooseai.CacheControlH\x03\x88\x01\x01\x42\x06\n\x04\x64\x61taB\x08\n\x06_magicB\x12\n\x10_hint_image_typeB\x10\n\x0e_cache_control\"+\n\x0bNamedWeight\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06weight\x18\x02 \x01(\x02\"N\n\rTokenOverride\x12\r\n\x05token\x18\x01 \x01(\t\x12\x1b\n\x0eoriginal_token\x18\x02 \x01(\tH\x00\x88\x01\x01\x42\x11\n\x0f_original_token\"\xae\x01\n\x10PromptParameters\x12\x11\n\x04init\x18\x01 \x01(\x08H\x00\x88\x01\x01\x12\x13\n\x06weight\x18\x02 \x01(\x02H\x01\x88\x01\x01\x12,\n\rnamed_weights\x18\xf4\x03 \x03(\x0b\x32\x14.gooseai.NamedWeight\x12\x30\n\x0ftoken_overrides\x18\xf5\x03 \x03(\x0b\x32\x16.gooseai.TokenOverrideB\x07\n\x05_initB\t\n\x07_weight\"\xcb\x01\n\x06Prompt\x12\x32\n\nparameters\x18\x01 \x01(\x0b\x32\x19.gooseai.PromptParametersH\x01\x88\x01\x01\x12\x0e\n\x04text\x18\x02 \x01(\tH\x00\x12!\n\x06tokens\x18\x03 \x01(\x0b\x32\x0f.gooseai.TokensH\x00\x12%\n\x08\x61rtifact\x18\x04 \x01(\x0b\x32\x11.gooseai.ArtifactH\x00\x12\x1a\n\x11include_in_answer\x18\xf4\x03 \x01(\x08\x42\x08\n\x06promptB\r\n\x0b_parameters\"\x85\x01\n\x0fSigmaParameters\x12\x16\n\tsigma_min\x18\x01 \x01(\x02H\x00\x88\x01\x01\x12\x16\n\tsigma_max\x18\x02 \x01(\x02H\x01\x88\x01\x01\x12\x17\n\nkarras_rho\x18\n \x01(\x02H\x02\x88\x01\x01\x42\x0c\n\n_sigma_minB\x0c\n\n_sigma_maxB\r\n\x0b_karras_rho\"n\n\rChurnSettings\x12\r\n\x05\x63hurn\x18\x01 \x01(\x02\x12\x17\n\nchurn_tmin\x18\x02 \x01(\x02H\x00\x88\x01\x01\x12\x17\n\nchurn_tmax\x18\x03 \x01(\x02H\x01\x88\x01\x01\x42\r\n\x0b_churn_tminB\r\n\x0b_churn_tmax\"\x8b\x04\n\x11SamplerParameters\x12\x10\n\x03\x65ta\x18\x01 \x01(\x02H\x00\x88\x01\x01\x12\x1b\n\x0esampling_steps\x18\x02 \x01(\x04H\x01\x88\x01\x01\x12\x1c\n\x0flatent_channels\x18\x03 \x01(\x04H\x02\x88\x01\x01\x12 \n\x13\x64ownsampling_factor\x18\x04 \x01(\x04H\x03\x88\x01\x01\x12\x16\n\tcfg_scale\x18\x05 \x01(\x02H\x04\x88\x01\x01\x12\x1d\n\x10init_noise_scale\x18\x06 \x01(\x02H\x05\x88\x01\x01\x12\x1d\n\x10step_noise_scale\x18\x07 \x01(\x02H\x06\x88\x01\x01\x12+\n\x05\x63hurn\x18\xf4\x03 \x01(\x0b\x32\x16.gooseai.ChurnSettingsH\x07\x88\x01\x01\x12-\n\x05sigma\x18\xf5\x03 \x01(\x0b\x32\x18.gooseai.SigmaParametersH\x08\x88\x01\x01\x12\x33\n\nnoise_type\x18\xf6\x03 \x01(\x0e\x32\x19.gooseai.SamplerNoiseTypeH\t\x88\x01\x01\x42\x06\n\x04_etaB\x11\n\x0f_sampling_stepsB\x12\n\x10_latent_channelsB\x16\n\x14_downsampling_factorB\x0c\n\n_cfg_scaleB\x13\n\x11_init_noise_scaleB\x13\n\x11_step_noise_scaleB\x08\n\x06_churnB\x08\n\x06_sigmaB\r\n\x0b_noise_type\"\x8b\x01\n\x15\x43onditionerParameters\x12 \n\x13vector_adjust_prior\x18\x01 \x01(\tH\x00\x88\x01\x01\x12(\n\x0b\x63onditioner\x18\x02 \x01(\x0b\x32\x0e.gooseai.ModelH\x01\x88\x01\x01\x42\x16\n\x14_vector_adjust_priorB\x0e\n\x0c_conditioner\"j\n\x12ScheduleParameters\x12\x12\n\x05start\x18\x01 \x01(\x02H\x00\x88\x01\x01\x12\x10\n\x03\x65nd\x18\x02 \x01(\x02H\x01\x88\x01\x01\x12\x12\n\x05value\x18\x03 \x01(\x02H\x02\x88\x01\x01\x42\x08\n\x06_startB\x06\n\x04_endB\x08\n\x06_value\"\xe4\x01\n\rStepParameter\x12\x13\n\x0bscaled_step\x18\x01 \x01(\x02\x12\x30\n\x07sampler\x18\x02 \x01(\x0b\x32\x1a.gooseai.SamplerParametersH\x00\x88\x01\x01\x12\x32\n\x08schedule\x18\x03 \x01(\x0b\x32\x1b.gooseai.ScheduleParametersH\x01\x88\x01\x01\x12\x32\n\x08guidance\x18\x04 \x01(\x0b\x32\x1b.gooseai.GuidanceParametersH\x02\x88\x01\x01\x42\n\n\x08_samplerB\x0b\n\t_scheduleB\x0b\n\t_guidance\"\x97\x01\n\x05Model\x12\x30\n\x0c\x61rchitecture\x18\x01 \x01(\x0e\x32\x1a.gooseai.ModelArchitecture\x12\x11\n\tpublisher\x18\x02 \x01(\t\x12\x0f\n\x07\x64\x61taset\x18\x03 \x01(\t\x12\x0f\n\x07version\x18\x04 \x01(\x02\x12\x18\n\x10semantic_version\x18\x05 \x01(\t\x12\r\n\x05\x61lias\x18\x06 \x01(\t\"\xbc\x01\n\x10\x43utoutParameters\x12*\n\x07\x63utouts\x18\x01 \x03(\x0b\x32\x19.gooseai.CutoutParameters\x12\x12\n\x05\x63ount\x18\x02 \x01(\rH\x00\x88\x01\x01\x12\x11\n\x04gray\x18\x03 \x01(\x02H\x01\x88\x01\x01\x12\x11\n\x04\x62lur\x18\x04 \x01(\x02H\x02\x88\x01\x01\x12\x17\n\nsize_power\x18\x05 \x01(\x02H\x03\x88\x01\x01\x42\x08\n\x06_countB\x07\n\x05_grayB\x07\n\x05_blurB\r\n\x0b_size_power\"=\n\x1aGuidanceScheduleParameters\x12\x10\n\x08\x64uration\x18\x01 \x01(\x02\x12\r\n\x05value\x18\x02 \x01(\x02\"\x97\x02\n\x1aGuidanceInstanceParameters\x12\x1e\n\x06models\x18\x02 \x03(\x0b\x32\x0e.gooseai.Model\x12\x1e\n\x11guidance_strength\x18\x03 \x01(\x02H\x00\x88\x01\x01\x12\x35\n\x08schedule\x18\x04 \x03(\x0b\x32#.gooseai.GuidanceScheduleParameters\x12/\n\x07\x63utouts\x18\x05 \x01(\x0b\x32\x19.gooseai.CutoutParametersH\x01\x88\x01\x01\x12$\n\x06prompt\x18\x06 \x01(\x0b\x32\x0f.gooseai.PromptH\x02\x88\x01\x01\x42\x14\n\x12_guidance_strengthB\n\n\x08_cutoutsB\t\n\x07_prompt\"~\n\x12GuidanceParameters\x12\x30\n\x0fguidance_preset\x18\x01 \x01(\x0e\x32\x17.gooseai.GuidancePreset\x12\x36\n\tinstances\x18\x02 \x03(\x0b\x32#.gooseai.GuidanceInstanceParameters\"n\n\rTransformType\x12.\n\tdiffusion\x18\x01 \x01(\x0e\x32\x19.gooseai.DiffusionSamplerH\x00\x12%\n\x08upscaler\x18\x02 \x01(\x0e\x32\x11.gooseai.UpscalerH\x00\x42\x06\n\x04type\"Y\n\x11\x45xtendedParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x05\x66loat\x18\x02 \x01(\x02H\x00\x12\r\n\x03int\x18\x03 \x01(\x04H\x00\x12\r\n\x03str\x18\x04 \x01(\tH\x00\x42\x07\n\x05value\"D\n\x12\x45xtendedParameters\x12.\n\nparameters\x18\x01 \x03(\x0b\x32\x1a.gooseai.ExtendedParameter\"P\n\x12HiresFixParameters\x12\x0e\n\x06\x65nable\x18\x01 \x01(\x08\x12\x19\n\x0coos_fraction\x18\x02 \x01(\x02H\x00\x88\x01\x01\x42\x0f\n\r_oos_fraction\"\xa8\x05\n\x0fImageParameters\x12\x13\n\x06height\x18\x01 \x01(\x04H\x00\x88\x01\x01\x12\x12\n\x05width\x18\x02 \x01(\x04H\x01\x88\x01\x01\x12\x0c\n\x04seed\x18\x03 \x03(\r\x12\x14\n\x07samples\x18\x04 \x01(\x04H\x02\x88\x01\x01\x12\x12\n\x05steps\x18\x05 \x01(\x04H\x03\x88\x01\x01\x12.\n\ttransform\x18\x06 \x01(\x0b\x32\x16.gooseai.TransformTypeH\x04\x88\x01\x01\x12*\n\nparameters\x18\x07 \x03(\x0b\x32\x16.gooseai.StepParameter\x12\x36\n\x10masked_area_init\x18\x08 \x01(\x0e\x32\x17.gooseai.MaskedAreaInitH\x05\x88\x01\x01\x12\x31\n\rweight_method\x18\t \x01(\x0e\x32\x15.gooseai.WeightMethodH\x06\x88\x01\x01\x12\x15\n\x08quantize\x18\n \x01(\x08H\x07\x88\x01\x01\x12\x34\n\textension\x18\xf4\x03 \x01(\x0b\x32\x1b.gooseai.ExtendedParametersH\x08\x88\x01\x01\x12\x30\n\x05hires\x18\xfe\x03 \x01(\x0b\x32\x1b.gooseai.HiresFixParametersH\t\x88\x01\x01\x12\x14\n\x06tiling\x18\x88\x04 \x01(\x08H\n\x88\x01\x01\x12\x16\n\x08tiling_x\x18\x89\x04 \x01(\x08H\x0b\x88\x01\x01\x12\x16\n\x08tiling_y\x18\x8a\x04 \x01(\x08H\x0c\x88\x01\x01\x42\t\n\x07_heightB\x08\n\x06_widthB\n\n\x08_samplesB\x08\n\x06_stepsB\x0c\n\n_transformB\x13\n\x11_masked_area_initB\x10\n\x0e_weight_methodB\x0b\n\t_quantizeB\x0c\n\n_extensionB\x08\n\x06_hiresB\t\n\x07_tilingB\x0b\n\t_tiling_xB\x0b\n\t_tiling_y\"J\n\x11\x43lassifierConcept\x12\x0f\n\x07\x63oncept\x18\x01 \x01(\t\x12\x16\n\tthreshold\x18\x02 \x01(\x02H\x00\x88\x01\x01\x42\x0c\n\n_threshold\"\xf4\x01\n\x12\x43lassifierCategory\x12\x0c\n\x04name\x18\x01 \x01(\t\x12,\n\x08\x63oncepts\x18\x02 \x03(\x0b\x32\x1a.gooseai.ClassifierConcept\x12\x17\n\nadjustment\x18\x03 \x01(\x02H\x00\x88\x01\x01\x12$\n\x06\x61\x63tion\x18\x04 \x01(\x0e\x32\x0f.gooseai.ActionH\x01\x88\x01\x01\x12\x35\n\x0f\x63lassifier_mode\x18\x05 \x01(\x0e\x32\x17.gooseai.ClassifierModeH\x02\x88\x01\x01\x42\r\n\x0b_adjustmentB\t\n\x07_actionB\x12\n\x10_classifier_mode\"\xb8\x01\n\x14\x43lassifierParameters\x12/\n\ncategories\x18\x01 \x03(\x0b\x32\x1b.gooseai.ClassifierCategory\x12,\n\x07\x65xceeds\x18\x02 \x03(\x0b\x32\x1b.gooseai.ClassifierCategory\x12-\n\x0frealized_action\x18\x03 \x01(\x0e\x32\x0f.gooseai.ActionH\x00\x88\x01\x01\x42\x12\n\x10_realized_action\"k\n\x0f\x41ssetParameters\x12$\n\x06\x61\x63tion\x18\x01 \x01(\x0e\x32\x14.gooseai.AssetAction\x12\x12\n\nproject_id\x18\x02 \x01(\t\x12\x1e\n\x03use\x18\x03 \x01(\x0e\x32\x11.gooseai.AssetUse\"\x94\x01\n\nAnswerMeta\x12\x13\n\x06gpu_id\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x13\n\x06\x63pu_id\x18\x02 \x01(\tH\x01\x88\x01\x01\x12\x14\n\x07node_id\x18\x03 \x01(\tH\x02\x88\x01\x01\x12\x16\n\tengine_id\x18\x04 \x01(\tH\x03\x88\x01\x01\x42\t\n\x07_gpu_idB\t\n\x07_cpu_idB\n\n\x08_node_idB\x0c\n\n_engine_id\"\xa9\x01\n\x06\x41nswer\x12\x11\n\tanswer_id\x18\x01 \x01(\t\x12\x12\n\nrequest_id\x18\x02 \x01(\t\x12\x10\n\x08received\x18\x03 \x01(\x04\x12\x0f\n\x07\x63reated\x18\x04 \x01(\x04\x12&\n\x04meta\x18\x06 \x01(\x0b\x32\x13.gooseai.AnswerMetaH\x00\x88\x01\x01\x12$\n\tartifacts\x18\x07 \x03(\x0b\x32\x11.gooseai.ArtifactB\x07\n\x05_meta\"\xeb\x02\n\x07Request\x12\x11\n\tengine_id\x18\x01 \x01(\t\x12\x12\n\nrequest_id\x18\x02 \x01(\t\x12-\n\x0erequested_type\x18\x03 \x01(\x0e\x32\x15.gooseai.ArtifactType\x12\x1f\n\x06prompt\x18\x04 \x03(\x0b\x32\x0f.gooseai.Prompt\x12)\n\x05image\x18\x05 \x01(\x0b\x32\x18.gooseai.ImageParametersH\x00\x12\x33\n\nclassifier\x18\x07 \x01(\x0b\x32\x1d.gooseai.ClassifierParametersH\x00\x12)\n\x05\x61sset\x18\x08 \x01(\x0b\x32\x18.gooseai.AssetParametersH\x00\x12\x38\n\x0b\x63onditioner\x18\x06 \x01(\x0b\x32\x1e.gooseai.ConditionerParametersH\x01\x88\x01\x01\x42\x08\n\x06paramsB\x0e\n\x0c_conditionerJ\x04\x08\t\x10\nJ\x04\x08\n\x10\x0b\"w\n\x08OnStatus\x12%\n\x06reason\x18\x01 \x03(\x0e\x32\x15.gooseai.FinishReason\x12\x13\n\x06target\x18\x02 \x01(\tH\x00\x88\x01\x01\x12$\n\x06\x61\x63tion\x18\x03 \x03(\x0e\x32\x14.gooseai.StageActionB\t\n\x07_target\"\\\n\x05Stage\x12\n\n\x02id\x18\x01 \x01(\t\x12!\n\x07request\x18\x02 \x01(\x0b\x32\x10.gooseai.Request\x12$\n\ton_status\x18\x03 \x03(\x0b\x32\x11.gooseai.OnStatus\"A\n\x0c\x43hainRequest\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x1d\n\x05stage\x18\x02 \x03(\x0b\x32\x0e.gooseai.Stage\",\n\x0b\x41syncStatus\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\"f\n\x0b\x41syncAnswer\x12\x1f\n\x06\x61nswer\x18\x01 \x03(\x0b\x32\x0f.gooseai.Answer\x12\x10\n\x08\x63omplete\x18\x02 \x01(\x08\x12$\n\x06status\x18\x03 \x01(\x0b\x32\x14.gooseai.AsyncStatus\"7\n\x0b\x41syncHandle\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x14\n\x0c\x61sync_handle\x18\x02 \x01(\t\"\x13\n\x11\x41syncCancelAnswer*E\n\x0c\x46inishReason\x12\x08\n\x04NULL\x10\x00\x12\n\n\x06LENGTH\x10\x01\x12\x08\n\x04STOP\x10\x02\x12\t\n\x05\x45RROR\x10\x03\x12\n\n\x06\x46ILTER\x10\x04*\xc6\x02\n\x0c\x41rtifactType\x12\x11\n\rARTIFACT_NONE\x10\x00\x12\x12\n\x0e\x41RTIFACT_IMAGE\x10\x01\x12\x12\n\x0e\x41RTIFACT_VIDEO\x10\x02\x12\x11\n\rARTIFACT_TEXT\x10\x03\x12\x13\n\x0f\x41RTIFACT_TOKENS\x10\x04\x12\x16\n\x12\x41RTIFACT_EMBEDDING\x10\x05\x12\x1c\n\x18\x41RTIFACT_CLASSIFICATIONS\x10\x06\x12\x11\n\rARTIFACT_MASK\x10\x07\x12\x13\n\x0f\x41RTIFACT_LATENT\x10\x08\x12\x13\n\x0f\x41RTIFACT_TENSOR\x10\t\x12\x12\n\rARTIFACT_LORA\x10\xf4\x03\x12\x13\n\x0e\x41RTIFACT_DEPTH\x10\xf5\x03\x12\x1d\n\x18\x41RTIFACT_TOKEN_EMBEDDING\x10\xf6\x03\x12\x18\n\x13\x41RTIFACT_HINT_IMAGE\x10\xf7\x03*M\n\x11GaussianDirection\x12\x12\n\x0e\x44IRECTION_NONE\x10\x00\x12\x10\n\x0c\x44IRECTION_UP\x10\x01\x12\x12\n\x0e\x44IRECTION_DOWN\x10\x02*\x83\x01\n\rChannelSource\x12\r\n\tCHANNEL_R\x10\x00\x12\r\n\tCHANNEL_G\x10\x01\x12\r\n\tCHANNEL_B\x10\x02\x12\r\n\tCHANNEL_A\x10\x03\x12\x10\n\x0c\x43HANNEL_ZERO\x10\x04\x12\x0f\n\x0b\x43HANNEL_ONE\x10\x05\x12\x13\n\x0f\x43HANNEL_DISCARD\x10\x06*\x8a\x01\n\x0bRescaleMode\x12\x12\n\x0eRESCALE_STRICT\x10\x00\x12\x11\n\rRESCALE_COVER\x10\x02\x12\x18\n\x14RESCALE_CONTAIN_ZERO\x10\x03\x12\x1d\n\x19RESCALE_CONTAIN_REPLICATE\x10\x04\x12\x1b\n\x17RESCALE_CONTAIN_REFLECT\x10\x05*t\n\rArtifactStage\x12\x1f\n\x1b\x41RTIFACT_BEFORE_ADJUSTMENTS\x10\x00\x12\x1e\n\x1a\x41RTIFACT_AFTER_ADJUSTMENTS\x10\x01\x12\"\n\x1e\x41RTIFACT_AFTER_POSTADJUSTMENTS\x10\x02*g\n\x0eMaskedAreaInit\x12\x19\n\x15MASKED_AREA_INIT_ZERO\x10\x00\x12\x1b\n\x17MASKED_AREA_INIT_RANDOM\x10\x01\x12\x1d\n\x19MASKED_AREA_INIT_ORIGINAL\x10\x02*5\n\x0cWeightMethod\x12\x10\n\x0cTEXT_ENCODER\x10\x00\x12\x13\n\x0f\x43ROSS_ATTENTION\x10\x01*\x9b\x04\n\x10\x44iffusionSampler\x12\x10\n\x0cSAMPLER_DDIM\x10\x00\x12\x10\n\x0cSAMPLER_DDPM\x10\x01\x12\x13\n\x0fSAMPLER_K_EULER\x10\x02\x12\x1d\n\x19SAMPLER_K_EULER_ANCESTRAL\x10\x03\x12\x12\n\x0eSAMPLER_K_HEUN\x10\x04\x12\x13\n\x0fSAMPLER_K_DPM_2\x10\x05\x12\x1d\n\x19SAMPLER_K_DPM_2_ANCESTRAL\x10\x06\x12\x11\n\rSAMPLER_K_LMS\x10\x07\x12 \n\x1cSAMPLER_K_DPMPP_2S_ANCESTRAL\x10\x08\x12\x16\n\x12SAMPLER_K_DPMPP_2M\x10\t\x12\x17\n\x13SAMPLER_K_DPMPP_SDE\x10\n\x12\x1f\n\x1aSAMPLER_DPMSOLVERPP_1ORDER\x10\xf4\x03\x12\x1f\n\x1aSAMPLER_DPMSOLVERPP_2ORDER\x10\xf5\x03\x12\x1f\n\x1aSAMPLER_DPMSOLVERPP_3ORDER\x10\xf6\x03\x12\x15\n\x10SAMPLER_DPM_FAST\x10\xa6\x04\x12\x19\n\x14SAMPLER_DPM_ADAPTIVE\x10\xa7\x04\x12)\n SAMPLER_DPMSOLVERPP_2S_ANCESTRAL\x10\xa8\x04\x1a\x02\x08\x01\x12 \n\x17SAMPLER_DPMSOLVERPP_SDE\x10\xa9\x04\x1a\x02\x08\x01\x12\x1f\n\x16SAMPLER_DPMSOLVERPP_2M\x10\xaa\x04\x1a\x02\x08\x01*H\n\x10SamplerNoiseType\x12\x18\n\x14SAMPLER_NOISE_NORMAL\x10\x00\x12\x1a\n\x16SAMPLER_NOISE_BROWNIAN\x10\x01*F\n\x08Upscaler\x12\x10\n\x0cUPSCALER_RGB\x10\x00\x12\x13\n\x0fUPSCALER_GFPGAN\x10\x01\x12\x13\n\x0fUPSCALER_ESRGAN\x10\x02*\xd8\x01\n\x0eGuidancePreset\x12\x18\n\x14GUIDANCE_PRESET_NONE\x10\x00\x12\x1a\n\x16GUIDANCE_PRESET_SIMPLE\x10\x01\x12\x1d\n\x19GUIDANCE_PRESET_FAST_BLUE\x10\x02\x12\x1e\n\x1aGUIDANCE_PRESET_FAST_GREEN\x10\x03\x12\x18\n\x14GUIDANCE_PRESET_SLOW\x10\x04\x12\x1a\n\x16GUIDANCE_PRESET_SLOWER\x10\x05\x12\x1b\n\x17GUIDANCE_PRESET_SLOWEST\x10\x06*\x91\x01\n\x11ModelArchitecture\x12\x1b\n\x17MODEL_ARCHITECTURE_NONE\x10\x00\x12\x1f\n\x1bMODEL_ARCHITECTURE_CLIP_VIT\x10\x01\x12\"\n\x1eMODEL_ARCHITECTURE_CLIP_RESNET\x10\x02\x12\x1a\n\x16MODEL_ARCHITECTURE_LDM\x10\x03*\xa2\x01\n\x06\x41\x63tion\x12\x16\n\x12\x41\x43TION_PASSTHROUGH\x10\x00\x12\x1f\n\x1b\x41\x43TION_REGENERATE_DUPLICATE\x10\x01\x12\x15\n\x11\x41\x43TION_REGENERATE\x10\x02\x12\x1e\n\x1a\x41\x43TION_OBFUSCATE_DUPLICATE\x10\x03\x12\x14\n\x10\x41\x43TION_OBFUSCATE\x10\x04\x12\x12\n\x0e\x41\x43TION_DISCARD\x10\x05*D\n\x0e\x43lassifierMode\x12\x17\n\x13\x43LSFR_MODE_ZEROSHOT\x10\x00\x12\x19\n\x15\x43LSFR_MODE_MULTICLASS\x10\x01*=\n\x0b\x41ssetAction\x12\r\n\tASSET_PUT\x10\x00\x12\r\n\tASSET_GET\x10\x01\x12\x10\n\x0c\x41SSET_DELETE\x10\x02*\x81\x01\n\x08\x41ssetUse\x12\x17\n\x13\x41SSET_USE_UNDEFINED\x10\x00\x12\x13\n\x0f\x41SSET_USE_INPUT\x10\x01\x12\x14\n\x10\x41SSET_USE_OUTPUT\x10\x02\x12\x1a\n\x16\x41SSET_USE_INTERMEDIATE\x10\x03\x12\x15\n\x11\x41SSET_USE_PROJECT\x10\x04*W\n\x0bStageAction\x12\x15\n\x11STAGE_ACTION_PASS\x10\x00\x12\x18\n\x14STAGE_ACTION_DISCARD\x10\x01\x12\x17\n\x13STAGE_ACTION_RETURN\x10\x02\x32\xbe\x02\n\x11GenerationService\x12\x31\n\x08Generate\x12\x10.gooseai.Request\x1a\x0f.gooseai.Answer\"\x00\x30\x01\x12;\n\rChainGenerate\x12\x15.gooseai.ChainRequest\x1a\x0f.gooseai.Answer\"\x00\x30\x01\x12\x39\n\rAsyncGenerate\x12\x10.gooseai.Request\x1a\x14.gooseai.AsyncHandle\"\x00\x12;\n\x0b\x41syncResult\x12\x14.gooseai.AsyncHandle\x1a\x14.gooseai.AsyncAnswer\"\x00\x12\x41\n\x0b\x41syncCancel\x12\x14.gooseai.AsyncHandle\x1a\x1a.gooseai.AsyncCancelAnswer\"\x00\x42;Z9github.com/stability-ai/api-interfaces/gooseai/generationb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'generation_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z9github.com/stability-ai/api-interfaces/gooseai/generation'
  _DIFFUSIONSAMPLER.values_by_name["SAMPLER_DPMSOLVERPP_2S_ANCESTRAL"]._options = None
  _DIFFUSIONSAMPLER.values_by_name["SAMPLER_DPMSOLVERPP_2S_ANCESTRAL"]._serialized_options = b'\010\001'
  _DIFFUSIONSAMPLER.values_by_name["SAMPLER_DPMSOLVERPP_SDE"]._options = None
  _DIFFUSIONSAMPLER.values_by_name["SAMPLER_DPMSOLVERPP_SDE"]._serialized_options = b'\010\001'
  _DIFFUSIONSAMPLER.values_by_name["SAMPLER_DPMSOLVERPP_2M"]._options = None
  _DIFFUSIONSAMPLER.values_by_name["SAMPLER_DPMSOLVERPP_2M"]._serialized_options = b'\010\001'
  _LORA.fields_by_name['weights']._options = None
  _LORA.fields_by_name['weights']._serialized_options = b'\030\001'
  _ARTIFACT.fields_by_name['lora']._options = None
  _ARTIFACT.fields_by_name['lora']._serialized_options = b'\030\001'
  _ARTIFACT.fields_by_name['token_embedding']._options = None
  _ARTIFACT.fields_by_name['token_embedding']._serialized_options = b'\030\001'
  _FINISHREASON._serialized_start=8449
  _FINISHREASON._serialized_end=8518
  _ARTIFACTTYPE._serialized_start=8521
  _ARTIFACTTYPE._serialized_end=8847
  _GAUSSIANDIRECTION._serialized_start=8849
  _GAUSSIANDIRECTION._serialized_end=8926
  _CHANNELSOURCE._serialized_start=8929
  _CHANNELSOURCE._serialized_end=9060
  _RESCALEMODE._serialized_start=9063
  _RESCALEMODE._serialized_end=9201
  _ARTIFACTSTAGE._serialized_start=9203
  _ARTIFACTSTAGE._serialized_end=9319
  _MASKEDAREAINIT._serialized_start=9321
  _MASKEDAREAINIT._serialized_end=9424
  _WEIGHTMETHOD._serialized_start=9426
  _WEIGHTMETHOD._serialized_end=9479
  _DIFFUSIONSAMPLER._serialized_start=9482
  _DIFFUSIONSAMPLER._serialized_end=10021
  _SAMPLERNOISETYPE._serialized_start=10023
  _SAMPLERNOISETYPE._serialized_end=10095
  _UPSCALER._serialized_start=10097
  _UPSCALER._serialized_end=10167
  _GUIDANCEPRESET._serialized_start=10170
  _GUIDANCEPRESET._serialized_end=10386
  _MODELARCHITECTURE._serialized_start=10389
  _MODELARCHITECTURE._serialized_end=10534
  _ACTION._serialized_start=10537
  _ACTION._serialized_end=10699
  _CLASSIFIERMODE._serialized_start=10701
  _CLASSIFIERMODE._serialized_end=10769
  _ASSETACTION._serialized_start=10771
  _ASSETACTION._serialized_end=10832
  _ASSETUSE._serialized_start=10835
  _ASSETUSE._serialized_end=10964
  _STAGEACTION._serialized_start=10966
  _STAGEACTION._serialized_end=11053
  _TOKEN._serialized_start=44
  _TOKEN._serialized_end=91
  _TOKENS._serialized_start=93
  _TOKENS._serialized_end=177
  _IMAGEADJUSTMENT_GAUSSIAN._serialized_start=179
  _IMAGEADJUSTMENT_GAUSSIAN._serialized_end=267
  _IMAGEADJUSTMENT_INVERT._serialized_start=269
  _IMAGEADJUSTMENT_INVERT._serialized_end=293
  _IMAGEADJUSTMENT_LEVELS._serialized_start=295
  _IMAGEADJUSTMENT_LEVELS._serialized_end=399
  _IMAGEADJUSTMENT_CHANNELS._serialized_start=402
  _IMAGEADJUSTMENT_CHANNELS._serialized_end=612
  _IMAGEADJUSTMENT_RESCALE._serialized_start=614
  _IMAGEADJUSTMENT_RESCALE._serialized_end=730
  _IMAGEADJUSTMENT_CROP._serialized_start=732
  _IMAGEADJUSTMENT_CROP._serialized_end=812
  _IMAGEADJUSTMENT_DEPTH._serialized_start=814
  _IMAGEADJUSTMENT_DEPTH._serialized_end=864
  _IMAGEADJUSTMENT_CANNYEDGE._serialized_start=866
  _IMAGEADJUSTMENT_CANNYEDGE._serialized_end=940
  _IMAGEADJUSTMENT_EDGEDETECTION._serialized_start=942
  _IMAGEADJUSTMENT_EDGEDETECTION._serialized_end=973
  _IMAGEADJUSTMENT_SEGMENTATION._serialized_start=975
  _IMAGEADJUSTMENT_SEGMENTATION._serialized_end=1005
  _IMAGEADJUSTMENT._serialized_start=1008
  _IMAGEADJUSTMENT._serialized_end=1583
  _SAFETENSORSMETA._serialized_start=1585
  _SAFETENSORSMETA._serialized_end=1630
  _SAFETENSORSTENSOR._serialized_start=1632
  _SAFETENSORSTENSOR._serialized_end=1697
  _SAFETENSORS._serialized_start=1699
  _SAFETENSORS._serialized_end=1801
  _LORAWEIGHT._serialized_start=1803
  _LORAWEIGHT._serialized_end=1851
  _LORA._serialized_start=1853
  _LORA._serialized_end=1937
  _ARTIFACTREFERENCE._serialized_start=1939
  _ARTIFACTREFERENCE._serialized_end=2040
  _TOKENEMBEDDING._serialized_start=2042
  _TOKENEMBEDDING._serialized_end=2105
  _CACHECONTROL._serialized_start=2107
  _CACHECONTROL._serialized_end=2195
  _ARTIFACT._serialized_start=2198
  _ARTIFACT._serialized_end=3010
  _NAMEDWEIGHT._serialized_start=3012
  _NAMEDWEIGHT._serialized_end=3055
  _TOKENOVERRIDE._serialized_start=3057
  _TOKENOVERRIDE._serialized_end=3135
  _PROMPTPARAMETERS._serialized_start=3138
  _PROMPTPARAMETERS._serialized_end=3312
  _PROMPT._serialized_start=3315
  _PROMPT._serialized_end=3518
  _SIGMAPARAMETERS._serialized_start=3521
  _SIGMAPARAMETERS._serialized_end=3654
  _CHURNSETTINGS._serialized_start=3656
  _CHURNSETTINGS._serialized_end=3766
  _SAMPLERPARAMETERS._serialized_start=3769
  _SAMPLERPARAMETERS._serialized_end=4292
  _CONDITIONERPARAMETERS._serialized_start=4295
  _CONDITIONERPARAMETERS._serialized_end=4434
  _SCHEDULEPARAMETERS._serialized_start=4436
  _SCHEDULEPARAMETERS._serialized_end=4542
  _STEPPARAMETER._serialized_start=4545
  _STEPPARAMETER._serialized_end=4773
  _MODEL._serialized_start=4776
  _MODEL._serialized_end=4927
  _CUTOUTPARAMETERS._serialized_start=4930
  _CUTOUTPARAMETERS._serialized_end=5118
  _GUIDANCESCHEDULEPARAMETERS._serialized_start=5120
  _GUIDANCESCHEDULEPARAMETERS._serialized_end=5181
  _GUIDANCEINSTANCEPARAMETERS._serialized_start=5184
  _GUIDANCEINSTANCEPARAMETERS._serialized_end=5463
  _GUIDANCEPARAMETERS._serialized_start=5465
  _GUIDANCEPARAMETERS._serialized_end=5591
  _TRANSFORMTYPE._serialized_start=5593
  _TRANSFORMTYPE._serialized_end=5703
  _EXTENDEDPARAMETER._serialized_start=5705
  _EXTENDEDPARAMETER._serialized_end=5794
  _EXTENDEDPARAMETERS._serialized_start=5796
  _EXTENDEDPARAMETERS._serialized_end=5864
  _HIRESFIXPARAMETERS._serialized_start=5866
  _HIRESFIXPARAMETERS._serialized_end=5946
  _IMAGEPARAMETERS._serialized_start=5949
  _IMAGEPARAMETERS._serialized_end=6629
  _CLASSIFIERCONCEPT._serialized_start=6631
  _CLASSIFIERCONCEPT._serialized_end=6705
  _CLASSIFIERCATEGORY._serialized_start=6708
  _CLASSIFIERCATEGORY._serialized_end=6952
  _CLASSIFIERPARAMETERS._serialized_start=6955
  _CLASSIFIERPARAMETERS._serialized_end=7139
  _ASSETPARAMETERS._serialized_start=7141
  _ASSETPARAMETERS._serialized_end=7248
  _ANSWERMETA._serialized_start=7251
  _ANSWERMETA._serialized_end=7399
  _ANSWER._serialized_start=7402
  _ANSWER._serialized_end=7571
  _REQUEST._serialized_start=7574
  _REQUEST._serialized_end=7937
  _ONSTATUS._serialized_start=7939
  _ONSTATUS._serialized_end=8058
  _STAGE._serialized_start=8060
  _STAGE._serialized_end=8152
  _CHAINREQUEST._serialized_start=8154
  _CHAINREQUEST._serialized_end=8219
  _ASYNCSTATUS._serialized_start=8221
  _ASYNCSTATUS._serialized_end=8265
  _ASYNCANSWER._serialized_start=8267
  _ASYNCANSWER._serialized_end=8369
  _ASYNCHANDLE._serialized_start=8371
  _ASYNCHANDLE._serialized_end=8426
  _ASYNCCANCELANSWER._serialized_start=8428
  _ASYNCCANCELANSWER._serialized_end=8447
  _GENERATIONSERVICE._serialized_start=11056
  _GENERATIONSERVICE._serialized_end=11374
# @@protoc_insertion_point(module_scope)
