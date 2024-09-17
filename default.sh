#!/bin/bash

# This file will be sourced in init.sh

# https://raw.githubusercontent.com/ai-dock/comfyui/main/config/provisioning/default.sh

# Packages are installed after nodes so we can fix them...

PYTHON_PACKAGES=(
    #"opencv-python==4.7.0.72"
)

NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/jags111/efficiency-nodes-comfyui"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/SLAPaper/ComfyUI-Image-Selector"
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes"
    "https://github.com/FizzleDorf/ComfyUI_FizzNodes"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/BlenderNeko/ComfyUI_Noise"
    "https://github.com/WASasquatch/was-node-suite-comfyui"
    "https://github.com/crystian/ComfyUI-Crystools"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM"
    "https://github.com/daxcay/ComfyUI-JDCN"
    "https://github.com/comfyanonymous/ComfyUI_experiments"
    "https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes"
    "https://github.com/sipherxyz/comfyui-art-venture"
    "https://github.com/M1kep/ComfyLiterals"
    "https://github.com/kijai/ComfyUI-segment-anything-2"
    "https://github.com/MrForExample/ComfyUI-3D-Pack"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO"
    "https://github.com/filliptm/ComfyUI_Fill-Nodes"
    "https://github.com/M1kep/Comfy_KepListStuff"
    "https://github.com/shadowcz007/comfyui-mixlab-nodes"
    # Product Photo
    "https://github.com/chflame163/ComfyUI_LayerStyle"
    "https://github.com/kijai/ComfyUI-IC-Light"
    "https://github.com/huchenlei/ComfyUI-IC-Light-Native"
    "https://github.com/spacepxl/ComfyUI-Image-Filters"
    "https://github.com/stavsap/comfyui-ollama"
    "https://github.com/jiaxiangc/ComfyUI-ResAdapter"
    ""
)

CHECKPOINT_MODELS=(
    "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper8_LCM.safetensors"
    "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors"
    "https://huggingface.co/Justin-Choo/epiCRealism-Natural_Sin_RC1_VAE/resolve/main/epicrealism_naturalSinRC1VAE.safetensors"
    "https://civitai.com/api/download/models/537505" # CyberRealistic
    # "https://civitai.com/api/download/models/413877" # CyberRealistic LCM
    "https://civitai.com/api/download/models/274039" # Juggernaut
    # "https://civitai.com/api/download/models/588174" UmamiLCM
    # "https://civitai.com/api/download/models/256668" # absolute reality LCM
    # "https://huggingface.co/Lykon/AbsoluteReality/resolve/main/AbsoluteReality_1.8.1_pruned.safetensors"
    # "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    # "https://civitai.com/api/download/models/344398" # photonLCM
    # "https://huggingface.co/churaed/mosaic/resolve/main/blokadachd_15.ckpt"
    # "https://huggingface.co/churaed/mosaic/resolve/main/mosaicsclptr.ckpt"
    # "https://huggingface.co/fluently/Fluently-v4-LCM/resolve/main/Fluently-v4-LCM.safetensors"

)

LORA_MODELS=(
    "https://civitai.com/api/download/models/87153" # more details
    "https://civitai.com/api/download/models/118644" # HD Helper
    # "https://civitai.com/api/download/models/451956" # HXZsculpture-1
    # "https://civitai.com/api/download/models/16576" # epi_noiseoffset2 
    # "https://huggingface.co/churaed/mosaic/resolve/main/mosaic_madness.safetensors"
    # "https://civitai.com/api/download/models/288694" # statue shigao
    # "https://civitai.com/api/download/models/188126" # Clay Sculpt - Style
    # "https://civitai.com/api/download/models/339716" # Colorful Glass Sculpture Artwork
    # "https://civitai.com/api/download/models/250533" # Sculpture

)

FLUX_MODELS=(
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
)

UNET_MODELS=(
    "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf"
    # "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_0.gguf"
    # "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q8_0.gguf"
    # "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_1.gguf"
)

XLAB_LORA=(
    # "https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/anime_lora_comfy_converted.safetensors"
    # "https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/art_lora_comfy_converted.safetensors"
    # "https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/disney_lora_comfy_converted.safetensors"
    # "https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/mjv6_lora_comfy_converted.safetensors"
    # "https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/realism_lora_comfy_converted.safetensors"
    # "https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/scenery_lora_comfy_converted.safetensors"
)

XLAB_CONTROLNET=(
    # "https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-canny-controlnet-v3.safetensors"
    # "https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-depth-controlnet-v3.safetensors"
    # "https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-hed-controlnet-v3.safetensors"
)

XLAB_IPADAPTER=(
    # "https://huggingface.co/XLabs-AI/flux-ip-adapter/resolve/main/flux-ip-adapter.safetensors"
)

FLUX_LORA=(
    "https://huggingface.co/churaed/dmovie-lora/resolve/main/v01/dmovie-v01_rank16_bf16.safetensors"
    # "https://huggingface.co/churaed/mosaic/resolve/main/cmemory3_rank16_bf16.safetensors"
    # "https://huggingface.co/churaed/mosaic/resolve/main/cmemory3_rank16_bf16-step02250.safetensors"
    # "https://huggingface.co/churaed/mosaic/resolve/main/cmemory3_rank16_bf16-step01500.safetensors"
    # "https://huggingface.co/churaed/mosaic/resolve/main/cmemory3_rank16_bf16-step00750.safetensors"
)

CLIP_MODELS=(
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
    "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q4_K_S.gguf"
    "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q8_0.gguf"
)

SDXL_MODELS=(
    "https://huggingface.co/fluently/Fluently-XL-Final/resolve/main/FluentlyXL-Final.safetensors"
    "https://huggingface.co/fluently/Fluently-XL-v3-inpainting/resolve/main/FluentlyXL-v3-inpainting.safetensors"
    # "https://huggingface.co/sd-community/sdxl-flash/resolve/main/SDXL-Flash.safetensors"
    # "https://civitai.com/api/download/models/714404" # MooMooE-comerceSDXL Product Photography
    # "https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/resolve/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors"
    # "https://huggingface.co/erohinem/SDXL/resolve/bb3b7fa6598742f81f3eae359fe39165ba29e6dd/juggernautXL_v9Rdphoto2Lightning.safetensors"
)

SDXL_LORA_MODELS=(
    # "https://huggingface.co/taki0112/lora-trained-xl_mosaic-art_split/resolve/main/pytorch_lora_weights.safetensors"
    # "https://huggingface.co/CiroN2022/mosaic-style/resolve/main/mosaic.safetensors"
    # "https://civitai.com/api/download/models/444936?type=Model&format=SafeTensor" # Colorful Mosaic SDXL v2
    # "https://civitai.com/api/download/models/390257?type=Model&format=SafeTensor" # ArtfullyMOSAIC SDXL V1
    # "https://civitai.com/api/download/models/206134?type=Model&format=SafeTensor" # Socrealistic Mosaic Style XL
    # "https://civitai.com/api/download/models/288512?type=Model&format=SafeTensor" # Mosaic Texture SDXL
    "https://civitai.com/api/download/models/703047" # Product Photography
    "https://civitai.com/api/download/models/249521" # Texta
)


ANIMATEDIFF_MODELS=(
    # "https://civitai.com/api/download/models/366178"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt"
    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"
    "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt"
    "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt"
    # "https://huggingface.co/wangfuyun/AnimateLCM-I2V/resolve/main/AnimateLCM_sd15_i2v.ckpt"
    "https://huggingface.co/moonshotmillion/AnimateDiff_LCM_Motion_Model_v1/resolve/main/animatediffLCMMotion_v10.ckpt"
)
ANIMATEDIFF_MOTION_LORA=(
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltDown.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltDown.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltUp.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltUp.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.ckpt"
)
ANIMATEDIFF_LORAS=(
    "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt"
    "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors"
    "https://huggingface.co/wangfuyun/AnimateLCM-I2V/resolve/main/AnimateLCM_sd15_i2v_lora.safetensors"
)

VAE_MODELS=(
    # "https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors" # FLUX vae
    "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
)

UPSCALE_MODELS=(
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth"
    "https://civitai.com/api/download/models/164891" # Ultramix
    "https://civitai.com/api/download/models/357054" # 4x_RealisticRescaler_100000_G
    "https://civitai.com/api/download/models/164898" # RealESRGAN_x4Plus
    # "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
    # "https://huggingface.co/Akumetsu971/SD_Anime_Futuristic_Armor/resolve/main/4x_NMKD-Siax_200k.pth"
)

CONTROLNET_MODELS=(
    # "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth"
    # "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth"
    # "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth"
    # "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth"
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth"
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth"
    # "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth"
    # "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth"
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth"
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth"
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth"
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_rgb.ckpt"
    # "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_scribble.ckpt"
    "https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.safetensors"
)

CONTROLNET_SDXL_MODELS=(
    "https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic/resolve/main/TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors"
)

SNAPSHOTS=(
    "https://raw.githubusercontent.com/churaed/vast-comfy/main/snapshot_0.1.json"
)

CLIP_VISION=(
    # "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" # CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
)

IPADAPTER=(
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors"
    # "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"
    # "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin"
    # "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors"
    # "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors"
    
)

YOLO=(
    "https://huggingface.co/camenduru/YoloWorld-EfficientSAM/resolve/main/efficient_sam_s_gpu.jit"
    "https://huggingface.co/camenduru/YoloWorld-EfficientSAM/resolve/main/efficient_sam_s_cpu.jit"
)

BIREFNET=(
    "https://huggingface.co/ViperYX/BiRefNet/resolve/main/BiRefNet-DIS_ep580.pth"
    "https://huggingface.co/ViperYX/BiRefNet/resolve/main/swin_large_patch4_window12_384_22kto1k.pth"
)
BIREFNET_PTH=(
    "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth"
)

EMBEDDINGS=(
    "https://civitai.com/api/download/models/9208"
    "https://civitai.com/api/download/models/77169"
    "https://civitai.com/api/download/models/94057"
    "https://civitai.com/api/download/models/82745"
    "https://huggingface.co/Lykon/DreamShaper/resolve/main/FastNegativeEmbedding.pt"
    "https://huggingface.co/Lykon/DreamShaper/resolve/main/FastNegativeEmbeddingStrong.pt"
    "https://huggingface.co/Lykon/DreamShaper/resolve/main/UnrealisticDream.pt"
)

OLLAMA_MODELS=(
    "llava:13b"
    "llava-llama3"
    "brxce/stable-diffusion-prompt-generator"
)

SAM_MODELS=(
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
)

IC_LIGHT_MODELS=(
    "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors"
    "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    DISK_GB_AVAILABLE=$(($(df --output=avail -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_USED=$(($(df --output=used -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_ALLOCATED=$(($DISK_GB_AVAILABLE + $DISK_GB_USED))
    provisioning_print_header
    provisioning_update_comfyui
    provisioning_get_nodes
    provisioning_install_python_packages
    provisioning_install_ollama
    provisioning_get_ollama_models

    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/embeddings" \
        "${EMBEDDINGS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/vae" \
        "${VAE_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/upscale_models" \
        "${UPSCALE_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/clip_vision" \
        "${CLIP_VISION[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/ipadapter" \
        "${IPADAPTER[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/snapshots" \
        "${SNAPSHOTS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/custom_nodes/ComfyUI-YoloWorld-EfficientSAM" \
        "${YOLO[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/BiRefNet" \
        "${BIREFNET[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/BiRefNet/pth" \
        "${BIREFNET_PTH[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/sams" \
        "${SAM_MODELS[@]}"

    if [[ ${DOWNLOAD_SD15,,} == "true" ]]; then
        printf "Downloading SD1.5 models...\n"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/checkpoints" \
            "${CHECKPOINT_MODELS[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/controlnet" \
            "${CONTROLNET_MODELS[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/clip" \
            "${CLIP_MODELS[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/loras" \
            "${LORA_MODELS[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/unet" \
            "${IC_LIGHT_MODELS[@]}"
    
    if [[ ${DOWNLOAD_AD,,} == "true" ]]; then
        printf "Downloading AnimateDiff models...\n"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/animatediff_models" \
            "${ANIMATEDIFF_MODELS[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/animatediff_motion_lora" \
            "${ANIMATEDIFF_MOTION_LORA[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/loras/AD" \
            "${ANIMATEDIFF_LORAS[@]}"
    
    if [[ ${DOWNLOAD_SDXL,,} == "true" ]]; then
        printf "Downloading SDXL models...\n"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/checkpoints/SDXL" \
            "${SDXL_MODELS[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/controlnet/SDXL" \
            "${CONTROLNET_SDXL_MODELS[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/loras/SDXL" \
            "${SDXL_LORA_MODELS[@]}"
    
    if [[ ${DOWNLOAD_FLUX,,} == "true" ]]; then
        printf "Downloading FLUX models...\n"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/checkpoints/FLUX1" \
            "${FLUX_MODELS[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/loras/FLUX1" \
            "${FLUX_LORA[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/unet" \
            "${UNET_MODELS[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/xlabs/ipadapters" \
            "${XLAB_IPADAPTER[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/xlabs/controlnets" \
            "${XLAB_CONTROLNET[@]}"
        provisioning_get_models \
            "${WORKSPACE}/ComfyUI/models/xlabs/loras" \
            "${XLAB_LORA[@]}"
    provisioning_print_end
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="/opt/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                    micromamba -n comfyui run ${PIP_INSTALL} -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                micromamba -n comfyui run ${PIP_INSTALL} -r "${requirements}"
            fi
        fi
    done
}

function provisioning_install_python_packages() {
    if [ ${#PYTHON_PACKAGES[@]} -gt 0 ]; then
        micromamba -n comfyui run ${PIP_INSTALL} ${PYTHON_PACKAGES[*]}
    fi
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    dir="$1"
    mkdir -p "$dir"
    shift
    if [[ $DISK_GB_ALLOCATED -ge $DISK_GB_REQUIRED ]]; then
        arr=("$@")
    else
        printf "WARNING: Low disk space allocation - Only the first model will be downloaded!\n"
        arr=("$1")
    fi
    
    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
    if [[ $DISK_GB_ALLOCATED -lt $DISK_GB_REQUIRED ]]; then
        printf "WARNING: Your allocated disk size (%sGB) is below the recommended %sGB - Some models will not be downloaded\n" "$DISK_GB_ALLOCATED" "$DISK_GB_REQUIRED"
    fi
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Web UI will start now\n\n"
}

# Download from $1 URL to $2 file path
function provisioning_download() {
    wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
}

# Update ComfyUI
function provisioning_update_comfyui() {
    printf "Updating ComfyUI...\n"
    if [[ -d "${WORKSPACE}/ComfyUI" ]]; then
        ( cd "${WORKSPACE}/ComfyUI" && git pull )
    else
        git clone https://github.com/comfyanonymous/ComfyUI.git "${WORKSPACE}/ComfyUI"
    fi
    if [[ -e "${WORKSPACE}/ComfyUI/requirements.txt" ]]; then
        micromamba -n comfyui run ${PIP_INSTALL} -r "${WORKSPACE}/ComfyUI/requirements.txt"
    fi
}

function provisioning_install_ollama() {
    curl -fsSL https://ollama.com/install.sh | sh
}

function provisioning_get_ollama_models() {
    if [[ -z $1 ]]; then return 1; fi
    
    printf "Downloading Ollama models...\n"
    for model in "$@"; do
        printf "Downloading Ollama model: %s\n" "${model}"
        ollama pull "${model}"
        printf "\n"
    done
}

provisioning_start