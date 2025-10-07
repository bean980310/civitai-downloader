from enum import Enum

class NsfwLevel(Enum):
    NONE="None"
    SOFT="Soft"
    MATURE="Mature"
    X="X"

# Enum Classes
class BaseModel(Enum):
    SD14="SD 1.4"
    SD15="SD 1.5"
    SD15LCM="SD 1.5 LCM"
    SD15HYPER="SD 1.5 Hyper"
    SD20="SD 2.0"
    SD21="SD 2.1"
    SDXL="SDXL 1.0"
    SD3="SD 3"
    SD35="SD 3.5"
    PONY="Pony"
    FLUXS="Flux .1 S"
    FLUXD="Flux .1 D"
    SDXLLCM="SDXL 1.0 LCM"
    SDXLTURBO= "SDXL Turbo"
    SDXLLIGHTNING= "SDXL Lightning"
    STABLE_CASCADE = "Stable Cascade"
    SVD = "SVD"
    SVD_XT = "SVD XT"
    PLAYGROUND_V2 = "Playground V2"
    PIXART_A = "PixArt A"
    PIXART_Σ = "PixArt Σ"
    HUNYUAN_1 = "Hunyuan 1"
    LUMINA = "Lumina"
    KOLORS="Kolors"
    ILLUSTRIOUS="Illustrious"
    OTHER="Other"

class ModelType(Enum):
    CHECKPOINT="Checkpoint"
    TEXTUAL_INVERSION="TextualInversion"
    HYPERNETWORK="Hypernetwork"
    AESTHETIC_GRADIENT="AestheticGradient"
    LORA="LORA"
    LOCON="LoCon"
    DORA="DoRA"
    CONTROLNET="Controlnet"
    UPSCALER="Upscaler"
    MOTIONMODULE="MotionModule"
    VAE="VAE"
    POSES="Poses"

class ModelMode(Enum):
    ARCHIVED = "Archived"
    TAKEN_DOWN = "TakenDown"

class ModelFp(Enum):
    FP16="fp16"
    FP32="fp32"
    BF16="bf16"
    FP8="fp8"
    NF4="nf4"

class ModelSize(Enum):
    FULL="full"
    PRUNED="pruned"

class ModelFormat(Enum):
    SAFETENSOR="SafeTensor"
    PICKLETENSOR="PickleTensor"
    DIFFUSERS="Diffusers"
    ONNX="ONNX"
    COREML="CoreML"
    GGUF="GGUF"
    OTHER="Other"

class AllowCommercialUse(Enum):
    NONE="None"
    IMAGE="Image"
    RENT="Rent"
    SELL="Sell"

class Sort(Enum):
    HIGHEST_RATED="Highest Rated"
    MOST_DOWNLOADED="Most Downloaded"
    NEWEST="Newest"

class Period(Enum):
    ALLTIME="AllTime"
    YEAR="Year"
    MONTH="Month"
    WEEK="Week"
    DAY="Day"
