"""The halves of a split MiniMax-H3 deployment, for both of its checkpoint partitions.

MiniMax-H3 is 195.9 GiB in bfloat16 and a single unquantized deployment is huge, so MiniMaxH3Blocks is cut
at its text_encoder step: the Qwen3-VL conditioner runs in the conditioner Space, everything else in a generator
Space, and prompt_embeds + text_token_tags is the whole wire format between them.
"""

from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3Ref2VAReferenceEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
    MiniMaxH3TextEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3AutoKeyframeVaeEncoderStep,
    MiniMaxH3AutoResizeStep,
    MiniMaxH3CoreDenoiseStep,
    MiniMaxH3DecodeStep,
    MiniMaxH3Ref2VACoreDenoiseStep,
    _generation_outputs,
)
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam


def _wire_outputs(num_frames: bool = True) -> list[OutputParam]:
    """The wire format of the split."""
    return [
        OutputParam.template("prompt_embeds"),
        OutputParam("text_token_tags", description="The per-row modality tag of every row of `prompt_embeds`."),
        OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
        OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
        *(
            [OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5.")]
            if num_frames
            else []
        ),
    ]


class MiniMaxH3ConditionerBlocks(SequentialPipelineBlocks):
    """The conditioner half of a split MiniMax-H3."""

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3AutoResizeStep, MiniMaxH3TextEncoderStep]
    block_names = ["resize", "text_encoder"]

    @property
    def description(self):
        return (
            "The conditioner half of a split MiniMax-H3 deployment: puts the keyframes onto the target canvas and "
            "encodes MiniMax-H3's presentation of the request into the prompt_embeds / text_token_tags pair the "
            "denoising half consumes."
        )

    @property
    def outputs(self):
        return _wire_outputs(num_frames=False)


class MiniMaxH3GeneratorBlocks(SequentialPipelineBlocks):
    """The denoising half of a split MiniMax-H3: MiniMaxH3Blocks with its text_encoder step removed."""

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3AutoResizeStep,
        MiniMaxH3AutoKeyframeVaeEncoderStep,
        MiniMaxH3CoreDenoiseStep,
        MiniMaxH3AfterDenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = ["resize", "vae_encoder", "denoise", "after_denoise", "decode"]

    @property
    def description(self):
        return (
            "The denoising half of a split MiniMax-H3 deployment: the t2va / fl2va branch of MiniMaxH3Blocks "
            "without its text-encoder step, so prompt_embeds and text_token_tags come in as inputs."
        )

    @property
    def outputs(self):
        return _generation_outputs()


class MiniMaxH3Ref2VAConditionerBlocks(SequentialPipelineBlocks):
    """The conditioner half of a split ref2va."""

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VASetupStep, MiniMaxH3Ref2VATextEncoderStep]
    block_names = ["setup", "text_encoder"]

    @property
    def description(self):
        return "The conditioner half of a split MiniMax-H3 ref2va deployment."

    @property
    def outputs(self):
        return _wire_outputs()


class MiniMaxH3Ref2VAGeneratorBlocks(SequentialPipelineBlocks):
    """The denoising half of a split ref2va."""

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3Ref2VASetupStep,
        MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3Ref2VACoreDenoiseStep,
        MiniMaxH3AfterDenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = ["setup", "reference_encoder", "denoise", "after_denoise", "decode"]

    @property
    def description(self):
        return "The denoising half of a split MiniMax-H3 ref2va deployment."

    @property
    def outputs(self):
        return _generation_outputs()
