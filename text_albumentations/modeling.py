from text_albumentations.runtime import (
    DEFAULT_MODEL_NAME,
    build_mlx_outlines_model,
    create_outlines_runtime,
    get_default_outlines_runtime,
)


def generate_structured_output(messages, output_type, temp=0.2, max_tokens=5000):
    runtime = get_default_outlines_runtime()
    return runtime.generate_structured(
        messages,
        output_type,
        temperature=temp,
        max_tokens=max_tokens,
    )


def generate_variations(out, output_type, context=None, temp=0.5, max_tokens=5000):
    runtime = get_default_outlines_runtime()
    return runtime.generate_variation(
        out,
        output_type,
        context=context,
        temperature=temp,
        max_tokens=max_tokens,
    )


def augment_data(out, output_type, context=None, temp=0.5, max_tokens=5000):
    return generate_variations(
        out,
        output_type,
        context=context,
        temp=temp,
        max_tokens=max_tokens,
    )


model_name = DEFAULT_MODEL_NAME
model = build_mlx_outlines_model(model_name)


def create_model_runtime(model):
    return create_outlines_runtime(model)
