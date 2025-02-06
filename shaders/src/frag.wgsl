struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frag_colour: vec3f,
    @location(1) tex_coord: vec2f,
}

@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var tex_texture: texture_2d<f32>;

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    // return vec4<f32>(in.tex_coord, 0.0, 1.0);
    // return textureSample(tex_texture, tex_sampler, in.tex_coord);
    return vec4<f32>(in.frag_colour * textureSample(tex_texture, tex_sampler, in.tex_coord).xyz, 1.0);
}