#version 450

#extension GL_ARB_separate_shader_objects : enable

struct CameraUniforms {
	vec3  origin;
	float focal_distance;
	vec3 u;
	float vfov;
	vec3  v;
	float dof_scale;
	vec3  w;
};

layout (binding = 0, rgba32f) uniform readonly image2D input_radiance;
layout (binding = 1, rgba32f) uniform image2D output_radiance;
layout (binding = 2) uniform Uniforms {
	CameraUniforms camera;
	uint frame_num;
	uint width;
	uint height;
} uniforms;

layout (location = 0) out vec4 out_colour;

void main() {
	ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
    vec3 old_sample = imageLoad(input_radiance, pixelCoords).rgb;
	out_colour = vec4(old_sample, 1.0);

	vec4 pixel = vec4(0.01 * uniforms.frame_num, 0.3, 0.5, 1.0);
	imageStore(output_radiance, pixelCoords, pixel);
}
