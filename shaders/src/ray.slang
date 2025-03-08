__import rng;

struct CameraUniforms {
	float3 origin;
	float focal_distance;
	float3 u;
	float vfov;
	float3 v;
	float dof_scale;
	float3 w;
}

struct Uniforms {
	CameraUniforms camera;
	uint frame_num;
	uint width;
	uint height;
};

struct Material {
	float3 albedo;
	float alpha;
	float refraction_index;
	float smoothness;
	float emissivity;
	float emission_strength;
	float3 emission_colour;
}

struct Sphere {
	float3 center;
	float radius;
	uint material;
	uint is_selected;
}

struct Quad {
	float3 q;
	uint is_selected;
	float3 u;
	float3 v;
	float3 normal;
	float d;
	float3 w;
	uint material;
}

struct Triangle {
	float3 a;
	uint is_selected;
	float3 b;
	uint material;
	float3 c;
}

struct AABB {
	float3 min;
	uint left_child_index;
	float3 max;
	uint right_child_index;
	uint object_type;
	uint object_index;
	uint is_populated;
}

// StructuredBuffer<float> buffer0;
// StructuredBuffer<float> buffer1;
// RWStructuredBuffer<float> result;
[[vk::binding(0)]]
Texture2D<float4> input_radiance;
[[vk::binding(1)]]
RWTexture2D<float4> output_radiance;
[[vk::binding(2)]]
Uniforms uniforms;
[[vk::binding(3)]]
Material[10] materials;
[[vk::binding(4)]]
StructuredBuffer<AABB> bvh;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint3 id: SV_DispatchThreadID) {
	uint status = 0;
	input_radiance.Load(int3(id.xy, 0), int2(0), status);
	// vec4 pixel = vec4(0.01 * 50, 0.3, 0.5, 1.0);
	float4 pixel = float4(1.0, 0.5, 0.5, 1.0);
	// vec4 pixel = vec4(0.01 * uniforms.frame_num, 0.3, 0.5, 1.0);
	output_radiance[id.xy] = pixel;
	// imageStore(output_radiance, pixelCoords, pixel);
}
