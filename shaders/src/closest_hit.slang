__import rng;

struct Attributes {
	float2 bary;
};

struct Payload {
	[[vk::location(0)]]
	float3 radiance;
	float3 attenuation;
	int done;
	float3 origin;
	float3 direction;
};

struct Material {
	float3 albedo;
	float alpha;
	float refraction_index;
	float smoothness;
	float emissivity;
	float emission_strength;
	float3 emission_colour;
	uint padding;
}

struct Vertex {
	float3 pos;
	uint pad1;
	float3 normal;
	uint pad2;
}

struct Buffers {
	uint64_t vertices;
	uint64_t indices;
	uint64_t material_indices;
	uint64_t materials;
}

struct CameraUniforms {
	float3 origin;
	float focal_distance;
	float3 u;
	float vfov;
	float3 v;
	float dof_scale;
	float3 w;
}

struct UBO {
	CameraUniforms camera;
	uint frame_num;
	uint ray_frame_num;
	uint width;
	uint height;
}

[[vk::binding(0)]]
RaytracingAccelerationStructure tlas;
[[vk::binding(3)]]
ConstantBuffer<UBO> ubo;
[[vk::binding(4)]]
StructuredBuffer<Buffers> scene;

const static float EPSILON = 1e-2;

bool is_reflective_schlick(float cosine, float refraction_index) {
	float r0 = (1. - refraction_index) / (1. + refraction_index);
	r0 = r0 * r0;
	return (r0 + (1. - r0) * pow((1. - cosine), 5.)) > rand_f32();
}

void reflect_ray(inout Payload ray, float3 normal, float3 pos, Material material) {
	let lambertian_reflection = normal + generate_random_unit_vector();
	let metallic_reflection = reflect(WorldRayDirection(), normal);
	ray.direction = lerp(lambertian_reflection, metallic_reflection, material.smoothness);
	// Bump the start of the reflected ray a little bit off the surface to
	// try to minimize self intersections due to floating point errors
	ray.origin = pos + normal * EPSILON;
	ray.attenuation = material.albedo;
}

void dielectric_scatter(inout Payload ray, float3 normal, float3 pos, Material material) {
	// Figure out which side of the surface we are hitting
	normal = faceforward(normal, ray.direction, normal);
	float refraction_index = dot(ray.direction, normal) > 0. ? 1.0 / material.refraction_index : material.refraction_index;

	float3 input_direction = normalize(ray.direction);
	float3 output_ray_direction = refract(input_direction, normal, refraction_index);

	float cos_theta = min(dot(-input_direction, normal), 1.0);

	bool is_reflective = is_reflective_schlick(cos_theta, material.refraction_index);

	// If angle is less than the critical angle, reflection occurs instead and the function returns vec3(0.)
	if ((output_ray_direction.x == 0.0 && output_ray_direction.y == 0.0 && output_ray_direction.z == 0.0) || is_reflective) {
		output_ray_direction = reflect(input_direction, normal);
		ray.origin = pos + normal * EPSILON;
		ray.direction = output_ray_direction;
	} else {
		ray.origin = pos;
		ray.direction = output_ray_direction;
	}
	ray.attenuation = material.albedo;
}

[shader("closesthit")]
void main(inout Payload p, in Attributes attribs) {
	init_rng(DispatchRaysIndex().xy, ubo.width, ubo.ray_frame_num + uint(length(WorldRayDirection()) * 100));
	// Get the data corresponding to this object from the Instance Index inserted during TLAS creation
	Buffers resources = scene[InstanceID()];
	let vertices = (Vertex*)resources.vertices;
	let indices = (uint*)resources.indices;
	let materials = (Material*)resources.materials;
	let material_indices = (int*)resources.material_indices;

	let mat_id = material_indices[PrimitiveIndex()];
	let mat = materials[mat_id];

	// Indices of the triangle
	uint3 ind = uint3(indices[3 * PrimitiveIndex()], indices[3 * PrimitiveIndex() + 1], indices[3 * PrimitiveIndex() + 2]);

	// Vertex of the triangle
	Vertex v0 = vertices[ind.x];
	Vertex v1 = vertices[ind.y];
	Vertex v2 = vertices[ind.z];

	// Barycentric coordinates of the triangle
	const float3 barycentrics = float3(1.0f - attribs.bary.x - attribs.bary.y, attribs.bary.x, attribs.bary.y);

	// Computing the normal at hit position
	float3 normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
	normal = normalize(mul(WorldToObject4x3(), normal).xyz);  // Transforming the normal to world space

	// Computing the coordinates of the hit position
	float3 pos = v0.pos.xyz * barycentrics.x + v1.pos.xyz * barycentrics.y + v2.pos.xyz * barycentrics.z;
	pos = mul(float4(pos, 1.0), ObjectToWorld4x3()).xyz;  // Transforming the position to world space

	if (mat.alpha < rand_f32()) {
		// Dielectric
		dielectric_scatter(p, normal, pos, mat);
	} else {
		// Reflect
		reflect_ray(p, normal, pos, mat);
	}
}
