#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

// ----------------------- RNG Tools -----------------------
// A slightly modified version of the "One-at-a-Time Hash" function by Bob Jenkins.
// See https://www.burtleburtle.net/bob/hash/doobs.html
uint jenkins_hash(uint i) {
	uint x = i;
	x += x << 10u;
	x ^= x >> 6u;
	x += x << 3u;
	x ^= x >> 11u;
	x += x << 15u;
	return x;
}

uint init_rng(uvec2 pixel, uint width, uint frame_num) {
	// Seed the PRNG using the scalar index of the pixel and the current frame count.
	uint seed = (pixel.x + pixel.y * width) ^ jenkins_hash(frame_num);
	return jenkins_hash(seed);
}

// The 32-bit "xor" function from Marsaglia G., "Xorshift RNGs", Section 3.
uint xorshift32(inout uint seed) {
	uint x = seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	seed = x;
	return x;
}

// Returns a random float in the range [0...1]. This sets the floating point exponent to zero and
// sets the most significant 23 bits of a random 32-bit unsigned integer as the mantissa. That
// generates a number in the range [1, 1.9999999], which is then mapped to [0, 0.9999999] by
// subtraction. See Ray Tracing Gems II, Section 14.3.4.
float rand_f32(inout uint seed) {
	return uintBitsToFloat(0x3f800000u | (xorshift32(seed) >> 9u)) - 1.;
}

vec3 generate_random_unit_vector(inout uint seed) {
	return normalize(vec3(rand_f32(seed) * 2. - 1., rand_f32(seed) * 2. - 1., rand_f32(seed) * 2. - 1.));
}

struct Ray {
	vec3 radiance;
	vec3 attenuation;
	int done;
	vec3 origin;
	vec3 direction;
};

layout(location = 0) rayPayloadInEXT Ray ray;

hitAttributeEXT vec3 attribs;

struct Material {
	vec3 albedo;
	float alpha;
	float refraction_index;
	float smoothness;
	float emissivity;
	float emission_strength;
	vec3 emission_colour;
	uint padding;
};

struct Vertex {
	vec3 pos;
	uint pad1;
	vec3 nrm;
	uint pad2;	// Can we specify some sort of alignment to avoid needing this
};

struct ObjBuffers {
	uint64_t vertices;
	uint64_t indices;
	uint64_t materialIndices;
	uint64_t materials;
};

struct CameraUniforms {
	vec3 origin;
	float focal_distance;
	vec3 u;
	float vfov;
	vec3 v;
	float dof_scale;
	vec3 w;
};

// clang-format off
layout(buffer_reference, scalar) buffer Vertices {Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices {uint i[]; }; // Triangle indices
layout(buffer_reference, scalar) buffer Materials {Material m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MatIndices {int i[]; }; // Material ID for each triangle

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 3)    buffer _scene_desc { ObjBuffers i[]; } scene_desc;
// clang-format on

layout(set = 0, binding = 2) uniform UBO {
	CameraUniforms camera;
	uint frame_num;
	uint width;
	uint height;
}
ubo;

// const float EPSILON = 0.1;
const float EPSILON = 1e-2;

bool is_reflective_schlick(uint seed, float cosine, float refraction_index) {
	float r0 = (1. - refraction_index) / (1. + refraction_index);
	r0 = r0 * r0;
	return (r0 + (1. - r0) * pow((1. - cosine), 5.)) > rand_f32(seed);
}

void dielectric_scatter(inout uint seed, vec3 normal, vec3 pos, Material mat) {
	// Figure out which side of the surface we are hitting
	normal = faceforward(normal, ray.direction, normal);
	float refraction_index = dot(ray.direction, normal) > 0. ? 1.0 / mat.refraction_index : mat.refraction_index;

	vec3 input_direction = normalize(ray.direction);
	vec3 output_ray_direction = refract(input_direction, normal, refraction_index);

	float cos_theta = min(dot(-input_direction, normal), 1.0);

	bool is_reflective = is_reflective_schlick(seed, cos_theta, mat.refraction_index);

	// If angle is less than the critical angle, reflection occurs instead and the function returns vec3(0.)
	if ((output_ray_direction.x == 0.0 && output_ray_direction.y == 0.0 && output_ray_direction.z == 0.0) || is_reflective) {
		output_ray_direction = reflect(input_direction, normal);
		ray.origin = pos + normal * EPSILON;
		ray.direction = output_ray_direction;
	} else {
		ray.origin = pos;
		ray.direction = output_ray_direction;
	}
	ray.attenuation = mat.albedo;
}

void reflect_ray(inout uint seed, vec3 normal, vec3 pos, Material mat) {
	vec3 lambertian_reflection = normal + generate_random_unit_vector(seed);
	// vec3 lambertian_reflection = normal;
	vec3 metallic_reflection = reflect(gl_WorldRayDirectionEXT, normal);
	vec3 reflected = mix(lambertian_reflection, metallic_reflection, mat.smoothness);
	// Bump the start of the reflected ray a little bit off the surface to
	// try to minimize self intersections due to floating point errors
	ray.origin = pos + normal * EPSILON;
	ray.direction = reflected;
	ray.attenuation = mat.albedo;
}

void main() {
	// Something unique to this ray and something unique to this timestamp
	uint seed = init_rng(gl_LaunchIDEXT.xy, ubo.width, ubo.frame_num + uint(length(gl_WorldRayDirectionEXT.xy) * 100));
	// When contructing the TLAS, we stored the model id in InstanceCustomIndexEXT, so the
	// the instance can quickly have access to the data

	// Object data
	ObjBuffers objResource = scene_desc.i[gl_InstanceCustomIndexEXT];
	MatIndices matIndices = MatIndices(objResource.materialIndices);
	Materials materials = Materials(objResource.materials);
	Indices indices = Indices(objResource.indices);
	Vertices vertices = Vertices(objResource.vertices);

	// Retrieve the material used on this triangle 'PrimitiveID'
	int mat_idx = matIndices.i[gl_PrimitiveID];
	Material mat = materials.m[mat_idx];  // Material for this triangle

	// Indices of the triangle
	uvec3 ind = uvec3(indices.i[3 * gl_PrimitiveID], indices.i[3 * gl_PrimitiveID + 1], indices.i[3 * gl_PrimitiveID + 2]);

	// Vertex of the triangle
	Vertex v0 = vertices.v[ind.x];
	Vertex v1 = vertices.v[ind.y];
	Vertex v2 = vertices.v[ind.z];

	// Barycentric coordinates of the triangle
	const vec3 barycentrics = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);

	// Computing the normal at hit position
	vec3 normal = v0.nrm.xyz * barycentrics.x + v1.nrm.xyz * barycentrics.y + v2.nrm.xyz * barycentrics.z;
	normal = normalize(vec3(normal.xyz * gl_WorldToObjectEXT));	 // Transforming the normal to world space

	// Computing the coordinates of the hit position
	vec3 pos = v0.pos.xyz * barycentrics.x + v1.pos.xyz * barycentrics.y + v2.pos.xyz * barycentrics.z;
	pos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));  // Transforming the position to world space

	vec3 debug_colour = vec3(1.0, 0.4, 0.1);

	if (mat.alpha < rand_f32(seed)) {
		// Dielectric
		dielectric_scatter(seed, normal, pos, mat);
	} else {
		// Reflect
		reflect_ray(seed, normal, pos, mat);
	}
}
