module scatter;
__import rng;
__import structs;

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

public void scatter(inout Payload ray, float3 pos, float3 normal, Material material) {
	if (material.alpha < rand_f32()) {
		// Dielectric
		dielectric_scatter(ray, normal, pos, material);
	} else {
		// Reflect
		reflect_ray(ray, normal, pos, material);
	}
}
