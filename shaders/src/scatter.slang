module scatter;
__import rng;
__import structs;
__import helpers;
__import constants;

bool is_reflective_schlick(float cosine, float refractive_index) {
	float r0 = (1. - refractive_index) / (1. + refractive_index);
	r0 = r0 * r0;
	return (r0 + (1. - r0) * pow((1. - cosine), 5.)) > rand_f32();
}

void reflect_ray(inout Payload ray, float3 pos, float3 normal, Material material) {
	let lambertian_reflection = normalize(normal + generate_random_unit_vector());
	let metallic_reflection = reflect(WorldRayDirection(), normal);
	ray.direction = lerp(lambertian_reflection, metallic_reflection, material.smoothness);
	// Bump the start of the reflected ray a little bit off the surface to
	// try to minimize self intersections due to floating point errors
	ray.origin = pos + normal * EPSILON;
	ray.attenuation = material.albedo;
}

void dielectric_scatter(inout Payload ray, float3 pos, float3 normal, Material material) {
	// Figure out which side of the surface we are hitting

	float relative_refractive_index = dot(ray.direction, normal) > 0. ? material.refractive_index : 1.0 / material.refractive_index;

	normal = faceforward(normal, ray.direction, normal);

	float3 input_direction = normalize(ray.direction);
	float3 output_ray_direction = refract(input_direction, normal, relative_refractive_index);

	float cos_theta = min(dot(-input_direction, normal), 1.0);

	bool is_reflective = is_reflective_schlick(cos_theta, material.refractive_index);

	// If angle is less than the critical angle, reflection occurs instead and the function returns vec3(0.)
	if ((output_ray_direction.x == 0.0 && output_ray_direction.y == 0.0 && output_ray_direction.z == 0.0) || is_reflective) {
		output_ray_direction = reflect(input_direction, normal);
		ray.origin = pos + normal * EPSILON;
		ray.direction = output_ray_direction;
	} else {
		ray.origin = pos;
		ray.direction = output_ray_direction;
		ray.refractive_index *= relative_refractive_index;
	}
	ray.attenuation = material.albedo;
}

public void scatter(inout Payload ray, float3 pos, float3 normal, Material material) {
	if (material.alpha < rand_f32()) {
		// Dielectric
		dielectric_scatter(ray, pos, normal, material);
	} else {
		// Reflect
		reflect_ray(ray, pos, normal, material);
	}

	if (material.emissivity > rand_f32()) {
		ray.is_light = true;
		ray.radiance = material.emission_colour * material.emission_strength;
	}
}
