__import rng;
__import structs;
__import scatter;
__import helpers;

[[vk::binding(0)]]
RaytracingAccelerationStructure tlas;
// 1 and 2 are the storage images
[[vk::binding(3)]]
ConstantBuffer<UBO> ubo;
[[vk::binding(4)]]
StructuredBuffer<Buffers> scene;
[[vk::binding(5)]]
StructuredBuffer<Material> materials;
[[vk::binding(6, 0)]]
RWTexture2D<float4> ocean_map;

[shader("closesthit")]
void main(inout Payload ray, in CustomAttributes att) {
	init_rng(DispatchRaysIndex().xy, ubo.width, ubo.ray_frame_num + uint(length(WorldRayDirection()) * 100) + ubo.random_num);
	// Get the primitive type from RayTCurrent (The second value from the ReportHit function)

	Material mat;

	if (HitKind() == ObjectType::Sphere) {
		let sphere = (Sphere*)scene[InstanceID()].primitive;
		mat = materials[sphere.material];
		let center = mul(ObjectToWorld(), float4(0.0, 0.0, 0.0, 1.0));

		// Get position on surface
		let pos = point_on_ray(RayTCurrent());

		// Get normal
		let normal = (pos - center) / sphere.radius;

		scatter(ray, pos, normal, mat);
	} else if (HitKind() == ObjectType::Ocean) {
		let ocean = (Ocean*)scene[InstanceID()].primitive;
		mat = materials[ocean.material];
		let pos = point_on_ray(RayTCurrent());
		float4 ocean_point = get_ocean_point(ocean, pos.xz);
		// float2 normal_map = get_ocean_point(ocean, pos.xz).ba * 1024 * 1024 * 5 * 5;

		// let normal = normalize(float3(-normal_map.x, 1, -normal_map.y));

		// printf("pos: %v3f, point: %v4f\n", pos, ocean_point);

		// scatter(ray, pos, normal, mat);
		ray.attenuation = float3(ocean_point.r, 0.0, ocean_point.g);
		// ray.attenuation = float3(0.5, 0.2, RayTCurrent() / 2.0);
		ray.radiance = float3(1.0, 1.0, 1.0);
		ray.done = 1;
		return;
	} else if (HitKind() > ObjectType::Lentil) {
		// var hit_type = HitKind() - ObjectType::Lentil;
		let lentil = (Lentil*)scene[InstanceID()].primitive;
		mat = materials[lentil.material];
		float3 center = mul(ObjectToWorld(), float4(0.0, 0.0, 0.0, 1.0));
		float3 axis = normalize(mul(ObjectToWorld(), float4(1.0, 0.0, 0.0, 0.0)));

		// if (ubo.ray_frame_num == 1) {
		// 	printf("%u\n", att.hit_type);
		// }
		let hit_type = att.hit_type;

		// Get position on surface
		let pos = point_on_ray(RayTCurrent());

		// Get normal
		float3 normal;
		// TODO: Normal for an aspherical surface

		if (hit_type == 1) {
			// Cylinder surface
			normal = -normalize(cross(cross((pos - center), axis), axis));	// For a cylinder
																			// ray.attenuation = float3(0.1, 0.1, 1.0);
		} else if (hit_type == 2) {
			// Lense surface A
			normal = (pos - (center + axis * (lentil.length - lentil.curvature_a))) / lentil.curvature_a;
			// ray.attenuation = float3(0.2, 1.0, 0.2);
		} else if (hit_type == 3) {
			// Lense surface B
			normal = (pos - (center - axis * (lentil.length - lentil.curvature_b))) / lentil.curvature_b;
			// ray.attenuation = float3(1.0, 0.2, 0.2);
		} else {
			ray.attenuation = float3(0.0, 0.9, RayTCurrent() / 2.0);
			// ray.radiance = float3(1.0, 1.0, 1.0);
			// ray.done = 1;
		}

		scatter(ray, pos, normal, mat);
	} else {
		ray.attenuation = float3(0.5, 0.2, RayTCurrent() / 2.0);
		ray.radiance = float3(1.0, 1.0, 1.0);
		ray.done = 1;
		return;
	}
}

float4 get_ocean_point(const Ocean* ocean, float2 coordinates) {
	// TODO: Combine with intersection::get_ocean_point
	coordinates *= 100;
	float scaling_factor = ocean.size * ocean.size;

	// Tile coordinates over size
	coordinates %= float2(ocean.size);
	coordinates += float2(ocean.size);
	coordinates %= float2(ocean.size);

	let floored = uint2(floor(coordinates));

	// Bilinear Interpolation
	// TODO: Consider a better algorithm
	// Bottom left
	let h_11 = ocean_map[uint2(floored.x, floored.y + 1) % uint2(ocean.size)] * scaling_factor;
	// Top right
	let h_22 = ocean_map[uint2(floored.x + 1, floored.y) % uint2(ocean.size)] * scaling_factor;
	// Top left
	let h_21 = ocean_map[floored] * scaling_factor;
	// Bottom right
	let h_12 = ocean_map[(floored + uint(1)) % uint2(ocean.size)] * scaling_factor;

	let x_lerp1 = lerp(abs(h_11), abs(h_12), coordinates.x - floored.x);
	let x_lerp2 = lerp(abs(h_21), abs(h_22), coordinates.x - floored.x);
	let y_lerp = lerp(x_lerp2, x_lerp1, coordinates.y - floored.y);

	return y_lerp;
}
