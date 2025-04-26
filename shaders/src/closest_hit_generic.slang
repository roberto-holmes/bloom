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
