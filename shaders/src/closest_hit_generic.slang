__import rng;
__import structs;
__import scatter;

[[vk::binding(0)]]
RaytracingAccelerationStructure tlas;
// 1 and 2 are the storage images
[[vk::binding(3)]]
ConstantBuffer<UBO> ubo;
[[vk::binding(4)]]
StructuredBuffer<Buffers> scene;
[[vk::binding(5)]]
StructuredBuffer<Material> materials;

float3 point_on_ray(float t) {
	return WorldRayOrigin() + t * WorldRayDirection();
}

[shader("closesthit")]
void main(inout Payload ray) {
	init_rng(DispatchRaysIndex().xy, ubo.width, ubo.ray_frame_num + uint(length(WorldRayDirection()) * 100));
	// Get the primitive type from RayTCurrent (The second value from the ReportHit function)
	uint object_type = *(uint*)scene[InstanceID()].primitive;

	Material mat;

	if (HitKind() == ObjectType::Sphere) {
		let sphere = (Sphere*)scene[InstanceID()].primitive;
		mat = materials[sphere.material];

		// Get position on surface
		let pos = point_on_ray(RayTCurrent());

		// Get normal
		let normal = (pos - sphere.center) / sphere.radius;

		scatter(ray, pos, normal, mat);
	} else {
		ray.attenuation = float3(0.5, 0.2, RayTCurrent() / 2.0);
		ray.radiance = float3(1.0, 1.0, 1.0);
		ray.done = 1;
		return;
	}
}
