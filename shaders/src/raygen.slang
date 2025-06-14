__import rng;
__import structs;
__import quaternion;
__import constants;

[[vk::binding(0)]]
RaytracingAccelerationStructure tlas;
[[vk::binding(1)]]
RWTexture2D<float4> read_image;
[[vk::binding(2)]]
RWTexture2D<float4> write_image;
[[vk::binding(3)]]
ConstantBuffer<UBO> ubo;

[[vk::binding(6)]]
RWTexture2D<float4> ocean;

[shader("raygeneration")]
void main() {
	uint3 LaunchID = DispatchRaysIndex();
	uint3 LaunchSize = DispatchRaysDimensions();

	if (ubo.camera.enabled == false) {
		write_image[int2(LaunchID.xy)] = float4(0.0);
		return;
	}

	const float aspect_ratio = float(ubo.width) / float(ubo.height);

	init_rng(DispatchRaysIndex().xy, ubo.width, ubo.ray_frame_num + DispatchRaysIndex().x + DispatchRaysIndex().y + ubo.random_num);

	// Start the rays from the center of each pixel with a random offset to perform antialiasing
	const float2 pixelCenter = float2(LaunchID.xy) + float2(0.5, 0.5) + random_vector_in_unit_disk().xy;
	// Coordinates are now between 0 and 1 (0,0 at top left, 1,1 at bottom right)
	const float2 uv_in = pixelCenter / float2(LaunchSize.xy);
	// Map coordinates so that 0,0 is in the center, y-up, x-right, screen height is 2 units
	float2 uv = (uv_in * 2.0 - float2(1.0)) * float2(aspect_ratio, -1.0);

	// Fix the perspective to the field of view from the ubo
	float viewport_scale_factor = 2. * ubo.camera.focal_distance * tan(ubo.camera.vfov / 2.);
	uv *= viewport_scale_factor;

	// float3x3 camera_rotation = transpose(float3x3(ubo.camera.u, ubo.camera.v, ubo.camera.w));
	let dof_offset = rotate_vector(random_vector_in_unit_disk(), ubo.camera.q) * ubo.camera.dof_scale;
	let direction = rotate_vector(float3(uv, ubo.camera.focal_distance), ubo.camera.q) - dof_offset;

	Payload payload;
	payload.origin = ubo.camera.origin + dof_offset;
	payload.direction = normalize(direction);

	// if (LaunchID.x == 0 && LaunchID.y == 0 && ubo.ray_frame_num % 100 == 0) {
	// 	printf("o: %v3f | ", ubo.camera.origin);
	// 	printf("q: %v4f | ", ubo.camera.q);
	// 	printf("ori: %v3f | ", payload.origin);
	// 	printf("dir: %v3f\n", payload.direction);
	// }

	float3 sample = read_image[int2(LaunchID.xy)].rgb;
	float3 throughput = 1.0;

	for (int depth = 0; depth < 16; depth++) {
		RayDesc rayDesc;
		rayDesc.Direction = payload.direction;
		rayDesc.Origin = payload.origin;
		rayDesc.TMin = 0.001;
		rayDesc.TMax = 10000.0;

		TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, rayDesc, payload);
		throughput *= payload.attenuation;

		if (payload.is_light) {
			sample += throughput * payload.radiance;
			payload.is_light = false;
		}

		if (payload.done == 1 || length(payload.attenuation) < 0.1) {
			// if (payload.done == 1) {
			break;
		}
	}
	sample += throughput * payload.radiance;
	write_image[int2(LaunchID.xy)] = float4(sample, 0.0);
	// write_image[int2(LaunchID.xy)] = float4(0.1, 0.4, 1.0, 1.0);
	// write_image[int2(LaunchID.xy)] = ocean[int2(0, 0)];
	// write_image[int2(LaunchID.xy)] = ocean[int2(LaunchID.xy)];
}
