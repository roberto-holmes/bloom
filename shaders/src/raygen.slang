__import rng;
__import structs;
__import quaternion;
__import constants;

[[vk::push_constant]]
ConstantBuffer<RayPushConstants> pc;

// (Binding, Set)
[[vk::binding(0, 0)]]
RaytracingAccelerationStructure tlas;
[[vk::binding(0, 1)]]
RWTexture2D<float4> images[];
[[vk::binding(0, 2)]]
RWTexture2D<float4> ocean;
[[vk::binding(0, 3)]]
Sampler2D<float4> textures[];

[shader("raygeneration")]
void main() {
	uint3 LaunchID = DispatchRaysIndex();
	uint3 LaunchSize = DispatchRaysDimensions();

	UBO ubo = *pc.ubo;

	if (pc.ubo.camera.enabled == false) {
		images[pc.frame_index][int2(LaunchID.xy)] = float4(0.0);
		return;
	}

	// if (all(LaunchID.xy == uint2(10, 10)) && pc.ubo.ray_frame_num < 10) {
	// if (all(LaunchID.xy == uint2(10, 10))) {
	// printf("%u\n", LaunchID.x);
	// printf("%u | TLAS: %lu | %lu\n", pc.ubo.ray_frame_num, (uint64_t)pc.tlas, (*(uint64_t*)pc.tlas));
	// }

	const float aspect_ratio = float(pc.ubo.width) / float(pc.ubo.height);

	init_rng(DispatchRaysIndex().xy, pc.ubo.width, pc.ubo.ray_frame_num + DispatchRaysIndex().x + DispatchRaysIndex().y + pc.ubo.random_num);

	// Start the rays from the center of each pixel with a random offset to perform antialiasing
	const float2 pixelCenter = float2(LaunchID.xy) + float2(0.5, 0.5) + random_vector_in_unit_disk().xy;
	// Coordinates are now between 0 and 1 (0,0 at top left, 1,1 at bottom right)
	const float2 uv_in = pixelCenter / float2(LaunchSize.xy);
	// Map coordinates so that 0,0 is in the center, y-up, x-right, screen height is 2 units
	float2 uv = (uv_in * 2.0 - float2(1.0)) * float2(aspect_ratio, -1.0);

	// Fix the perspective to the field of view from the ubo
	float viewport_scale_factor = 2. * pc.ubo.camera.focal_distance * tan(pc.ubo.camera.vfov / 2.);
	uv *= viewport_scale_factor;

	// float3x3 camera_rotation = transpose(float3x3(pc.ubo.camera.u, pc.ubo.camera.v, pc.ubo.camera.w));
	let dof_offset = rotate_vector(random_vector_in_unit_disk(), pc.ubo.camera.q) * pc.ubo.camera.dof_scale;
	let direction = rotate_vector(float3(uv, pc.ubo.camera.focal_distance), pc.ubo.camera.q) - dof_offset;

	Payload payload;
	payload.origin = pc.ubo.camera.origin + dof_offset;
	payload.direction = normalize(direction);

	// if (LaunchID.x == 0 && LaunchID.y == 0 && pc.ubo.ray_frame_num % 100 == 0) {
	// 	printf("o: %v3f | ", pc.ubo.camera.origin);
	// 	printf("q: %v4f | ", pc.ubo.camera.q);
	// 	printf("ori: %v3f | ", payload.origin);
	// 	printf("dir: %v3f\n", payload.direction);
	// }

	// Copy data from previous sample
	float3 sample = images[(pc.frame_index + 1) % 2][int2(LaunchID.xy)].rgb;
	float3 throughput = 1.0;

	for (int depth = 0; depth < 16; depth++) {
		RayDesc rayDesc;
		rayDesc.Direction = payload.direction;
		rayDesc.Origin = payload.origin;
		rayDesc.TMin = 0.001;
		rayDesc.TMax = 10000.0;

		//! There is likely an issue with pc.tlas; this line causes the shader to crash
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
	// sample += throughput * payload.radiance;

	// let ocean_sample = ocean[int2(LaunchID.xy)].r * 1024 * 1024;

	// let o = -ocean[int2(LaunchID.xy)] * 1024 * 1024 * 5 * 5;

	// let normal = normalize(float3(o.b, 1, o.a));

	// if (all(LaunchID.xy == uint2(10, 10))) {
	// 	printf("n: %v3f\n", normal);
	// }
	// if (LaunchID.x > 1024) {
	// 	let pos = int2(LaunchID.x - 1024, LaunchID.y);
	// 	let ocean_sample = ocean[pos].r * 1024 * 1024;
	// write_image[int2(LaunchID.xy)] = float4(ocean_sample, 0.0, ocean_sample, 1.0);

	// } else {
	// 	write_image[int2(LaunchID.xy)] = float4(normal, 1.0);
	// }

	// sample += float3(0.1, 0.4, 1.0);
	// TODO: Figure out how to access the mipmaps/lods (or if they are even correct in the first place)
	sample += textures[1].SampleLevel(float2(LaunchID.xy)/1024.0, 1.0).rgb;

	images[pc.frame_index][int2(LaunchID.xy)] = float4(sample, 0.0);
	// images[pc.frame_index][int2(LaunchID.xy)] = float4(0.1, 0.4, 1.0, 1.0); // Light Blue
	// images[pc.frame_index][int2(LaunchID.xy)] = float4(1.0, 0.1, 1.0, 1.0); // Almost Magenta
	// images[pc.frame_index][int2(LaunchID.xy)] = float4(0.1, 0.4, 1.0, 1.0);
	// write_image[int2(LaunchID.xy)] = ocean[int2(0, 0)];
	// images[0][int2(LaunchID.xy)] = ocean[int2(LaunchID.xy)] * 1024;
	// write_image[int2(LaunchID.xy)] = float4(ocean_sample, 0.0, ocean_sample, 1.0);

	// images[pc.frame_index][int2(LaunchID.xy)] = float4(ocean[int2(LaunchID.xy)].b * 1024, 0.0, 0.0, 1.0);

	// float2 normal_map = ocean[int2(LaunchID.xy)].ba * 1024 * 1024;
	// let normal = normalize(float3(-normal_map.x, 1, -normal_map.y));
	// write_image[int2(LaunchID.xy)] = float4(normal, 1.0);
}
