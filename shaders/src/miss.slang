__import structs;

[[vk::push_constant]]
ConstantBuffer<RayPushConstants> pc;
[[vk::binding(0, 3)]]
SamplerCube<float4> textures[];

[shader("miss")]
void main(inout Payload p) {
	if (pc.skybox_index != 0) {
		let dir = WorldRayDirection();
		p.radiance = textures[pc.skybox_index].SampleLevel(dir, 1.0).rgb;
	} else{
		let t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
		p.radiance = (1. - t) * float3(1.) + t * float3(0.3, 0.5, 1.);
	}

	// p.radiance /= 100.0;

	// p.radiance = float3(0.4, 0.1, 0.3);
	// p.radiance = float3(0.3, 0.5, 1.0);
	// p.radiance = float3(0.0);
	// p.radiance = float3(1.0, 1.0, 1.0);
	// p.attenuation = float3(1.0, 1.0, 1.0);
	p.done = 1;
}
