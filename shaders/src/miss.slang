__import structs;

[shader("miss")]
void main(inout Payload p) {
	let t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
	p.radiance = (1. - t) * float3(1.) + t * float3(0.3, 0.5, 1.);

	// p.radiance = float3(0.4, 0.1, 0.3);
	// p.radiance = float3(0.3, 0.5, 1.0);
	// p.radiance = float3(0.0);
	// p.radiance = float3(1.0, 1.0, 1.0);
	p.attenuation = float3(1.0, 1.0, 1.0);
	p.done = 1;
}
