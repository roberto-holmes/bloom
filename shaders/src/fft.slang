struct PushConstants {
	float wind_speed_x;
	float wind_speed_z;
	float wind_fetch_x;
	float wind_fetch_z;
	uint64_t time_stamp_ns;
}

[[vk::binding(0)]]
RWTexture2D<float4> output;
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint3 id: SV_DispatchThreadID) {
	// float4 pixel = float4(1.0, 0.5, 0.5, 1.0);
	// output[id.xy] = pixel;
	output[id.xy] = float4(id.x / 32, id.y / 32, 0.0, 1.0);
}
