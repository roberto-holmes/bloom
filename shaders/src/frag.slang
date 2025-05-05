struct PushConstants {
	uint32_t accumulated_frames;
}

[[vk::binding(0)]]
RWTexture2D<float4> read_image;
// [[vk::binding(1)]]
// ConstantBuffer<UBO> ubo;
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("fragment")]
float4 main(float4 pos: SV_Position)
	: SV_Target {
	// Only run for pos 0, 0
	// if (!any(uint2(pos.xy))) { printf("v+ %u \t%v3f\n", pc.accumulated_frames, sample.rgb); }
	float3 sample = read_image[uint2(pos.xy)].rgb;
	sample /= pc.accumulated_frames;

	return float4(sample, 1.0);
}
