__import structs;
__import constants;

[[vk::binding(0)]]
RWTexture2D<float4> read_image;
[[vk::binding(1)]]
ConstantBuffer<UBO> ubo;

[shader("fragment")]
float4 main(float4 pos: SV_Position)
	: SV_Target {
	// Only run for pos 0, 0
	// if (!any(uint2(pos.xy))) {	printf("v-: %u %u - %v3f\n", ubo.ray_frame_num, ubo.ray_frame_num_out, sample.rgb); }
	float3 sample = read_image[uint2(pos.xy)].rgb;
	if (ubo.ray_frame_num_out >= MAX_SAMPLES) {
		sample /= MAX_SAMPLES;
	} else if (ubo.ray_frame_num_out > 1) {
		sample /= ubo.ray_frame_num_out;
	}

	return float4(sample, 1.0);
}
