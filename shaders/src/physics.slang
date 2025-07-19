__import structs;
__import constants;
__import rng;
__import helpers;

struct Instance {
	AccelerationStructureInstanceKHR asi;
	uint32_t id;
	uint32_t pad0;
	uint32_t pad1;
	uint32_t pad2;
}

struct AccelerationStructureInstanceKHR {
	float4 row1;
	float4 row2;
	float4 row3;
	uint32_t index_mask;
	uint32_t offset_flags;
	uint64_t reference;
}

struct PushConstants {
	uint64_t time_stamp_ns;
	uint64_t* instances;
	uint32_t instance_count;
}

// [[vk::binding(0, 0)]]
// RWTexture2D<float4> ocean_map;	// TODO: Consider replacing with a float2 texture
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(8, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
	if (id.x >= pc.instance_count) {
		return;
	}

	Instance* inst = ((Instance*)*(pc.instances + id.x));

	// if (pc.time_stamp_ns < 25_000_000) {
	// 	printf("%u: %u [%lu]\n", id.x, inst.id, (uint64_t)inst);
	// }

	if (inst.id == 7) {
		inst.asi.row2 = float4(0.0, 1.0, 0.0, pc.time_stamp_ns / float(1000_000_000));
	}
}
